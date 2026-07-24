
import os
from datetime import datetime
# from typing import List, Tuple, Optional, Dict, Any
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl
import rasterio as rio
from tqdm import tqdm
import glob
# import warnings
# import rasterio.windows
from matplotlib.colors import LightSource
from pyproj import Transformer

# Import utilities
from elevation_utils import (
    wgs84_to_3413, find_and_unzip, get_elevation_window,
    read_elevation_from_compressed, extract_elevation_profile, 
    extract_elevation_profile_compressed, transect_from_mosaic
)
from analysis import (
    extract_date_obj, extract_date_label, extract_year,
    filter_outlier_profiles, process_elevation_profiles
)
from config import (
    OUTPUT_DIR, DEFAULT_NUM_SAMPLES, DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_TYPE, COREG_PARAMS, get_output_path
)


def detect_lake_boundaries(all_profiles, 
                          min_depth=1.5,                # Minimum depression depth (m)
                          min_span_meters=200,           # Minimum lake width
                          max_span_meters=6000,          # Maximum lake width
                          min_profiles_showing=3,        # Profiles that must show feature
                          smooth_window=3,               # Smoothing for profiles
                          prominence_threshold=0.3,      # Relative prominence for peak detection
                          verbose=True):
    """
    Detect potential subglacial lake boundaries from elevation profile divergence.
    
    Lakes are identified as regions where some profiles show a localized 
    depression (or bulge) that is absent in other profiles. This is the 
    characteristic signature of active subglacial lakes.
    
    The key insight: lakes cause some profiles to diverge LOCALLY from the 
    ensemble. We detect this by:
    1. Computing the median profile as reference
    2. Finding where individual profiles deviate from the median
    3. Looking for consistent spatial patterns across multiple profiles
    
    Parameters
    ----------
    all_profiles : dict
        Profile data (filtered)
    min_depth : float
        Minimum vertical deviation from median to flag (meters)
    min_span_meters, max_span_meters : float
        Allowed lake width range
    min_profiles_showing : int
        Minimum number of profiles that must show the feature
    smooth_window : int
        Smoothing window size for individual profiles
    prominence_threshold : float
        Relative threshold for peak-finding (0-1)
    verbose : bool
        Print detection details
        
    Returns
    -------
    dict with lake_boundaries, anomaly_map, etc.
    """
    if not all_profiles["profiles"]:
        if verbose:
            print("No profiles to analyze")
        return {'lake_boundaries': [], 'anomaly_map': None}
    
    profiles_sorted = sorted(
        all_profiles["profiles"],
        key=lambda x: extract_date_obj(x["metadata"]["dem_name"])
    )
    
    n_profiles = len(profiles_sorted)
    distances = profiles_sorted[0]['transect']
    n_points = len(distances)
    dx = np.mean(np.diff(distances))
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"SUBSGLACIAL LAKE DETECTION")
        print(f"{'='*80}")
        print(f"Profiles: {n_profiles}")
        print(f"Transect points: {n_points}")
        print(f"Spatial resolution: {dx:.1f} m")
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Build elevation matrix and compute ensemble statistics
    # ──────────────────────────────────────────────────────────────────────
    
    elevation_matrix = np.full((n_profiles, n_points), np.nan)
    
    for i, profile in enumerate(profiles_sorted):
        elevations = np.array(profile['profile_values'])
        valid = ~np.isnan(elevations)
        if np.sum(valid) > 10:
            elevation_matrix[i, :] = np.interp(
                distances, profile['transect'][valid], elevations[valid],
                left=np.nan, right=np.nan
            )
    
    # Compute median profile (robust reference) and IQR
    median_profile = np.nanmedian(elevation_matrix, axis=0)
    q25 = np.nanpercentile(elevation_matrix, 25, axis=0)
    q75 = np.nanpercentile(elevation_matrix, 75, axis=0)
    iqr = q75 - q25
    
    if verbose:
        print(f"Median elevation range: {np.nanmin(median_profile):.0f} – {np.nanmax(median_profile):.0f} m")
        print(f"Median IQR: {np.nanmedian(iqr):.2f} m")
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Compute deviation of each profile from the median
    # ──────────────────────────────────────────────────────────────────────
    
    deviation_matrix = np.full((n_profiles, n_points), np.nan)
    
    for i in range(n_profiles):
        valid = ~np.isnan(elevation_matrix[i, :])
        if np.sum(valid) > 10:
            # Smooth the profile first
            if smooth_window > 1:
                kernel = np.ones(smooth_window) / smooth_window
                smoothed = np.convolve(elevation_matrix[i, :], kernel, mode='same')
                deviation_matrix[i, :] = smoothed - median_profile
            else:
                deviation_matrix[i, :] = elevation_matrix[i, :] - median_profile
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 3: Find where profiles consistently deviate (anomaly score)
    # ──────────────────────────────────────────────────────────────────────
    
    # Count how many profiles show significant deviation at each point
    # AND whether the deviation is consistently in the same direction
    n_deviating = np.zeros(n_points)
    mean_deviation = np.zeros(n_points)
    deviation_sign_consistency = np.zeros(n_points)
    
    for j in range(n_points):
        devs = deviation_matrix[:, j]
        valid_devs = devs[~np.isnan(devs)]
        
        if len(valid_devs) >= min_profiles_showing:
            # Count profiles with |deviation| > threshold
            sig_deviating = np.abs(valid_devs) > min_depth
            n_deviating[j] = np.sum(sig_deviating)
            
            if n_deviating[j] > 0:
                mean_deviation[j] = np.mean(valid_devs[sig_deviating])
                
                # Check sign consistency: are deviations mostly same sign?
                signs = np.sign(valid_devs[sig_deviating])
                if len(signs) > 0:
                    majority_sign = np.sign(np.mean(signs))
                    consistency = np.sum(signs == majority_sign) / len(signs)
                    deviation_sign_consistency[j] = consistency
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 4: Create combined anomaly score
    # ──────────────────────────────────────────────────────────────────────
    
    # Anomaly = number of deviating profiles × mean deviation magnitude × sign consistency
    norm_n = n_deviating / max(n_profiles, 1)
    norm_mag = np.abs(mean_deviation) / max(np.max(np.abs(mean_deviation)), 0.01)
    
    anomaly_scores = norm_n * norm_mag * deviation_sign_consistency
    
    # Smooth the anomaly
    if smooth_window > 1:
        kernel = np.ones(max(3, smooth_window)) / max(3, smooth_window)
        anomaly_scores = np.convolve(anomaly_scores, kernel, mode='same')
    
    anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0)
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 5: Find peaks in the anomaly score
    # ──────────────────────────────────────────────────────────────────────
    
    from scipy.signal import find_peaks
    
    # Dynamic threshold based on anomaly distribution
    positive = anomaly_scores[anomaly_scores > 0]
    if len(positive) > 10:
        threshold = np.percentile(positive, 80)
    else:
        threshold = 0.05
    
    # Find peaks
    peaks, properties = find_peaks(
        anomaly_scores,
        height=threshold,
        distance=int(min_span_meters / dx),
        prominence=threshold * prominence_threshold,
        width=int(min_span_meters / dx / 2),
    )
    
    if verbose:
        print(f"\nAnomaly threshold: {threshold:.4f}")
        print(f"Peaks found: {len(peaks)}")
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 6: For each peak, find the full extent (where anomaly drops)
    # ──────────────────────────────────────────────────────────────────────
    
    lake_candidates = []
    
    for peak_idx in peaks:
        peak_score = anomaly_scores[peak_idx]
        
        # Find left boundary: where anomaly drops to 30% of peak
        left_idx = peak_idx
        while left_idx > 0:
            if anomaly_scores[left_idx] < 0.3 * peak_score:
                break
            left_idx -= 1
        
        # Find right boundary
        right_idx = peak_idx
        while right_idx < n_points - 1:
            if anomaly_scores[right_idx] < 0.3 * peak_score:
                break
            right_idx += 1
        
        span = distances[right_idx] - distances[left_idx]
        
        # Filter by size
        if span < min_span_meters or span > max_span_meters:
            if verbose:
                print(f"  Peak at {distances[peak_idx]:.0f}m: span {span:.0f}m - rejected (size)")
            continue
        
        # Calculate statistics for this region
        n_affected = np.max(n_deviating[left_idx:right_idx+1])
        mean_dev = np.mean(np.abs(mean_deviation[left_idx:right_idx+1]))
        
        if n_affected < min_profiles_showing:
            if verbose:
                print(f"  Peak at {distances[peak_idx]:.0f}m: {n_affected:.0f} profiles - rejected")
            continue
        
        # Refine boundaries using gradient
        grad = np.abs(np.gradient(anomaly_scores))
        
        # Left: find where gradient peaks near left boundary
        left_search = max(0, left_idx - int(0.1 * span / dx))
        right_search = min(peak_idx, left_idx + int(0.3 * span / dx))
        if right_search > left_search + 1:
            left_boundary = left_search + np.argmax(grad[left_search:right_search])
        else:
            left_boundary = left_idx
        
        # Right: find where gradient peaks near right boundary
        left_search_r = max(peak_idx, right_idx - int(0.3 * span / dx))
        right_search_r = min(n_points - 1, right_idx + int(0.1 * span / dx))
        if right_search_r > left_search_r + 1:
            right_boundary = left_search_r + np.argmax(grad[left_search_r:right_search_r])
        else:
            right_boundary = right_idx
        
        refined_span = distances[right_boundary] - distances[left_boundary]
        
        if refined_span < min_span_meters:
            continue
        
        # Calculate confidence
        n_score = min(n_affected / n_profiles, 1.0)
        mag_score = min(mean_dev / (min_depth * 3), 1.0)
        consistency_score = np.mean(deviation_sign_consistency[left_boundary:right_boundary+1])
        
        confidence = 0.3 * n_score + 0.3 * mag_score + 0.4 * consistency_score
        
        lake_candidates.append({
            'start_distance': distances[left_boundary],
            'end_distance': distances[right_boundary],
            'start_index': int(left_boundary),
            'end_index': int(right_boundary),
            'span_meters': refined_span,
            'confidence': confidence,
            'mean_deviation': mean_dev,
            'n_profiles_affected': int(n_affected),
            'peak_anomaly': float(peak_score),
            'peak_distance': distances[peak_idx],
        })
        
        if verbose:
            direction = "depression" if mean_deviation[peak_idx] < 0 else "bulge"
            print(f"  ✓ LAKE CANDIDATE {len(lake_candidates)}:")
            print(f"    Peak at {distances[peak_idx]:.0f}m, {direction}")
            print(f"    Boundaries: {distances[left_boundary]:.0f} – {distances[right_boundary]:.0f}m "
                  f"(span: {refined_span:.0f}m)")
            print(f"    Confidence: {confidence:.2f} | Profiles affected: {n_affected}")
    
    # Sort by distance
    lake_candidates.sort(key=lambda x: x['start_distance'])
    
    # Remove overlaps (keep highest confidence)
    if len(lake_candidates) > 1:
        non_overlapping = []
        for lake in lake_candidates:
            overlap = False
            for selected in non_overlapping:
                if (lake['start_distance'] < selected['end_distance'] and 
                    lake['end_distance'] > selected['start_distance']):
                    overlap = True
                    if lake['confidence'] > selected['confidence']:
                        non_overlapping.remove(selected)
                        non_overlapping.append(lake)
                    break
            if not overlap:
                non_overlapping.append(lake)
        lake_candidates = sorted(non_overlapping, key=lambda x: x['start_distance'])
    
    if verbose:
        if len(lake_candidates) == 0:
            print(f"\n  No lake candidates detected.")
        else:
            print(f"\n  Final: {len(lake_candidates)} non-overlapping lake candidate(s).")
    
    # ──────────────────────────────────────────────────────────────────────
    # STEP 7: Get coordinates
    # ──────────────────────────────────────────────────────────────────────
    
    first_profile = profiles_sorted[0]
    x0, y0, x1, y1 = first_profile['coords']
    
    for lake in lake_candidates:
        frac_start = lake['start_distance'] / first_profile['distance']
        frac_end = lake['end_distance'] / first_profile['distance']
        
        lake['start_coords_3413'] = (x0 + frac_start * (x1 - x0), y0 + frac_start * (y1 - y0))
        lake['end_coords_3413'] = (x0 + frac_end * (x1 - x0), y0 + frac_end * (y1 - y0))
        
        transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
        start_lon, start_lat = transformer.transform(*lake['start_coords_3413'])
        end_lon, end_lat = transformer.transform(*lake['end_coords_3413'])
        
        lake['start_coords_4326'] = (start_lon, start_lat)
        lake['end_coords_4326'] = (end_lon, end_lat)
    
    return {
        'lake_boundaries': lake_candidates,
        'anomaly_map': anomaly_scores,
        'n_deviating': n_deviating,
        'mean_deviation': mean_deviation,
        'median_profile': median_profile,
        'deviation_matrix': deviation_matrix,
        'elevation_matrix': elevation_matrix,
        'distances': distances,
        'threshold': threshold,
    }


def plot_lake_detection_results(all_profiles, lake_results, coreg_mode,
                               cmap='terrain', margin_km=2, lake_name=None):
    """Plot transect profiles with detected lake boundaries overlaid.
    
    Modifies the combined profile plot to show lake boundaries as vertical
    bands and markers on the map panels.
    
    Parameters
    ----------
    all_profiles : dict
        Profile data
    lake_results : dict
        Output from detect_lake_boundaries()
    coreg_mode : str
        Coregistration mode
    cmap : str
        Colormap for DEM
    margin_km : float
        Margin around transect for maps
    lake_name : str, optional
        Lake name for title
        
    Returns
    -------
    str
        Path to saved plot
    """
    if not all_profiles["profiles"]:
        raise ValueError("No profiles to plot")
    
    lake_boundaries = lake_results.get('lake_boundaries', [])
    has_lakes = len(lake_boundaries) > 0
    
    # Use first DEM for maps
    selected_profile = all_profiles["profiles"][0]
    raster_metadata = selected_profile["metadata"]
    x0, y0, x1, y1 = selected_profile["coords"]
    
    # Create figure with 4 subplots if lakes found (adds anomaly panel)
    if has_lakes:
        fig = plt.figure(figsize=(18, 12))
        # 3 rows: [maps row], [profiles row], [anomaly row]
        gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], 
                             height_ratios=[2, 2, 1],
                             wspace=0.35, hspace=0.30)
        
        # Maps occupy left column, rows 0-1
        gs_map_top = gs[0, 0]
        gs_map_bottom = gs[1, 0]
        
        # Profiles occupy right column, rows 0-1
        gs_profiles = gs[0:2, 1]
        
        # Anomaly spans full width, row 2
        gs_anomaly = gs[2, :]
    else:
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1],
                             wspace=0.32, hspace=0.22)
        gs_map_top = gs[0, 0]
        gs_map_bottom = gs[1, 0]
        gs_profiles = gs[:, 1]

    # Get DEM file
    demfile = raster_metadata["path"]
    if not os.path.exists(demfile):
        demfile = find_and_unzip(demfile)
    
    with rio.open(demfile) as src:
        # Calculate window with margin
        margin = margin_km * 1000
        min_x, max_x = min(x0, x1) - margin, max(x0, x1) + margin
        min_y, max_y = min(y0, y1) - margin, max(y0, y1) + margin
        
        window = src.window(min_x, min_y, max_x, max_y)
        window_bounds = rio.windows.bounds(window, src.transform)
        
        raster_data = src.read(1, window=window)
        
        if src.nodata is not None:
            raster_data_masked = np.ma.masked_equal(raster_data, src.nodata)
        else:
            raster_data_masked = np.ma.masked_invalid(raster_data)
        
        # Hillshade
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(raster_data_masked, vert_exag=2, 
                                dx=src.res[0], dy=src.res[1])
        
        # TOP LEFT: Elevation map
        ax1 = fig.add_subplot(gs[0, 0])
        img1 = ax1.imshow(raster_data_masked, cmap=cmap,
                         extent=(window_bounds[0], window_bounds[2],
                                window_bounds[1], window_bounds[3]),
                         origin="upper", aspect=1.2, interpolation="none", rasterized=True)
        
        # Transect line
        ax1.plot([x0, x1], [y0, y1], "k--", linewidth=1.5)
        ax1.plot(x0, y0, "ro", markersize=8, markeredgewidth=1.5,
                markeredgecolor="white", label="Start (A)")
        ax1.plot(x1, y1, "bo", markersize=8, markeredgewidth=1.5,
                markeredgecolor="white", label="End (B)")
        
        # Plot lake boundaries as crosses on the map
        if has_lakes:
            lake_colors = plt.cm.Set1(np.linspace(0, 1, len(lake_boundaries)))
            for i, lake in enumerate(lake_boundaries):
                sx, sy = lake['start_coords_3413']
                ex, ey = lake['end_coords_3413']
                color = lake_colors[i]
                # Start and end boundary markers
                ax1.plot(sx, sy, '+', color=color, markersize=12, markeredgewidth=2.5,
                        zorder=10, label=f'Lake {i+1} start' if i == 0 else '')
                ax1.plot(ex, ey, 'x', color=color, markersize=12, markeredgewidth=2.5,
                        zorder=10, label=f'Lake {i+1} end' if i == 0 else '')
                # Connect with a colored line segment
                ax1.plot([sx, ex], [sy, ey], '-', color=color, linewidth=3, alpha=0.6,
                        zorder=9, label=f'Lake {i+1}' if i == 0 else '')

        # Scale bar
        scale_length = 3000
        x_pos = 0.1 * (window_bounds[2] - window_bounds[0]) + window_bounds[0]
        y_pos = 0.05 * (window_bounds[3] - window_bounds[1]) + window_bounds[1]
        ax1.plot([x_pos, x_pos + scale_length], [y_pos, y_pos], color="black",
                linewidth=3, solid_capstyle="butt")
        ax1.text(x_pos + scale_length / 2, y_pos + 0.016 * (window_bounds[3] - window_bounds[1]),
                "3 km", ha="center", va="bottom", fontsize=10, color="black")
        
        refdemname = raster_metadata["dem_name"]
        date_label = extract_date_label(refdemname)
        ax1.set_title(f"Reference DEM - {date_label}", pad=12, fontsize=11)
        ax1.legend(fontsize=7, loc="upper right")
        
        cbar1 = fig.colorbar(img1, ax=ax1, orientation="vertical", pad=0.2,
                            fraction=0.033, aspect=25)
        cbar1.set_label("Elevation (m)", rotation=270, labelpad=13, fontsize=11)
        
        # Secondary axes
        transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
        ax1.secondary_xaxis("top", functions=(
            lambda x: transformer.transform(x, np.full_like(x, window_bounds[1]))[0],
            lambda x: x))
        ax1.secondary_yaxis("right", functions=(
            lambda y: transformer.transform(np.full_like(y, window_bounds[0]), y)[1],
            lambda y: y))
        
        # BOTTOM LEFT: Hillshade
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(hillshade, cmap="gray",
                  extent=(window_bounds[0], window_bounds[2],
                         window_bounds[1], window_bounds[3]),
                  origin="upper", aspect=1.2, interpolation="none", rasterized=True)
        
        ax2.plot([x0, x1], [y0, y1], "k--", linewidth=1.5)
        ax2.plot(x0, y0, "ro", markersize=8, markeredgewidth=1.5,
                markeredgecolor="white")
        ax2.plot(x1, y1, "bo", markersize=8, markeredgewidth=1.5,
                markeredgecolor="white")
        
        # Lake boundaries on hillshade
        if has_lakes:
            lake_colors = plt.cm.Set1(np.linspace(0, 1, len(lake_boundaries)))
            for i, lake in enumerate(lake_boundaries):
                sx, sy = lake['start_coords_3413']
                ex, ey = lake['end_coords_3413']
                color = lake_colors[i]
                ax2.plot(sx, sy, '+', color=color, markersize=12, markeredgewidth=2.5, zorder=10)
                ax2.plot(ex, ey, 'x', color=color, markersize=12, markeredgewidth=2.5, zorder=10)
                ax2.plot([sx, ex], [sy, ey], '-', color=color, linewidth=3, alpha=0.6, zorder=9)
        
        ax2.plot([x_pos, x_pos + scale_length], [y_pos, y_pos], color="black",
                linewidth=3, solid_capstyle="butt")
        ax2.text(x_pos + scale_length / 2, y_pos + 0.016 * (window_bounds[3] - window_bounds[1]),
                "3 km", ha="center", va="bottom", fontsize=10, color="black")
        
        ax2.set_title(f"Hillshade - {date_label}", pad=10, fontsize=11)
    
    # RIGHT: Elevation profiles
    ax3 = fig.add_subplot(gs_profiles)
    
    # Sort profiles
    profiles_sorted = sorted(all_profiles["profiles"],
                            key=lambda x: extract_date_obj(x["metadata"]["dem_name"]))
    
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(profiles_sorted)))
    
    # Plot profiles
    for i, profile in enumerate(profiles_sorted):
        label = extract_date_label(profile["metadata"]["dem_name"])
        profvalues = np.array(profile["profile_values"])
        elprofile = np.ma.masked_where(profvalues < -500, profvalues)
        
        ax3.plot(profile["transect"], elprofile, color=colors[i],
                label=label, linewidth=1.5)
    
    # Add lake boundary vertical lines
    if has_lakes:
        lake_colors = plt.cm.Set1(np.linspace(0, 1, len(lake_boundaries)))
        for i, lake in enumerate(lake_boundaries):
            color = lake_colors[i]
            # Shaded region for lake extent
            ax3.axvspan(lake['start_distance'], lake['end_distance'],
                       alpha=0.15, color=color, zorder=0)
            
            # Boundary lines
            ax3.axvline(x=lake['start_distance'], color=color,
                       linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Lake {i+1} boundary' if i == 0 else '')
            ax3.axvline(x=lake['end_distance'], color=color,
                       linestyle='--', linewidth=2.5, alpha=0.8)
            
            # Annotation
            # y_limits = ax3.get_ylim()
            # ax3.annotate(f'L{i+1}\n{lake["span_meters"]:.0f}m',
            #             xy=(lake['start_distance'] + lake['span_meters']/2, y_limits[1]),
            #             ha='center', va='top', fontsize=8,
            #             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax3.set_xlim(lake_results['distances'][0], lake_results['distances'][-1])
    ax3.set_xlabel("Distance along transect (m)", fontsize=11)
    ax3.set_ylabel("Elevation (m)", fontsize=11)
    ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    
    # Legend
    if len(profiles_sorted) > 10:
        handles, labels = ax3.get_legend_handles_labels()
        step = max(1, len(handles) // 10)
        ax3.legend(handles[::step], labels[::step], fontsize=8, loc="best")
    else:
        ax3.legend(fontsize=8, loc="best")
    
    # ANOMALY PANEL (if lakes detected and space available)
    if has_lakes:
        ax4 = fig.add_subplot(gs_anomaly, sharex=ax3)
        
        anomaly_map = lake_results['anomaly_map']
        distances = lake_results['distances']
        threshold = lake_results.get('threshold', 0)
        
        # Plot individual components
        if 'local_std' in lake_results:
            local_std_norm = lake_results['local_std'] / max(np.nanmax(lake_results['local_std']), 0.001)
            ax4.plot(distances, local_std_norm, 'b-', linewidth=0.8, alpha=0.4, label='Local std')
        
        if 'trend_residual_std' in lake_results:
            trend_norm = lake_results['trend_residual_std'] / max(np.nanmax(lake_results['trend_residual_std']), 0.001)
            ax4.plot(distances, trend_norm, 'g-', linewidth=0.8, alpha=0.4, label='Trend residuals')
        
        if 'curvature_abs' in lake_results:
            curv_norm = lake_results['curvature_abs'] / max(np.nanmax(lake_results['curvature_abs']), 0.001)
            ax4.plot(distances, curv_norm, 'purple', linewidth=0.8, alpha=0.4, label='Curvature')
        
        # ax4.fill_between(distances, 0, anomaly_map, alpha=0.3, color='red')
        ax4.plot(distances, anomaly_map, 'r-', linewidth=1.5, label='Anomaly score')
        ax4.axhline(y=threshold, color='brown', linestyle='--', linewidth=1.5,
                   label=f'Threshold ({threshold:.2f})')
        
        # Highlight detected lake regions
        lake_colors = plt.cm.Set1(np.linspace(0, 1, len(lake_boundaries)))
        for i, lake in enumerate(lake_boundaries):
            color = lake_colors[i]
            ax4.axvspan(lake['start_distance'], lake['end_distance'],
                       alpha=0.15, color=color)
            ax4.axvline(x=lake['start_distance'], color=color,
                       linestyle=':', linewidth=1.5, alpha=0.7)
            ax4.axvline(x=lake['end_distance'], color=color,
                       linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax4.set_ylim(bottom=0)
        ax4.set_xlim(distances[0], distances[-1])
        ax4.set_xlabel("Distance along transect (m)", fontsize=10)
        ax4.set_ylabel("Anomaly Score", fontsize=10)
        ax4.set_title("Lake Detection: Anomaly Components & Combined Score", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.5)
    
    # Title
    if len(profiles_sorted) >= 2:
        date_start = extract_date_label(profiles_sorted[-1]["metadata"]["dem_name"])
        date_end = extract_date_label(profiles_sorted[0]["metadata"]["dem_name"])
    else:
        date_start = date_end = extract_date_label(profiles_sorted[0]["metadata"]["dem_name"])
    
    if has_lakes:
        title = f"Elevation Profiles with Lake Detection: {date_start} – {date_end}"
    else:
        title = f"Elevation Profiles: {date_start} – {date_end}"
    
    if lake_name:
        title = f"{lake_name} — {title}"
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        
    # Save
    coords = all_profiles["transect_coords"]
    xs, ys = coords[0]
    xe, ye = coords[1]
    
    yearstart = extract_year(all_profiles["profiles"][-1]["metadata"]["dem_name"])
    yearend = extract_year(all_profiles["profiles"][0]["metadata"]["dem_name"])
    
    if lake_name:
        output_path = get_output_path('transects_combined',
            f"profile_lakes_{lake_name}_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}-{yearstart}-{yearend}_{coreg_mode}.png")
    else:
        output_path = get_output_path('transects_combined',
            f"profile_lakes_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}-{yearstart}-{yearend}_{coreg_mode}.png")
    
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Lake detection plot saved to: {output_path}")
    
    return output_path
