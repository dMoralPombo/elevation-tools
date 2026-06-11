"""
ArcticDEM Analysis Functions
=============================
Profile processing, elevation history, and visualization.
"""

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

# Import utilities
from elevation_utils import (
    wgs84_to_3413, find_and_unzip, get_elevation_window,
    read_elevation_from_compressed, extract_elevation_profile, 
    extract_elevation_profile_compressed, transect_from_mosaic
)
from config import (
    OUTPUT_DIR, DEFAULT_NUM_SAMPLES, DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_TYPE, COREG_PARAMS, get_output_path
)


# ============================================================================
# DATE HELPERS
# ============================================================================

def extract_date_label(dem_name):
    """Extract readable date from DEM filename."""
    parts = dem_name.split("_")
    if len(parts) > 1 and len(parts[1]) >= 6:
        date_str = parts[1]
        year = date_str[:4]
        month = date_str[4:6]
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        try:
            return f"{month_names[int(month)]} {year}"
        except (ValueError, IndexError):
            return f"{year}-{month}"
    return dem_name[:8]


def extract_year(dem_name):
    """Extract year from DEM filename."""
    parts = dem_name.split("_")
    if len(parts) > 1 and len(parts[1]) >= 4:
        return int(parts[1][:4])
    return 0


def extract_date_obj(dem_name):
    """Extract datetime object from DEM filename."""
    parts = dem_name.split("_")
    if len(parts) > 1 and len(parts[1]) >= 8:
        try:
            return datetime.strptime(parts[1][:8], "%Y%m%d")
        except:
            return datetime(int(parts[1][:4]), 1, 1)
    return datetime(2000, 1, 1)


# ============================================================================
# ELEVATION HISTORY PROCESSING
# ============================================================================

def process_elevation_history(
    geocells, pairnames, dates, archdir, coords,
    window_size=DEFAULT_WINDOW_SIZE,
    window_type=DEFAULT_WINDOW_TYPE,
    coreg_mode="none"
):
    """Track elevation values at a point across multiple DEMs.
    
    Parameters
    ----------
    geocells, pairnames, dates : lists
        DEM identifiers from STAC search
    archdir : str
        Archive directory path
    coords : tuple
        (lon, lat) in WGS84
    window_size, window_type : as in get_elevation_window
    coreg_mode : str
        'none', 'altim', or 'mosaic'
        
    Returns
    -------
    dict
        Elevation history data and metadata
    """
    print(f"\n=== Processing elevation history for {len(pairnames)} DEMs ===")
    print(f"Window: {window_size}x{window_size} {window_type}")
    
    # Transform coordinates
    coords_3413 = wgs84_to_3413(*coords)
    x, y = coords_3413
    print(f"Coordinates (EPSG:3413): {x:.1f}, {y:.1f}")
    
    # Parse dates
    date_objs = [datetime.strptime(d, "%Y-%m-%dT%H:%M:%SZ") for d in dates]
    
    # Initialize storage
    history = {
        'elevations': [], 'elevations_std': [], 'valid_pixels': [],
        'dates': [], 'pairnames': [], 'metadata': [],
        'coords_4326': coords, 'coords_3413': coords_3413,
        'window_size': window_size, 'window_type': window_type,
    }
    
    # Build coregistration suffix
    if coreg_mode != 'none':
        params = COREG_PARAMS.get(coreg_mode, {})
        ref = params.get('reference_data', 'unknown')
        vel = params.get('filter_vel', '0')
        dhdt = params.get('filter_dhdt', '0')
        suffix = f"_{ref}_v_{vel}-0_dh_{dhdt}-0"
    else:
        suffix = ""
    
    # Process each DEM
    for i, (geocell, pairname) in enumerate(zip(geocells, pairnames)):
        print(f"\nProcessing {i+1}/{len(pairnames)}: {pairname}")
        
        if coreg_mode == 'none':
            # Use compressed files
            raster_path = os.path.join(
                archdir,
                f"{geocell}/SETSM_s2s041_{pairname}_2m_lsf_seg1_dem.tif"
            )
            mean_elev, std_elev, valid_count = read_elevation_from_compressed(
                raster_path, x, y, window_size, window_type
            )
        else:
            # Use coregistered files
            base_dir = os.path.join(archdir, geocell)
            pattern = f"SETSM_s2s041_{pairname}_2m_lsf_seg1_dem*{suffix}*coregistered.tif"
            matches = glob.glob(os.path.join(base_dir, pattern))
            
            if matches:
                with rio.open(matches[0]) as src:
                    mean_elev, std_elev, valid_count = get_elevation_window(
                        src, x, y, window_size, window_type
                    )
            else:
                print(f"  No coregistered file found - skipping")
                mean_elev, std_elev, valid_count = np.nan, np.nan, 0
        
        # Store results
        history['elevations'].append(mean_elev)
        history['elevations_std'].append(std_elev)
        history['valid_pixels'].append(valid_count)
        history['dates'].append(date_objs[i])
        history['pairnames'].append(pairname)
        history['metadata'].append({
            'geocell': geocell,
            'valid': not np.isnan(mean_elev),
            'valid_pixels': valid_count,
            'std': std_elev,
        })
        
        if not np.isnan(mean_elev):
            print(f"  Elevation: {mean_elev:.1f} ± {std_elev:.1f} m (n={valid_count})")
    
    # Print summary
    valid_count = sum(1 for m in history['metadata'] if m.get('valid', False))
    print(f"\n=== Complete: {valid_count}/{len(pairnames)} valid points ===")
    
    return history


# ============================================================================
# ELEVATION PROFILE PROCESSING
# ============================================================================

def process_elevation_profiles(
    transect_coords, pairnames, geocells, archdir,
    num_samples=DEFAULT_NUM_SAMPLES, coreg_mode='none'
):
    """Extract elevation profiles from multiple DEMs along a transect.
    
    Parameters
    ----------
    transect_coords : tuple
        ((start_lon, start_lat), (end_lon, end_lat)) in WGS84
    pairnames, geocells : lists
        DEM identifiers
    archdir : str
        Archive directory
    num_samples : int
        Points along transect
    coreg_mode : str
        'none', 'altim', or 'mosaic'
        
    Returns
    -------
    dict
        All profile data
    """
    all_profiles = {
        'transect_coords': transect_coords,
        'profiles': [],
        'mosaic': None,
    }

    # Add the mosaic transect first, if available
    transect_mosaic, profile_mosaic, transect_distance, x0, y0, x1, y1 = (transect_from_mosaic(transect_coords, num_samples=100))

    # Store mosaic profile
    all_profiles['mosaic'] = {
        'transect': transect_mosaic,
        'profile_values': profile_mosaic,
        'distance': transect_distance,
        'coords': (x0, y0, x1, y1),
    }
    if profile_mosaic is not None:
        print("Mosaic profile extracted successfully.")

    # Build suffix for coregistered files
    if coreg_mode != 'none':
        params = COREG_PARAMS.get(coreg_mode, {})
        ref = params.get('reference_data', 'unknown')
        vel = params.get('filter_vel', '0')
        dhdt = params.get('filter_dhdt', '0')
        suffix = f"_{ref}_v_{vel}-0_dh_{dhdt}-0"
    else:
        suffix = ""
    
    print(f"\nProcessing {len(pairnames)} DEMs for elevation profiles...")
    
    for i, (pairname, geocell) in enumerate(tqdm(zip(pairnames, geocells), 
                                                   total=len(pairnames))):
        try:
            if coreg_mode == 'none':
                raster_path = os.path.join(
                    archdir,
                    f"{geocell}/SETSM_s2s041_{pairname}_2m_lsf_seg1_dem.tif"
                )
                dem_name = f"SETSM_{pairname}"
                transect, elevations, distance, x0, y0, x1, y1 = \
                    extract_elevation_profile_compressed(transect_coords, raster_path, num_samples)
            else:
                raster_path = os.path.join(
                    archdir,
                    f"{geocell}/SETSM_s2s041_{pairname}_2m_lsf_seg1_dem{suffix}_coregistered.tif"
                )
                dem_name = f"SETSM_{pairname}"
                transect, elevations, distance, x0, y0, x1, y1 = \
                    extract_elevation_profile(transect_coords, raster_path, num_samples)
            
            all_profiles['profiles'].append({
                'transect': transect,
                'profile_values': elevations,
                'distance': distance,
                'coords': (x0, y0, x1, y1),
                'metadata': {
                    'dem_name': dem_name,
                    'pairname': pairname,
                    'geocell': geocell,
                    'path': raster_path,
                }
            })
        except Exception as e:
            print(f"Error processing {pairname}: {e}")
            continue
    
    print(f"Successfully processed {len(all_profiles['profiles'])} profiles")
    return all_profiles


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_elevation_history(history, output_path=None, coreg_mode='none'):
    """Plot elevation time series with error bars.
    
    Parameters
    ----------
    history : dict
        Output from process_elevation_history
    output_path : str, optional
        Path to save plot
    coreg_mode : str
        Coregistration mode for filename
        
    Returns
    -------
    str
        Path to saved plot
    """
    # Create DataFrame
    df = pl.DataFrame({
        'date': history['dates'],
        'elevation': history['elevations'],
        'elevation_std': history['elevations_std'],
        'valid_pixels': history['valid_pixels'],
        'pairname': history['pairnames'],
    }).sort('date')
    
    # Filter valid values
    valid_df = df.filter(~pl.col('elevation').is_nan())
    if len(valid_df) == 0:
        print("No valid data to plot")
        return None
    
    # Remove outliers
    median = valid_df['elevation'].median()
    std = valid_df['elevation'].std()
    valid_df = valid_df.filter(
        (pl.col('elevation') <= median + 2 * std) &
        (pl.col('elevation') >= median - 2 * std)
    )
    
    valid_pd = valid_df.to_pandas()
    window_size = history.get('window_size', 3)
    window_type = history.get('window_type', 'square')
    window_desc = f"{window_size}x{window_size} {window_type}"
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot with error bars
    if window_size == 1:
        ax.plot(valid_pd['date'], valid_pd['elevation'], 'o', 
                markersize=6, color='royalblue', alpha=0.9)
    else:
        ax.errorbar(valid_pd['date'], valid_pd['elevation'],
                   yerr=valid_pd['elevation_std'],
                   fmt='o', markersize=6, color='royalblue',
                   ecolor='gray', elinewidth=1.5, capsize=3,
                   alpha=1.0, markeredgecolor='white',
                   label=f'Mean ± SD ({window_desc})')
    
    # Trend line
    if len(valid_pd) > 3:
        dates_num = mdates.date2num(valid_pd['date'])
        z = np.polyfit(dates_num, valid_pd['elevation'], 1)
        p = np.poly1d(z)
        slope = z[0] * 365.25
        
        trend_color = 'red' if slope < -0.5 else 'green' if slope > 0.5 else 'orange'
        ax.plot(valid_pd['date'], p(dates_num), '--', color=trend_color,
               linewidth=1.5, alpha=0.6, label=f'Trend: {slope:.2f} m/yr')
    
    # Mean line
    mean_elev = valid_pd['elevation'].mean()
    ax.axhline(y=mean_elev, color='gray', linestyle=':', linewidth=1, alpha=0.5,
              label=f'Mean: {mean_elev:.1f}m')
    
    # Format axes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_ylabel('Elevation (m)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    
    coords = history['coords_4326']
    ax.set_title(f"Elevation History at ({coords[0]:.3f}°E, {coords[1]:.3f}°N)\n"
                f"{window_desc} window", fontsize=14, fontweight='bold')
    
    ax.legend(loc='best', fontsize=9, frameon=True, fancybox=True)
    plt.tight_layout()
    
    # Save
    years = f"{valid_pd['date'].iloc[0].year}-{valid_pd['date'].iloc[-1].year}"
    # output_path = get_output_path(
    #     'elevation_histories',
    #     f"elevation_history_{coords[0]:.3f}_{coords[1]:.3f}_{coreg_mode}_{years}.png"
    # )
    output_path = os.path.join(
        output_path if output_path else OUTPUT_DIR,
        f"elevation_history_{coords[0]:.3f}_{coords[1]:.3f}_{coreg_mode}_{years}.png"
    )
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


def plot_combined_profiles(all_profiles, coreg_mode='none', 
                          cmap='terrain', lake_name=None, margin_km=2):
    """Plot elevation profiles from multiple DEMs with reference map.
    
    Parameters
    ----------
    all_profiles : dict
        Output from process_elevation_profiles
    coreg_mode : str
        For filename suffix
    cmap : str
        Colormap for DEM
    lake_name : str, optional
        Name for title
    margin_km : float, optional
        Margin around transect for map plot (in km)
        
    Returns
    -------
    str
        Path to saved plot
    """
    if not all_profiles["profiles"]:
        raise ValueError("No profile data available for plotting")

    # Use first DEM for the map plot
    selected_profile = all_profiles["profiles"][0]
    raster_metadata = selected_profile["metadata"]
    x0, y0, x1, y1 = selected_profile["coords"]

    # Create figure with 3 subplots: 2 left (stacked), 1 right
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1], 
                         wspace=0.32, hspace=0.22)

    # Get DEM file - use the path directly since it's already a file path for coregistered DEMs
    demfile = raster_metadata["path"]
    
    # For compressed DEMs, we might need to handle them differently
    if not os.path.exists(demfile): # and '.gz' in demfile:
        # Try to find the uncompressed version or handle compressed
        demfile = find_and_unzip(demfile)

    with rio.open(demfile) as src:
        # Calculate the bounding box with margin
        margin = margin_km * 1000  # meters
        min_x, max_x = min(x0, x1) - margin, max(x0, x1) + margin
        min_y, max_y = min(y0, y1) - margin, max(y0, y1) + margin

        # Calculate window to read
        window = src.window(min_x, min_y, max_x, max_y)
        window_bounds = rio.windows.bounds(window, src.transform)

        # Read raster data
        raster_data = src.read(1, window=window)
        # Handle nodata values properly
        if src.nodata is not None:
            raster_data_masked = np.ma.masked_equal(raster_data, src.nodata)
        else:
            # Try common nodata values
            for nodata_candidate in [-9999, -32768, 0]:
                if np.any(raster_data == nodata_candidate):
                    raster_data_masked = np.ma.masked_equal(raster_data, nodata_candidate)
                    break
            else:
                raster_data_masked = np.ma.masked_invalid(raster_data)

        # Create hillshade
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(raster_data_masked, vert_exag=2, dx=src.res[0], dy=src.res[1])

        # TOP LEFT - DEM with elevation colormap
        ax1 = fig.add_subplot(gs[0, 0])
        img1 = ax1.imshow(raster_data_masked, cmap=cmap, 
                         extent=(window_bounds[0], window_bounds[2], 
                                window_bounds[1], window_bounds[3]),
                         origin="upper", aspect=1.2, interpolation="none", rasterized=True)

        # Plot transect line and markers
        ax1.plot([x0, x1], [y0, y1], "k--", linewidth=1.5)
        ax1.plot(x0, y0, "ro", markersize=8, markeredgewidth=1.5, 
                markeredgecolor="white", label="Start (A)")
        ax1.plot(x1, y1, "bo", markersize=8, markeredgewidth=1.5, 
                markeredgecolor="white", label="End (B)")

        # Add scale bar
        scale_length = 3000  # 3 km in meters
        x_pos = 0.1 * (window_bounds[2] - window_bounds[0]) + window_bounds[0]
        y_pos = 0.05 * (window_bounds[3] - window_bounds[1]) + window_bounds[1]
        ax1.plot([x_pos, x_pos + scale_length], [y_pos, y_pos], color="black", 
                linewidth=3, solid_capstyle="butt")
        ax1.text(x_pos + scale_length / 2, y_pos + 0.016 * (window_bounds[3] - window_bounds[1]), 
                "3 km", ha="center", va="bottom", fontsize=10, color="black")

        # Format top left plot
        refdemname = raster_metadata["dem_name"]
        dem_parts = refdemname.split("_")
        if len(dem_parts) > 1 and len(dem_parts[1]) >= 6:
            date_str = dem_parts[1]
            year, month = date_str[:4], date_str[4:6]
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            try:
                title_date = f"{month_names[int(month)]} {year}"
            except (ValueError, IndexError):
                title_date = refdemname
        else:
            title_date = refdemname

        ax1.set_title(f"Reference DEM Elevation - {title_date}", pad=12, fontsize=11)
        ax1.legend(fontsize=8, loc="upper right")
        cbar1 = fig.colorbar(img1, ax=ax1, orientation="vertical", pad=0.2, 
                            fraction=0.033, aspect=25)
        cbar1.set_label("Elevation (m)", rotation=270, labelpad=13, fontsize=11)

        # Add secondary axes for EPSG:4326 coordinates
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
        ax1.secondary_xaxis("top", functions=(
            lambda x: transformer.transform(x, np.full_like(x, window_bounds[1]))[0], 
            lambda x: x))
        ax1.secondary_yaxis("right", functions=(
            lambda y: transformer.transform(np.full_like(y, window_bounds[0]), y)[1], 
            lambda y: y))

        # BOTTOM LEFT - Hillshade
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(hillshade, cmap="gray", 
                  extent=(window_bounds[0], window_bounds[2], 
                         window_bounds[1], window_bounds[3]),
                  origin="upper", aspect=1.2, interpolation="none", rasterized=True)

        # Plot transect line and markers
        ax2.plot([x0, x1], [y0, y1], "k--", linewidth=1.5)
        ax2.plot(x0, y0, "ro", markersize=8, markeredgewidth=1.5, 
                markeredgecolor="white", label="Start (A)")
        ax2.plot(x1, y1, "bo", markersize=8, markeredgewidth=1.5, 
                markeredgecolor="white", label="End (B)")

        # Add scale bar
        ax2.plot([x_pos, x_pos + scale_length], [y_pos, y_pos], color="black", 
                linewidth=3, solid_capstyle="butt")
        ax2.text(x_pos + scale_length / 2, y_pos + 0.016 * (window_bounds[3] - window_bounds[1]), 
                "3 km", ha="center", va="bottom", fontsize=10, color="black")

        ax2.set_title(f"Hillshade - {title_date}", pad=10, fontsize=11)
        ax2.legend(fontsize=8, loc="upper right")
        ax2.secondary_xaxis("top", functions=(
            lambda x: transformer.transform(x, np.full_like(x, window_bounds[1]))[0], 
            lambda x: x))
        ax2.secondary_yaxis("right", functions=(
            lambda y: transformer.transform(np.full_like(y, window_bounds[0]), y)[1], 
            lambda y: y))

    # RIGHT PLOT - All elevation profiles (spans both rows)
    ax3 = fig.add_subplot(gs[:, 1])

    # Define colors for profiles
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(all_profiles["profiles"])))

    # Set consistent limits
    all_values = np.concatenate([p["profile_values"] for p in all_profiles["profiles"]])
    valid_values = all_values[~np.isnan(all_values)]
    median, std = np.nanmedian(valid_values), np.nanstd(valid_values)
    print(f"Median elevation: {median:.1f} ± {std:.1f} m")

    # Add the transect of the mosaic, if available
    if "mosaic" in all_profiles and isinstance(all_profiles["mosaic"].get("profile_values"), np.ndarray):
        mosaic_profile = np.array(all_profiles["mosaic"]["profile_values"])
        elprofile_mosaic = np.ma.masked_where(mosaic_profile < -5000, mosaic_profile)
        combined_mask = ((elprofile_mosaic > median + 3 * std) | 
                        (elprofile_mosaic < median - 3 * std) | 
                        elprofile_mosaic.mask)
        filtered_profile_m = np.ma.masked_where(combined_mask, elprofile_mosaic)
        ax3.plot(all_profiles["mosaic"]["transect"], filtered_profile_m, color="black", 
                label="Mosaic", linewidth=3, linestyle="--", alpha=1.0)

    # Plot each profile
    for i, profile in enumerate(all_profiles["profiles"]):
        # Extract year and month for label
        dem_name = profile["metadata"]["dem_name"]
        dem_parts = dem_name.split("_")
        if len(dem_parts) > 1 and len(dem_parts[1]) >= 6:
            date_str = dem_parts[1]
            year, month = date_str[:4], date_str[4:6]
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            try:
                label = f"{month_names[int(month)]} {year}"
            except (ValueError, IndexError):
                label = dem_parts[1][:6] if len(dem_parts[1]) >= 6 else dem_parts[1]
        else:
            label = dem_parts[1] if len(dem_parts) > 1 else dem_name

        profvalues = np.array(profile["profile_values"])
        elprofile = np.ma.masked_where(profvalues < -5000, profvalues)
        combined_mask = ((elprofile > median + 2 * std) | 
                        (elprofile < median - 2 * std) | 
                        elprofile.mask)
        filtered_profile = np.ma.masked_where(combined_mask, elprofile)
        ax3.plot(profile["transect"], filtered_profile, color=colors[i], 
                label=label, linewidth=1.5)

    # Format right plot
    ax3.set_xlabel("Distance along transect (m)", fontsize=11)
    ax3.set_ylabel("Elevation (m)", fontsize=11)
    ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # If there are many profiles, make a subsample for the legend of 10 elements:
    if len(all_profiles["profiles"]) > 10:
        handles, labels = ax3.get_legend_handles_labels()
        step = max(1, len(handles) // 10)
        ax3.legend(handles[::step], labels[::step], fontsize=8, loc="best")
    else:
        ax3.legend(fontsize=8, loc="best")

    # Set y-limits with minimal padding
    if len(valid_values) > 0:
        if "mosaic" in all_profiles and isinstance(all_profiles["mosaic"].get("profile_values"), np.ndarray):
            y_min, y_max = np.nanmin(filtered_profile_m), np.nanmax(filtered_profile_m)
        else:
            y_min, y_max = np.nanmin(filtered_profile), np.nanmax(filtered_profile)
        y_range, y_padding = y_max - y_min, 0.3 * (y_max - y_min)
        print(f"Y-axis limits: {y_min - y_padding:.2f} to {y_max + y_padding:.2f}")
        print(f"Range: {y_range:.2f}, Padding: {y_padding:.2f}")
        ax3.set_ylim(y_min - y_padding, y_max + y_padding)

    ax3.set_xlim(0, all_profiles["profiles"][0]["distance"])

    # Add A/B markers
    first_profile_values = all_profiles["profiles"][0]["profile_values"]
    ax3.annotate("A", xy=(0, first_profile_values[0]), xytext=(0.0, 1.01), 
                textcoords="axes fraction", color="red", fontsize=13, fontweight="bold")
    ax3.annotate("B", xy=(all_profiles["profiles"][0]["distance"], first_profile_values[-1]), 
                xytext=(1.0, 1.01), textcoords="axes fraction", ha="right", 
                color="blue", fontsize=13, fontweight="bold")

    # Main title
    def extract_year_month(dem_name):
        parts = dem_name.split("_")
        if len(parts) > 1 and len(parts[1]) >= 6:
            date_str = parts[1]
            year, month = date_str[:4], date_str[4:6]
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            try:
                return f"{month_names[int(month)]} {year}"
            except (ValueError, IndexError):
                return year
        return parts[1][:4] if len(parts) > 1 else ""

    date_start = extract_year_month(all_profiles["profiles"][-1]["metadata"]["dem_name"])
    date_end = extract_year_month(all_profiles["profiles"][0]["metadata"]["dem_name"])
    fig.suptitle(f"Elevation Profile Comparison: {date_start} - {date_end}", 
                fontsize=15, fontweight="bold", y=0.98)

    # Adjust layout
    plt.subplots_adjust(left=0.06, right=0.96, bottom=0.08, top=0.94)

    # Extract coordinates and years for filename
    coords_4326 = all_profiles["transect_coords"]
    xs, ys = coords_4326[0]
    xe, ye = coords_4326[1]
    yearstart = all_profiles["profiles"][-1]["metadata"]["dem_name"].split("_")[1][:4]
    yearend = all_profiles["profiles"][0]["metadata"]["dem_name"].split("_")[1][:4]

    # Define output name
    output_path = OUTPUT_DIR
    if lake_name is not None:
        output_name = os.path.join(output_path, "transects_combined", 
                                  f"profile_{lake_name}_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}-{yearstart}-{yearend}_{coreg_mode}.png")
    else:
        output_name = os.path.join(output_path, "transects_combined", 
                                  f"profile_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}-{yearstart}-{yearend}_{coreg_mode}.png")

    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    fig.savefig(output_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Combined profile plot saved to: {output_name}")
    return output_name


def plot_heatmap(all_profiles, reference_year=None, lake_name=None, output_path=None):
    """Create a heatmap showing elevation changes relative to a reference DEM.
    Red = higher than reference, Blue = lower than reference (drainage).
    
    Parameters
    ----------
    all_profiles : dict
        Dictionary containing elevation profiles
    reference_year : int, optional
        Year to use as reference profile (if None, uses oldest DEM)
    lake_name : str, optional
        Name of the lake for plot labeling
    output_path : str, optional
        Path to save the plot

    Returns
    -------
    str
        Path to the saved heatmap image
    """
    if not all_profiles["profiles"]:
        raise ValueError("No profile data available for plotting")
    
    # Sort profiles by date
    profiles_sorted = sorted(all_profiles["profiles"], 
                            key=lambda x: extract_date_obj(x["metadata"]["dem_name"]))
    
    # Filter out failed transects (those with mostly NaN values)
    valid_profiles = []
    for profile in profiles_sorted:
        profvalues = np.array(profile["profile_values"])
        valid_percentage = np.sum(~np.isnan(profvalues)) / len(profvalues)
        if valid_percentage > 0.5:  # Keep if more than 50% valid data
            valid_profiles.append(profile)
        else:
            print(f"  ⚠️ Excluding failed transect: {extract_date_label(profile['metadata']['dem_name'])} "
                  f"(only {valid_percentage:.1%} valid data)")
    
    if not valid_profiles:
        raise ValueError("No valid profiles after filtering")
    
    profiles_sorted = valid_profiles
    
    # Select reference profile (oldest by default)
    if reference_year is None:
        reference_profile = profiles_sorted[0]
    else:
        reference_profile = min(profiles_sorted, 
                               key=lambda x: abs(extract_year(x["metadata"]["dem_name"]) - reference_year))
    
    reference_date = extract_date_label(reference_profile["metadata"]["dem_name"])
    ref_values = np.array(reference_profile["profile_values"])
    transect_km = reference_profile["transect"] / 1000
    
    # Create difference matrix
    diff_matrix = []
    dates = []
    years = []
    
    for profile in profiles_sorted:
        profvalues = np.array(profile["profile_values"])
        # Calculate difference from reference
        diff = profvalues - ref_values
        
        # Mask out NaN values
        diff = np.ma.masked_invalid(diff)
        
        diff_matrix.append(diff)
        dates.append(extract_date_label(profile["metadata"]["dem_name"]))
        years.append(extract_year(profile["metadata"]["dem_name"]))
    
    diff_matrix = np.array(diff_matrix)
    
    # Create figure with 2 subplots (heatmap + time series)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
    
    # HEATMAP
    ax1 = fig.add_subplot(gs[0])
    
    # Set symmetric color limits around zero to show both positive and negative changes
    max_abs_diff = np.nanmax(np.abs(diff_matrix))
    vlim = max(5, max_abs_diff)  # At least 5 meters range
    
    # Use RdBu_r colormap (red for higher, blue for lower than reference)
    # RdBu_r: Red = higher elevation, Blue = lower elevation (drainage)
    im = ax1.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', 
                   extent=[0, transect_km[-1], len(dates)-0.5, -0.5],
                   vmin=-vlim, vmax=vlim, interpolation='nearest')
    
    # Customize heatmap
    # If there are many profiles, show only every every 3rd date label to avoid clutter:
    if len(dates) > 10:
        ax1.set_yticks(range(0, len(dates), 3))
        ax1.set_yticklabels([dates[i] for i in range(0, len(dates), 3)], fontsize=9)
    else:
        ax1.set_yticks(range(len(dates)))
        ax1.set_yticklabels(dates, fontsize=9)

    ax1.set_ylabel('Date', fontsize=12)
    ax1.set_xlabel('Distance along transect (km)', fontsize=12)
    ax1.set_title(f'Elevation Change Relative to {reference_date}\n(Red = Higher, Blue = Lower)', 
                 fontsize=12, pad=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Elevation Change (m)', fraction=0.05, pad=0.02)
    
    # Mark reference line
    ref_idx = profiles_sorted.index(reference_profile)
    ax1.axhline(y=ref_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(transect_km[-1] * 0.98, ref_idx, ' Reference', fontsize=8, 
            color='black', va='center', fontweight='bold')
    
    # Add gridlines for better readability
    ax1.set_xticks(np.arange(0, transect_km[-1] + 0.5, 0.5))
    ax1.grid(True, which='both', axis='both', linestyle=':', linewidth=0.3, alpha=0.5, color='gray')
    
    # TIME SERIES PLOT (Mean elevation change over time)
    ax2 = fig.add_subplot(gs[1])
    
    # Calculate mean change for each profile (excluding NaN)
    mean_changes = [np.nanmean(diff) for diff in diff_matrix]
    
    # Plot with color coding (red for positive, blue for negative)
    for i, (year, mean_change) in enumerate(zip(years, mean_changes)):
        color = 'red' if mean_change > 0 else 'blue' if mean_change < 0 else 'gray'
        alpha = 0.7
        marker = 'o'
        markersize = 6
        
        # Highlight large negative changes (potential drainage events)
        if mean_change < -2:
            marker = 's'
            markersize = 8
            color = 'darkred'
        
        ax2.plot(i, mean_change, color=color, marker=marker, markersize=markersize, 
                alpha=alpha, linestyle='none')
    
    # Connect points with line
    ax2.plot(range(len(mean_changes)), mean_changes, 'k-', linewidth=1, alpha=0.3)
    
    # Add zero line
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add horizontal bands for significance
    if len(mean_changes) > 3:
        # Calculate standard deviation of pre-reference period
        pre_ref_indices = [i for i, y in enumerate(years) if y < years[ref_idx]]
        if pre_ref_indices:
            pre_ref_changes = [mean_changes[i] for i in pre_ref_indices]
            std_pre_ref = np.nanstd(pre_ref_changes)
            ax2.fill_between([-0.5, len(mean_changes)-0.5], -std_pre_ref, std_pre_ref, 
                            alpha=0.15, color='gray', label=f'±1σ (pre-reference)')
    
    # Format time series plot
    # If there are many profiles, show only every 3rd date label to avoid clutter:
    if len(dates) > 10:
        ax2.set_xticks(range(0, len(dates), 3))
        ax2.set_xticklabels([dates[i] for i in range(0, len(dates), 3)], rotation=45, ha='right', fontsize=8)
    else:
        ax2.set_xticks(range(len(dates)))
        ax2.set_xticklabels(dates, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean Change (m)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    ax2.set_title('Mean Elevation Change Along Transect', fontsize=11, pad=10)
    ax2.legend(loc='best', fontsize=8)
    
    # Add drainage annotation if large drop detected
    if len(mean_changes) > 1:
        drops = [i for i in range(1, len(mean_changes)) 
                if mean_changes[i] - mean_changes[i-1] < -2]
        for drop_idx in drops:
            ax2.axvline(x=drop_idx, color='red', linestyle=':', alpha=0.5, linewidth=1)
            ax1.axhline(y=drop_idx, color='red', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Main title
    title_text = f'Elevation Change Heatmap'
    if lake_name:
        title_text = f'{lake_name} - {title_text}'
    
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save if output_path provided
    if output_path:
        coords = all_profiles.get("transect_coords", ((0,0), (0,0)))
        xs, ys = coords[0] if len(coords) > 0 else (0, 0)
        xe, ye = coords[1] if len(coords) > 1 else (0, 0)
        
        output_name = os.path.join(output_path, "transects_combined", 
                                  f"diff_heatmap_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}.png")
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        fig.savefig(output_name, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to: {output_name}")
    
    plt.show()
    return fig


def plot_relative_differences(all_profiles, coreg_mode, lake_name=None, output_path=None):
    """Create original-style plot alongside difference plot.
    
    Parameters    
    ----------
    all_profiles : dict
        Dictionary containing elevation profiles
    coreg_mode : str
        Coregistration mode for filename suffix
    lake_name : str, optional
        Name of the lake for plot labeling
    output_path : str, optional
        Path to save the plot
    
    Returns
    -------
    str
        Path to the saved plot
    """
    if not all_profiles["profiles"]:
        raise ValueError("No profile data available for plotting")
    
    # Sort profiles by date
    profiles_sorted = sorted(all_profiles["profiles"], 
                            key=lambda x: extract_date_obj(x["metadata"]["dem_name"]))
    
    # Use the oldest as reference
    reference_profile = profiles_sorted[0]
    reference_path = reference_profile["metadata"]["path"]
    reference_date = extract_date_label(reference_profile["metadata"]["dem_name"])
    ref_values = np.array(reference_profile["profile_values"])
    transect_m = reference_profile["transect"]
    
    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # LEFT PLOT: Original-style absolute elevations with viridis
    n_profiles = len(profiles_sorted)
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    
    for i, profile in enumerate(profiles_sorted):
        date_label = extract_date_label(profile["metadata"]["dem_name"])
        profvalues = np.array(profile["profile_values"])
        valid_mask = ~np.isnan(profvalues)
        
        if np.any(valid_mask):
            # Check if this is reference profile by comparing paths
            is_reference = (profile["metadata"]["path"] == reference_path)
            linewidth = 3 if is_reference else 1.5
            linestyle = '--' if is_reference else '-'
            alpha = 1.0 if is_reference else 0.7
            
            ax1.plot(transect_m[valid_mask], profvalues[valid_mask], 
                    color=colors[i], linewidth=linewidth, alpha=alpha, 
                    linestyle=linestyle, label=date_label if not is_reference else f'Reference ({date_label})')
    
    ax1.set_xlabel('Distance along transect (m)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
    # If there are many profiles, make a subsample for the legend of 10 elements:
    if len(all_profiles["profiles"]) > 10:
        handles, labels = ax1.get_legend_handles_labels()
        step = max(1, len(handles) // 10)
        ax1.legend(handles[::step], labels[::step], bbox_to_anchor=(1.02, 1), fontsize=8, loc="best",
                frameon=True, fancybox=True, shadow=True)
    else:
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
                frameon=True, fancybox=True, shadow=True)

    ax1.set_title(f'Elevation Profiles (viridis colormap)', fontsize=12, pad=12)
    
    # RIGHT PLOT: Difference plot
    for profile in profiles_sorted:
        # Skip reference profile by comparing paths
        if profile["metadata"]["path"] == reference_path:
            continue
            
        date_label = extract_date_label(profile["metadata"]["dem_name"])
        profvalues = np.array(profile["profile_values"])
        valid_mask = ~np.isnan(profvalues) & ~np.isnan(ref_values)
        
        if np.any(valid_mask):
            elevation_diff = profvalues[valid_mask] - ref_values[valid_mask]
            year = extract_year(profile["metadata"]["dem_name"])
            
            if year < extract_year(reference_profile["metadata"]["dem_name"]):
                color = 'lightcoral'
            else:
                # Blue intensity increases with time
                year_diff = min(year - extract_year(reference_profile["metadata"]["dem_name"]), 10)
                blue_intensity = 0.4 + (year_diff / 10) * 0.5
                color = plt.cm.viridis(blue_intensity)
            
            ax2.plot(transect_m[valid_mask], elevation_diff, 
                    color=color, linewidth=1.5, alpha=0.7, label=date_label)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('Distance along transect (m)', fontsize=12)
    ax2.set_ylabel('Elevation Change (m)', fontsize=12)
    ax2.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

    # If there are many profiles, make a subsample for the legend of 10 elements:
    if len(all_profiles["profiles"]) > 10:
        handles, labels = ax2.get_legend_handles_labels()
        step = max(1, len(handles) // 10)
        ax2.legend(handles[::step], labels[::step], bbox_to_anchor=(1.02, 1), fontsize=8, loc="best",
                frameon=True, fancybox=True, shadow=True)
    else:
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
                frameon=True, fancybox=True, shadow=True)

    ax2.set_title(f'Change Relative to {reference_date}', fontsize=12, pad=12)
    
    # Add A/B markers
    first_valid_profile = next((p for p in profiles_sorted if np.any(~np.isnan(p["profile_values"]))), None)
    if first_valid_profile:
        first_values = np.array(first_valid_profile["profile_values"])
        first_valid_mask = ~np.isnan(first_values)
        if np.any(first_valid_mask):
            ax1.annotate('A', xy=(transect_m[0], first_values[first_valid_mask][0]), 
                        xytext=(0.0, 1.01), textcoords='axes fraction', 
                        color='red', fontsize=13, fontweight='bold')
            ax1.annotate('B', xy=(transect_m[-1], first_values[first_valid_mask][-1]), 
                        xytext=(1.0, 1.01), textcoords='axes fraction', ha='right',
                        color='blue', fontsize=13, fontweight='bold')
            
            ax2.annotate('A', xy=(transect_m[0], 0), xytext=(0.0, 1.01), 
                        textcoords='axes fraction', color='red', fontsize=13, fontweight='bold')
            ax2.annotate('B', xy=(transect_m[-1], 0), xytext=(1.0, 1.01), 
                        textcoords='axes fraction', ha='right', color='blue', 
                        fontsize=13, fontweight='bold')
    
    # Main title
    date_start = extract_date_label(profiles_sorted[0]["metadata"]["dem_name"])
    date_end = extract_date_label(profiles_sorted[-1]["metadata"]["dem_name"])
    title_text = f'Elevation Analysis: {date_start} - {date_end}'
    if lake_name:
        title_text = f'{lake_name} - {title_text}'
    
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        coords = all_profiles.get("transect_coords", ((0,0), (0,0)))
        xs, ys = coords[0] if len(coords) > 0 else (0, 0)
        xe, ye = coords[1] if len(coords) > 1 else (0, 0)
        output_name = os.path.join(output_path, "transects_combined", 
                                  f"profile_diff_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}_{coreg_mode}.png")
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        fig.savefig(output_name, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_name}")
    
    plt.show()
    return fig
# ============================================================================
# MAIN WORKFLOW FUNCTIONS
# ============================================================================

def run_elevation_history(archdir, output_path=None, coreg_mode='none',
                         coords=None, time_range="2010-01-01/2026-12-31",
                         window_size=3, window_type='square'):
    """
    Complete workflow for elevation history analysis.
    
    Parameters
    ----------
    archdir : str
        Archive directory
    output_path : str, optional
        Output directory
    coreg_mode : str
        'none', 'altim', or 'mosaic'
    coords : tuple, optional
        (lon, lat) in WGS84
    time_range : str
        "YYYY-MM-DD/YYYY-MM-DD"
    window_size, window_type : as in get_elevation_window
        
    Returns
    -------
    dict
        Elevation history results
    """
    from elevation_utils import search_arcticdem_strips, filter_strip_dems, get_dem_metadata
    
    if output_path is None:
        output_path = OUTPUT_DIR
    
    # Get coordinates if not provided
    if coords is None:
        # Use interactive selection (requires ipyleaflet)
        from interactive_maps import select_point_interactive
        coords = select_point_interactive(timeout=30)
    
    print(f"Coordinates: ({coords[0]:.3f}, {coords[1]:.3f})")
    print(f"Time range: {time_range}")
    
    # STAC search
    bbox = (coords[0] - 0.001, coords[1] - 0.001,
            coords[0] + 0.001, coords[1] + 0.001)
    
    items_gdf, items = search_arcticdem_strips(bbox, time_range)
    items_gdf = filter_strip_dems(items_gdf, max_cloud_cover=0.2)
    
    pairnames, geocells, dates = get_dem_metadata(items_gdf)
    
    # Process history
    history = process_elevation_history(
        geocells, pairnames, dates, archdir, coords,
        window_size=window_size, window_type=window_type,
        coreg_mode=coreg_mode
    )
    
    # Plot
    print(f"\nGenerating elevation history plot to {output_path}...")
    plot_path = plot_elevation_history(history, output_path, coreg_mode)
    history['plot_path'] = plot_path
    
    # Save data
    year_i = time_range[:4]
    year_f = time_range[-10:-6] if '/' in time_range else time_range[-4:]
    
    if coreg_mode == 'none':
        suf = '_nc'
    elif coreg_mode == 'altim':
        suf = '_altim'
    else:
        suf = '_mosaic'
    
    data_path = get_output_path(
        'elevation_histories',
        f"elevation_history_{coords[0]:.3f}_{coords[1]:.3f}{suf}_{year_i}-{year_f}.txt"
    )
    
    with open(data_path, 'w') as f:
        f.write(f"Elevation History at ({coords[0]:.3f}, {coords[1]:.3f})\n")
        f.write("Date, Elevation (m), Pairname, Std, Valid Pixels\n")
        for date, elev, pair, std, vp in zip(
            history['dates'], history['elevations'],
            history['pairnames'], history['elevations_std'],
            history['valid_pixels']
        ):
            f.write(f"{date.isoformat()}, {elev:.2f}, {pair}, {std:.2f}, {vp}\n")
    
    print(f"Data saved: {data_path}")
    return history


def run_transect_analysis(archdir, output_path=None, coreg_mode='none',
                         transect_coords=None, time_range="2011-01-01/2026-12-31",
                         max_dems=18, lake_name=None):
    """Complete workflow for transect elevation profile analysis.
    
    Parameters
    ----------
    archdir : str
        Archive directory
    output_path : str, optional
        Output directory
    coreg_mode : str
        'none', 'altim', or 'mosaic'
    transect_coords : tuple, optional
        ((start_lon, start_lat), (end_lon, end_lat))
    time_range : str
        "YYYY-MM-DD/YYYY-MM-DD"
    max_dems : int
        Maximum number of DEMs to process
    lake_name : str, optional
        Name for plot titles
        
    Returns
    -------
    dict
        All profile data and plot paths
    """
    from elevation_utils import search_arcticdem_strips, filter_strip_dems, get_dem_metadata
    
    if output_path is None:
        output_path = OUTPUT_DIR
    
    # Get coordinates if not provided
    if transect_coords is None:
        from interactive_maps import select_transect_interactive
        transect_coords = select_transect_interactive(timeout=30)
    
    start, end = transect_coords
    print(f"Transect: ({start[0]:.3f}, {start[1]:.3f}) -> ({end[0]:.3f}, {end[1]:.3f})")
    
    # STAC search
    west = min(start[0], end[0]) - 0.001
    south = min(start[1], end[1]) - 0.001
    east = max(start[0], end[0]) + 0.001
    north = max(start[1], end[1]) + 0.001
    bbox = (west, south, east, north)
    
    items_gdf, items = search_arcticdem_strips(bbox, time_range)
    items_gdf = filter_strip_dems(items_gdf, max_items=max_dems)
    
    pairnames, geocells, dates = get_dem_metadata(items_gdf)
    
    # Process profiles
    all_profiles = process_elevation_profiles(
        transect_coords, pairnames, geocells, archdir,
        coreg_mode=coreg_mode
    )
    
    # Save data
    xs, ys = start
    xe, ye = end
    year_start = time_range[:4]
    year_end = time_range[-10:-6] if '/' in time_range else time_range[-4:]
    
    if lake_name:
        data_path = get_output_path(
            'transects_combined',
            f"combined_profiles_{lake_name}_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}_"
            f"{year_start}-{year_end}_{coreg_mode}.txt"
        )
    else:
        data_path = get_output_path(
            'transects_combined',
            f"combined_profiles_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}_"
            f"{year_start}-{year_end}_{coreg_mode}.txt"
        )
    
    with open(data_path, 'w') as f:
        f.write("Combined Elevation Profiles\n")
        f.write("DEM, StartX, StartY, EndX, EndY, Distance(m), MinElev(m), MaxElev(m)\n")
        for profile in all_profiles['profiles']:
            dem = profile['metadata']['dem_name']
            x0, y0, x1, y1 = profile['coords']
            dist = profile['distance']
            min_e = np.nanmin(profile['profile_values'])
            max_e = np.nanmax(profile['profile_values'])
            f.write(f"{dem}, {x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}, {dist:.1f}, {min_e:.1f}, {max_e:.1f}\n")
    
    print(f"Data saved: {data_path}")
    
    # Generate plots
    plot_path = plot_combined_profiles(
        all_profiles, coreg_mode=coreg_mode, lake_name=lake_name
    )
    all_profiles['plot_path'] = plot_path
    
    return all_profiles


def additional_plots(all_profiles, reference_year=None, coreg_mode='none', lake_name=None):
    """Generate additional plots for a transect analysis.
    
    Parameters
    ----------
    all_profiles : dict
        Output from run_transect_analysis
    reference_year : int, optional
        Year to use as reference for relative differences (default: None -> oldest DEM)
    coreg_mode : str, optional
        Coregistration mode
    lake_name : str, optional
        Name for plot titles
    output_path : str, optional
        Directory to save plots
    """
    output_path = OUTPUT_DIR

    print(f"\n=== Generating elevation change heatmap and relative difference plots ===")
    output_heatmap = plot_heatmap(all_profiles, reference_year=reference_year, lake_name=lake_name, output_path=output_path)
    all_profiles['heatmap_path'] = output_heatmap

    output_rel_diff = plot_relative_differences(all_profiles, coreg_mode=coreg_mode, lake_name=lake_name, output_path=output_path)
    all_profiles['relative_diff_path'] = output_rel_diff