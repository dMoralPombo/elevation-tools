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

# Import utilities
from elevation_utils import (
    wgs84_to_3413, find_and_unzip, get_elevation_window,
    read_elevation_from_compressed, extract_elevation_profile
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
                    extract_elevation_profile(transect_coords, raster_path, num_samples)
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
                          cmap='terrain', lake_name=None):
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
        
    Returns
    -------
    str
        Path to saved plot
    """
    if not all_profiles['profiles']:
        raise ValueError("No profiles to plot")
    
    # Use first DEM for map
    first = all_profiles['profiles'][0]
    x0, y0, x1, y1 = first['coords']
    
    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2], wspace=0.2)
    
    # Left: DEM with transect
    ax1 = fig.add_subplot(gs[0])
    raster_path = find_and_unzip(first['metadata']['path'])
    
    with rio.open(raster_path) as src:
        data = src.read(1)
        data_masked = np.ma.masked_equal(data, src.nodata)
        
        ax1.imshow(data_masked, cmap=cmap,
                  extent=(src.bounds.left, src.bounds.right,
                         src.bounds.bottom, src.bounds.top),
                  origin='upper', aspect=1.2, rasterized=True)
    
    # Plot transect line
    ax1.plot([x0, x1], [y0, y1], 'k--', linewidth=1.5)
    ax1.plot(x0, y0, 'ro', markersize=8, markeredgecolor='white', label='Start (A)')
    ax1.plot(x1, y1, 'bo', markersize=8, markeredgecolor='white', label='End (B)')
    
    dem_name = first['metadata']['dem_name']
    ax1.set_title(f"Reference DEM\n{dem_name}", pad=12, fontsize=12)
    ax1.set_xlabel("X (m) EPSG:3413", fontsize=11)
    ax1.set_ylabel("Y (m) EPSG:3413", fontsize=11)
    ax1.legend(fontsize=10)
    
    # Right: All profiles
    ax2 = fig.add_subplot(gs[1])
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(all_profiles['profiles'])))
    
    for i, profile in enumerate(all_profiles['profiles']):
        label = extract_date_label(profile['metadata']['dem_name'])
        elevations = np.array(profile['profile_values'])
        
        # Filter outliers
        valid = elevations[~np.isnan(elevations)]
        if len(valid) > 0:
            median, std = np.nanmedian(valid), np.nanstd(valid)
            mask = (elevations > median + 3*std) | (elevations < median - 3*std)
            elevations = np.where(mask, np.nan, elevations)
        
        ax2.plot(profile['transect'], elevations, color=colors[i],
                linewidth=1.5, label=label)
    
    ax2.set_xlabel("Distance along transect (m)", fontsize=11)
    ax2.set_ylabel("Elevation (m)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    # Legend handling
    if len(all_profiles['profiles']) > 10:
        handles, labels = ax2.get_legend_handles_labels()
        step = max(1, len(handles) // 10)
        ax2.legend(handles[::step], labels[::step], fontsize=8, loc='best')
    else:
        ax2.legend(fontsize=8, loc='best')
    
    # Title
    year_start = extract_year(all_profiles['profiles'][-1]['metadata']['dem_name'])
    year_end = extract_year(all_profiles['profiles'][0]['metadata']['dem_name'])
    title = f"Elevation Profiles: {year_start}-{year_end}"
    if lake_name:
        title = f"{lake_name} - {title}"
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    coords = all_profiles['transect_coords']
    xs, ys = coords[0]
    xe, ye = coords[1]
    
    if lake_name:
        output_path = get_output_path(
            'transects_combined',
            f"profile_{lake_name}_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}_{coreg_mode}.png"
        )
    else:
        output_path = get_output_path(
            'transects_combined',
            f"profile_{xs:.3f}_{ys:.3f}_{xe:.3f}_{ye:.3f}_{coreg_mode}.png"
        )
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


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