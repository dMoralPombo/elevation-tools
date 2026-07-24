"""
DEM Difference Analysis
========================
Compute and visualize elevation differences between pairs of ArcticDEM strips.
Works directly with coregistered GeoTIFF files, handling overlapping regions
and different extents automatically.
"""

import gc
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio import warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer

# Import shared utilities
from config import OUTPUT_DIR, get_output_path


# ============================================================================
# DATE EXTRACTION
# ============================================================================

def extract_date_from_strip(strip_name):
    """Extract date from strip name.
    
    Handles names like:
    - 'SETSM_s2s041_WV02_20220428_...'
    - 'WV01_20240619_...'
    
    Parameters
    ----------
    strip_name : str
        Strip identifier
        
    Returns
    -------
    datetime or None
        Parsed date object
    """
    parts = strip_name.split("_")
    
    # Look for an 8-digit date string in any part
    for part in parts:
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d")
            except ValueError:
                pass
    
    return None


def format_date_label(date_obj):
    """Format datetime object to readable label.
    
    Parameters
    ----------
    date_obj : datetime or None
        
    Returns
    -------
    str
        Formatted date (e.g., 'Apr 2024') or 'Unknown'
    """
    if date_obj:
        return date_obj.strftime("%b %Y")
    return "Unknown"


# ============================================================================
# CORE DIFFERENCE COMPUTATION
# ============================================================================

def find_coregistered_dem(strip_name, archive_dir, coreg_suffix=None):
    """Find a coregistered DEM file matching a strip name.
    
    Parameters
    ----------
    strip_name : str
        Strip identifier (can be partial)
    archive_dir : str
        Base archive directory
    coreg_suffix : str, optional
        Suffix for coregistered files (e.g., '_mosaic_v_0-0_dh_0-0000_...')
        If None, searches for any coregistered file.
        
    Returns
    -------
    str or None
        Path to the found DEM file
    """
    # Try different search patterns
    patterns = []
    
    if coreg_suffix:
        patterns.append(os.path.join(archive_dir, f"*{strip_name}*{coreg_suffix}*coregistered.tif"))
        patterns.append(os.path.join(archive_dir, f"*{strip_name}*coregistered.tif"))
    else:
        patterns.append(os.path.join(archive_dir, f"*{strip_name}*coregistered.tif"))
    
    # Also try without coregistered suffix (for raw DEMs)
    patterns.append(os.path.join(archive_dir, f"*{strip_name}*.tif"))
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Try recursive search
    for pattern in patterns:
        recursive_pattern = os.path.join(archive_dir, "**", os.path.basename(pattern))
        matches = glob.glob(recursive_pattern, recursive=True)
        if matches:
            return matches[0]
    
    return None


def compute_dem_difference_from_geotiffs(strip_a, strip_b, archive_dir, 
                                         coreg_suffix=None, resampling=Resampling.bilinear):
    """Compute elevation difference between two strips directly from GeoTIFFs.
    
    Handles strips with different extents by computing the intersection
    and reprojecting/resampling to a common grid.
    
    Parameters
    ----------
    strip_a : str
        Name of recent/first strip (minuend) - can be partial match
    strip_b : str
        Name of old/second strip (subtrahend) - can be partial match
    archive_dir : str
        Base directory containing coregistered DEMs
    coreg_suffix : str, optional
        Suffix pattern for coregistered files
    resampling : rasterio Resampling method
        Resampling method for reprojection
        
    Returns
    -------
    diff_masked : np.ma.array or None
        Masked elevation difference (a - b)
    valid_mask : np.ndarray or None
        Boolean mask of valid pixels
    metadata : dict or None
        Dictionary with georeferencing information
    """
    # Find the DEM files
    dem_path_a = find_coregistered_dem(strip_a, archive_dir, coreg_suffix)
    dem_path_b = find_coregistered_dem(strip_b, archive_dir, coreg_suffix)
    
    if dem_path_a is None:
        print(f"  Could not find DEM for strip A: {strip_a}")
        return None, None, None
    if dem_path_b is None:
        print(f"  Could not find DEM for strip B: {strip_b}")
        return None, None, None
    
    print(f"  DEM A: {os.path.basename(dem_path_a)[:60]}...")
    print(f"  DEM B: {os.path.basename(dem_path_b)[:60]}...")
    
    try:
        with rio.open(dem_path_a) as src_a, rio.open(dem_path_b) as src_b:
            # Check if CRS matches
            if src_a.crs != src_b.crs:
                print(f"  ⚠ CRS mismatch: {src_a.crs} vs {src_b.crs}")
                # Reproject B to A's CRS
                print(f"  Reprojecting strip B to match strip A...")
                
                # Calculate the intersection in A's CRS
                bounds_a = src_a.bounds
                bounds_b = warp.transform_bounds(src_b.crs, src_a.crs, 
                                                 src_b.bounds.left, src_b.bounds.bottom,
                                                 src_b.bounds.right, src_b.bounds.top)
                
                # Compute intersection
                intersect_left = max(bounds_a.left, bounds_b[0])
                intersect_bottom = max(bounds_a.bottom, bounds_b[1])
                intersect_right = min(bounds_a.right, bounds_b[2])
                intersect_top = min(bounds_a.top, bounds_b[3])
                
                if intersect_left >= intersect_right or intersect_bottom >= intersect_top:
                    print("  ✗ No overlapping area between strips")
                    return None, None, None
                
                # Use A's resolution
                resolution = src_a.res[0]
                width = int((intersect_right - intersect_left) / resolution)
                height = int((intersect_top - intersect_bottom) / resolution)
                
                # Read and reproject
                arr_a = src_a.read(1, window=src_a.window(intersect_left, intersect_bottom,
                                                          intersect_right, intersect_top))
                
                # For B, read full and reproject
                arr_b_full = src_b.read(1)
                transform_b = src_b.transform
                crs_b = src_b.crs
                
                # Create destination array
                dst_transform = from_bounds(intersect_left, intersect_bottom,
                                           intersect_right, intersect_top,
                                           width, height)
                arr_b = np.zeros((height, width), dtype=arr_b_full.dtype)
                
                reproject(
                    source=arr_b_full,
                    destination=arr_b,
                    src_transform=transform_b,
                    src_crs=crs_b,
                    dst_transform=dst_transform,
                    dst_crs=src_a.crs,
                    resampling=resampling,
                )
                
                bounds = (intersect_left, intersect_bottom, intersect_right, intersect_top)
                transform = dst_transform
                crs = src_a.crs
                
            else:
                # Same CRS - calculate intersection directly
                bounds_a = src_a.bounds
                bounds_b = src_b.bounds
                
                intersect_left = max(bounds_a.left, bounds_b.left)
                intersect_bottom = max(bounds_a.bottom, bounds_b.bottom)
                intersect_right = min(bounds_a.right, bounds_b.right)
                intersect_top = min(bounds_a.top, bounds_b.top)
                
                if intersect_left >= intersect_right or intersect_bottom >= intersect_top:
                    print("  ✗ No overlapping area between strips")
                    return None, None, None
                
                # Read overlapping windows
                window_a = src_a.window(intersect_left, intersect_bottom,
                                       intersect_right, intersect_top)
                window_b = src_b.window(intersect_left, intersect_bottom,
                                       intersect_right, intersect_top)
                
                arr_a = src_a.read(1, window=window_a)
                arr_b = src_b.read(1, window=window_b)
                
                bounds = (intersect_left, intersect_bottom, intersect_right, intersect_top)
                transform = src_a.window_transform(window_a)
                crs = src_a.crs
            
            # Ensure same shape
            if arr_a.shape != arr_b.shape:
                min_h = min(arr_a.shape[0], arr_b.shape[0])
                min_w = min(arr_a.shape[1], arr_b.shape[1])
                arr_a = arr_a[:min_h, :min_w]
                arr_b = arr_b[:min_h, :min_w]
            
            # Get nodata values
            nodata_a = src_a.nodata if src_a.nodata is not None else -9999
            nodata_b = src_b.nodata if src_b.nodata is not None else -9999
            
            # Create valid pixel masks
            mask_a = (arr_a != nodata_a) & ~np.isnan(arr_a)
            mask_b = (arr_b != nodata_b) & ~np.isnan(arr_b)
            
            # Also mask extreme outliers
            mask_a &= (arr_a > -500) & (arr_a < 5000)
            mask_b &= (arr_b > -500) & (arr_b < 5000)
            
            # Valid where both strips have data
            valid_mask = mask_a & mask_b
            
            # Compute difference
            diff = arr_a.astype(np.float32) - arr_b.astype(np.float32)
            diff_masked = np.ma.array(diff, mask=~valid_mask)
            
            metadata = {
                'bounds': bounds,
                'transform': transform,
                'crs': crs,
                'shape': diff.shape,
                'dem_path_a': dem_path_a,
                'dem_path_b': dem_path_b,
            }
            
            return diff_masked, valid_mask, metadata
            
    except Exception as e:
        print(f"  ✗ Error computing difference: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def process_multiple_pairs(strip_pairs, archive_dir, coreg_suffix=None, 
                          output_dir=None, **kwargs):
    """Process multiple strip pairs from GeoTIFF files.
    
    Parameters
    ----------
    strip_pairs : list of tuples
        List of (strip_recent, strip_old) tuples (can be partial names)
    archive_dir : str
        Base directory containing coregistered DEMs
    coreg_suffix : str, optional
        Suffix pattern for coregistered files
    output_dir : str, optional
        Directory to save output files
    **kwargs
        Additional arguments passed to compute_dem_difference_from_geotiffs()
        
    Returns
    -------
    dict
        Results with (strip_a, strip_b) keys and (diff, mask, metadata) values
    """
    results = {}
    
    for strip_a, strip_b in strip_pairs:
        print(f"\nProcessing: {strip_a[:50]}...")
        print(f"         - {strip_b[:50]}...")
        
        diff, mask, metadata = compute_dem_difference_from_geotiffs(
            strip_a, strip_b, archive_dir, coreg_suffix, **kwargs
        )
        
        if diff is not None:
            results[(strip_a, strip_b)] = {
                'diff': diff,
                'mask': mask,
                'metadata': metadata,
            }
            
            if output_dir:
                _save_difference_geotiff(diff, mask, metadata, output_dir, strip_a, strip_b)
        else:
            print("  ✗ Failed to process pair")
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)}/{len(strip_pairs)} pairs successfully")
    print(f"{'='*60}")
    
    return results


def _save_difference_geotiff(diff, mask, metadata, output_dir, strip_a, strip_b):
    """Save elevation difference as GeoTIFF.
    
    Parameters
    ----------
    diff : np.ma.array
        Masked elevation difference
    mask : np.ndarray
        Valid pixel mask
    metadata : dict
        Georeferencing metadata
    output_dir : str
        Output directory
    strip_a, strip_b : str
        Strip names for filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create short names for filename
    short_a = strip_a[:30].replace('/', '_')
    short_b = strip_b[:30].replace('/', '_')
    
    output_path = output_dir / f"diff_{short_a}_vs_{short_b}.tif"
    
    # Prepare data for writing (fill masked values with nodata)
    out_data = diff.data.copy()
    out_data[~mask] = -9999
    
    with rio.open(
        str(output_path), 'w',
        driver='GTiff',
        height=diff.shape[0],
        width=diff.shape[1],
        count=1,
        dtype=np.float32,
        crs=metadata['crs'],
        transform=metadata['transform'],
        nodata=-9999,
        compress='lzw',
    ) as dst:
        dst.write(out_data.astype(np.float32), 1)
        dst.update_tags(
            STRIP_A=strip_a,
            STRIP_B=strip_b,
            DESCRIPTION=f"Elevation difference: {strip_a} - {strip_b}",
        )
    
    print(f"  ✓ Saved: {output_path.name}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def _secondary_axis_labels(left, right, bottom, top, n_ticks=4):
    """Compute EPSG:3413 tick positions and corresponding EPSG:4326 labels."""
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    
    x_ticks = np.linspace(left, right, n_ticks + 2)[1:-1]
    y_ticks = np.linspace(bottom, top, n_ticks + 2)[1:-1]
    
    lon_ticks, _ = transformer.transform(x_ticks, np.full_like(x_ticks, top))
    _, lat_ticks = transformer.transform(np.full_like(y_ticks, right), y_ticks)
    
    return x_ticks, y_ticks, lon_ticks, lat_ticks


def _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks):
    """Add secondary top/right axes with lon/lat labels."""
    secax_x = ax.secondary_xaxis("top")
    secax_x.set_xticks(x_ticks)
    secax_x.set_xticklabels([f"{lon:.2f}" for lon in lon_ticks])
    secax_x.set_xlabel("Longitude (°) - EPSG 4326", labelpad=12)
    secax_x.tick_params(labelsize=7)
    
    secax_y = ax.secondary_yaxis("right")
    secax_y.set_yticks(y_ticks)
    secax_y.set_yticklabels([f"{lat:.2f}" for lat in lat_ticks])
    secax_y.set_ylabel("Latitude (°) - EPSG 4326", labelpad=10, rotation=270)
    secax_y.tick_params(labelsize=7)


def _overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=None):
    """Overlay shapefile features on a matplotlib Axes."""
    try:
        import fiona
        from shapely.geometry import shape
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer as ProjTransformer
        from matplotlib.patches import PathPatch
        from matplotlib.collections import PatchCollection
        from matplotlib.path import Path
    except ImportError as e:
        print(f"  ⚠ Shapefile overlay skipped – missing dependency: {e}")
        return
    
    if style is None:
        style = dict(facecolor="none", edgecolor="#2b8a0f", linewidth=1.5, zorder=6)
    
    patches = []
    
    try:
        with fiona.open(shp_path) as src:
            shp_crs = src.crs_wkt if hasattr(src, "crs_wkt") else src.crs.to_wkt()
            reproj = ProjTransformer.from_crs(shp_crs, raster_crs, always_xy=True)
            
            def _reproj_coords(geom):
                return shp_transform(reproj.transform, shape(geom))
            
            for feat in src:
                geom_raw = feat.get("geometry")
                if geom_raw is None:
                    continue
                try:
                    geom = _reproj_coords(geom_raw)
                except Exception:
                    continue
                
                gtype = geom.geom_type
                
                if gtype in ("Polygon", "MultiPolygon"):
                    polys = [geom] if gtype == "Polygon" else list(geom.geoms)
                    for poly in polys:
                        coords = list(poly.exterior.coords)
                        codes = ([Path.MOVETO] + [Path.LINETO] * (len(coords) - 2) 
                                + [Path.CLOSEPOLY])
                        path = Path(coords, codes)
                        patches.append(PathPatch(path))
                
                elif gtype in ("LineString", "MultiLineString"):
                    lines = [geom] if gtype == "LineString" else list(geom.geoms)
                    for line in lines:
                        xs, ys = line.xy
                        ax.plot(xs, ys, **{k: v for k, v in style.items() 
                                           if k not in ("facecolor",)})
                
                elif gtype in ("Point", "MultiPoint"):
                    pts = [geom] if gtype == "Point" else list(geom.geoms)
                    for pt in pts:
                        ax.plot(pt.x, pt.y, "o", 
                               color=style.get("edgecolor", "red"),
                               markersize=3, zorder=style.get("zorder", 6))
        
        if patches:
            pc = PatchCollection(patches, **style)
            ax.add_collection(pc)
        
        print(f"  ✓ Shapefile overlaid: {os.path.basename(shp_path)}")
    
    except Exception as e:
        print(f"  ⚠ Could not overlay shapefile {shp_path}: {e}")


def plot_elevation_difference(
    diff_masked,
    valid_mask,
    strip_a,
    strip_b,
    output_dir=None,
    tile=None,
    cbar_title="Elevation Difference (m)",
    cmap="RdBu_r",
    add_grid=True,
    vmin=None,
    vmax=None,
    max_pixels=2000,
    auto_crop=True,
    crop_threshold=0.5,
    extent=None,
    shp_path=None,
    shp_style=None,
    metadata=None,
):
    """Plot elevation difference between two DEM strips.
    
    Parameters
    ----------
    diff_masked : np.ma.array
        Masked elevation difference array
    valid_mask : np.ndarray
        Boolean valid-pixel mask
    strip_a : str
        Recent strip name (minuend)
    strip_b : str
        Old strip name (subtrahend)
    output_dir : str, optional
        Output directory for saving plot
    tile : str, optional
        Tile identifier for title and filename
    cbar_title : str
        Colorbar label
    cmap : str
        Colormap name
    add_grid : bool
        Whether to draw gridlines
    vmin, vmax : float or None
        Colormap limits
    max_pixels : int
        Maximum array dimension after downsampling
    auto_crop : bool
        Crop to valid-data bounding box
    crop_threshold : float
        Fraction of total size below which cropping triggers
    extent : tuple or None
        (left, right, bottom, top) - auto-detected from metadata if None
    shp_path : str, optional
        Path to shapefile for overlay
    shp_style : dict, optional
        Style for shapefile features
    metadata : dict, optional
        Georeferencing metadata from compute_dem_difference_from_geotiffs()
        
    Returns
    -------
    str
        Path to saved PNG file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Extract dates
    date_a = extract_date_from_strip(strip_a)
    date_b = extract_date_from_strip(strip_b)
    date_label_a = format_date_label(date_a)
    date_label_b = format_date_label(date_b)
    
    # Build titles
    if tile:
        main_title = f"{cbar_title} — {tile}"
    else:
        main_title = f"{cbar_title}"
    
    subtitle = f"{date_label_a}  −  {date_label_b}"
    strip_note = f"{strip_a[:40]}   vs   {strip_b[:40]}"
    
    interactive_mode = hasattr(sys, "ps1") or sys.flags.interactive
    
    # Get extent from metadata if not provided
    if extent is None and metadata is not None:
        bounds = metadata.get('bounds')
        if bounds:
            extent = (bounds[0], bounds[1], bounds[2], bounds[3])
    
    # Find valid region
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        print(f"  No valid data for {strip_a} - {strip_b}")
        return None
    
    row_min, row_max = valid_rows.min(), valid_rows.max() + 1
    col_min, col_max = valid_cols.min(), valid_cols.max() + 1
    
    valid_height = row_max - row_min
    valid_width = col_max - col_min
    total_height, total_width = diff_masked.shape
    
    # Auto-crop
    should_crop = False
    if auto_crop:
        if (valid_height / total_height < crop_threshold or 
            valid_width / total_width < crop_threshold):
            should_crop = True
            print(f"  Auto-cropping: valid region {valid_width}×{valid_height} "
                  f"/ {total_width}×{total_height}")
    
    crop_row_min = crop_col_min = 0
    crop_row_max, crop_col_max = total_height, total_width
    
    if should_crop:
        pad = 50
        crop_row_min = max(0, row_min - pad)
        crop_row_max = min(total_height, row_max + pad)
        crop_col_min = max(0, col_min - pad)
        crop_col_max = min(total_width, col_max + pad)
        
        raster_data = diff_masked[crop_row_min:crop_row_max,
                                   crop_col_min:crop_col_max].copy()
        crop_valid_mask = valid_mask[crop_row_min:crop_row_max,
                                      crop_col_min:crop_col_max]
        
        # Adjust extent for cropping
        if extent is not None:
            left, bottom, right, top = extent
            pixel_width = (right - left) / total_width
            pixel_height = (top - bottom) / total_height
            extent = (
                left + crop_col_min * pixel_width,
                left + crop_col_max * pixel_width,
                bottom + (total_height - crop_row_max) * pixel_height,
                bottom + (total_height - crop_row_min) * pixel_height,
            )
    else:
        raster_data = diff_masked.copy()
        crop_valid_mask = valid_mask
    
    plot_height, plot_width = raster_data.shape
    
    # Downsample if needed
    downsample_factor = max(1, int(np.ceil(max(plot_height, plot_width) / max_pixels)))
    
    if downsample_factor > 1:
        h_new = plot_height // downsample_factor
        w_new = plot_width // downsample_factor
        raster_ds = (
            raster_data[:h_new * downsample_factor, :w_new * downsample_factor]
            .reshape(h_new, downsample_factor, w_new, downsample_factor)
            .mean(axis=(1, 3))
        )
        print(f"  Downsampled {downsample_factor}x: "
              f"{plot_width}×{plot_height} → {raster_ds.shape[1]}×{raster_ds.shape[0]}")
        
        # Adjust extent for downsampling
        if extent is not None:
            left, bottom, right, top = extent
            extent = (left, bottom, 
                     left + (right - left) * (w_new / plot_width),
                     bottom + (top - bottom) * (h_new / plot_height))
    else:
        raster_ds = raster_data
    
    # Colormap limits
    if vmin is None or vmax is None:
        valid_data = raster_ds.compressed()
        if len(valid_data) > 0:
            abs_max = np.percentile(np.abs(valid_data), 99)
            vmin = -abs_max if vmin is None else vmin
            vmax = abs_max if vmax is None else vmax
    
    # Colormap with black for no-data
    import matplotlib.cm as mcm
    cmap_obj = mcm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    cmap_obj.set_under("black")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("black")
    
    interpolation = "bilinear" if downsample_factor > 1 else "nearest"
    
    if extent is not None:
        left, right, bottom, top = extent
        plot_extent = (left, right, bottom, top)
    else:
        plot_extent = (0, plot_width, 0, plot_height)
    
    img = ax.imshow(
        raster_ds, cmap=cmap_obj, vmin=vmin, vmax=vmax,
        extent=plot_extent, origin="upper",
        interpolation=interpolation, aspect="equal",
    )
    
    # Axis labels
    if extent is not None:
        left, right, bottom, top = extent
        x_ticks, y_ticks, lon_ticks, lat_ticks = _secondary_axis_labels(
            left, right, bottom, top
        )
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("X (m) - EPSG 3413", fontsize=9)
        ax.set_ylabel("Y (m) - EPSG 3413", fontsize=9)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks)
        
        # Shapefile overlay
        if shp_path is not None:
            _overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=shp_style)
    else:
        ax.set_xlabel("Column (pixels)", fontsize=9)
        ax.set_ylabel("Row (pixels)", fontsize=9)
        ax.tick_params(labelsize=7)
    
    # Titles
    ax.set_title(main_title, fontweight="bold", fontsize=12, y=1.06)
    ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, fontsize=9,
            color="#666666", ha="center", va="bottom", fontweight="normal")
    ax.text(0.5, 1.015, strip_note, transform=ax.transAxes, fontsize=7,
            color="#999999", ha="center", va="bottom", 
            fontweight="normal", style="italic")
    
    # Grid
    if add_grid:
        ax.grid(visible=True, which="both", color="gray", 
               linestyle="--", linewidth=0.4, alpha=0.6)
    
    # Colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", pad=0.08,
                       fraction=0.030, shrink=0.75)
    cbar.set_label(cbar_title, labelpad=10, rotation=270, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    
    # Statistics box
    valid_data = raster_ds.compressed()
    n_valid = int(np.sum(crop_valid_mask))
    
    if len(valid_data) > 0:
        stats_text = (
            f"Valid pixels: {n_valid:,}\n"
            f"Mean: {np.mean(valid_data):.2f} m\n"
            f"Std:  {np.std(valid_data):.2f} m\n"
            f"Min:  {np.min(valid_data):.2f} m\n"
            f"Max:  {np.max(valid_data):.2f} m"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=7,
               verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
    
    plt.tight_layout()
    
    # Save
    image_dir = Path(output_dir) / "dem_differences" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    short_a = strip_a[:20].replace('/', '_')
    short_b = strip_b[:20].replace('/', '_')
    tile_str = f"{tile}_" if tile else ""
    
    png_path = image_dir / f"{tile_str}diff_{short_a}_vs_{short_b}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    print(f"  ✓ PNG saved: {png_path}")
    
    if interactive_mode:
        plt.show()
    else:
        plt.close(fig)
    
    # Clean up
    del raster_data, raster_ds, img
    gc.collect()
    
    return str(png_path)


def run_dem_difference(strip_pairs, archive_dir, output_dir=None,
                      coreg_suffix=None, tile=None, shp_path=None, **plot_kwargs):
    """Complete workflow for DEM difference analysis from GeoTIFFs.
    
    Parameters
    ----------
    strip_pairs : list of tuples
        List of (strip_recent, strip_old) tuples (partial names OK)
    archive_dir : str
        Base directory containing coregistered DEMs
    output_dir : str, optional
        Output directory (defaults to config OUTPUT_DIR)
    coreg_suffix : str, optional
        Suffix for coregistered files (e.g., '_mosaic_v_0-0_dh_0-0000_...')
    tile : str, optional
        Tile identifier for plot titles
    shp_path : str, optional
        Path to shapefile for overlay
    **plot_kwargs
        Additional arguments passed to plot_elevation_difference()
        
    Returns
    -------
    dict
        Results with statistics and plot paths
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print(f"\n{'='*60}")
    print(f"DEM Difference Analysis")
    print(f"{'='*60}")
    print(f"Number of pairs: {len(strip_pairs)}")
    print(f"Archive directory: {archive_dir}")
    if tile:
        print(f"Tile: {tile}")
    
    # Process pairs
    results_raw = process_multiple_pairs(
        strip_pairs, archive_dir, coreg_suffix, output_dir
    )
    
    # Plot and collect statistics
    all_results = {}
    
    for (strip_a, strip_b), data in results_raw.items():
        diff = data['diff']
        mask = data['mask']
        metadata = data['metadata']
        
        print(f"\n{strip_a[:40]}... - {strip_b[:40]}...:")
        print(f"  Valid pixels: {np.sum(mask):,} / {mask.size:,}")
        print(f"  Mean difference: {np.mean(diff.compressed()):.2f} m")
        print(f"  Std difference: {np.std(diff.compressed()):.2f} m")
        print(f"  Min/Max: {np.min(diff.compressed()):.2f} / "
              f"{np.max(diff.compressed()):.2f} m")
        print(f"  Overlap area shape: {diff.shape}")
        
        # Plot
        png_path = plot_elevation_difference(
            diff, mask, strip_a, strip_b,
            output_dir=output_dir,
            tile=tile,
            shp_path=shp_path,
            metadata=metadata,
            **plot_kwargs
        )
        
        all_results[(strip_a, strip_b)] = {
            'diff': diff,
            'mask': mask,
            'metadata': metadata,
            'plot_path': png_path,
            'stats': {
                'valid_pixels': int(np.sum(mask)),
                'mean': float(np.mean(diff.compressed())),
                'std': float(np.std(diff.compressed())),
                'min': float(np.min(diff.compressed())),
                'max': float(np.max(diff.compressed())),
                'shape': diff.shape,
            }
        }
    
    print(f"\n{'='*60}")
    print(f"Analysis complete. {len(all_results)} pairs processed.")
    print(f"{'='*60}")
    
    return all_results