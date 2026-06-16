"""
DEM Difference Analysis
========================
Compute and visualize elevation differences between pairs of ArcticDEM strips.
Uses pre-computed zarr files for efficient processing.
"""

import gc
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import ListedColormap
from pyproj import Transformer

# Import shared utilities
from config import OUTPUT_DIR, get_output_path


# ============================================================================
# DATE EXTRACTION (consistent with analysis.py)
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

def compute_dem_difference(strip_a, strip_b, zarr_dir):
    """Compute elevation difference between two strips using zarr files.
    
    Calculates: strip_a - strip_b (direct subtraction of elevations).
    Masks pixels with -9999 values and where both strips don't have valid data.
    
    Parameters
    ----------
    strip_a : str
        Name of recent/first strip (minuend)
    strip_b : str
        Name of old/second strip (subtrahend)
    zarr_dir : str
        Directory containing zarr files
        
    Returns
    -------
    diff : np.ma.array or None
        Masked elevation difference (a - b)
    valid_mask : np.ndarray or None
        Boolean mask of valid pixels
    """
    zarr_dir = Path(zarr_dir)
    
    # Find zarr files (handle partial name matching)
    zarr_path_a = glob.glob(str(zarr_dir / f"processed_{strip_a[:30]}*.zarr"))
    zarr_path_b = glob.glob(str(zarr_dir / f"processed_{strip_b[:30]}*.zarr"))
    
    if not zarr_path_a or not zarr_path_b:
        print(f"  Could not find zarr files for {strip_a[:30]} or {strip_b[:30]}")
        return None, None
    
    # Open zarr files
    try:
        store_a = zarr.open(zarr_path_a[0], mode="r")
        store_b = zarr.open(zarr_path_b[0], mode="r")
    except Exception as e:
        print(f"  Error opening zarr files: {e}")
        return None, None
    
    # Extract arrays from zarr (handle both Array and Group)
    try:
        if isinstance(store_a, zarr.Array):
            arr_a = store_a[:]
        else:
            first_key = list(store_a.keys())[0]
            arr_a = store_a[first_key][:]
    except Exception as e:
        print(f"  Error extracting array from store_a: {e}")
        return None, None
    
    try:
        if isinstance(store_b, zarr.Array):
            arr_b = store_b[:]
        else:
            first_key = list(store_b.keys())[0]
            arr_b = store_b[first_key][:]
    except Exception as e:
        print(f"  Error extracting array from store_b: {e}")
        return None, None
    
    # Ensure same shape
    if arr_a.shape != arr_b.shape:
        print(f"  Warning: Shape mismatch - a: {arr_a.shape}, b: {arr_b.shape}")
        min_rows = min(arr_a.shape[0], arr_b.shape[0])
        min_cols = min(arr_a.shape[1], arr_b.shape[1])
        arr_a = arr_a[:min_rows, :min_cols]
        arr_b = arr_b[:min_rows, :min_cols]
    
    # Create valid pixel masks
    mask_a = (arr_a != -9999) & ~np.isnan(arr_a)
    mask_b = (arr_b != -9999) & ~np.isnan(arr_b)
    
    # Valid where both strips have data
    valid_mask = mask_a & mask_b
    
    # Compute difference and mask invalid pixels
    diff = arr_a - arr_b
    diff_masked = np.ma.array(diff, mask=~valid_mask)
    
    return diff_masked, valid_mask


def process_multiple_pairs(strip_pairs, zarr_dir, output_dir=None):
    """Process multiple strip pairs and optionally save results.
    
    Parameters
    ----------
    strip_pairs : list of tuples
        List of (strip_recent, strip_old) tuples
    zarr_dir : str
        Directory containing zarr files
    output_dir : str, optional
        Directory to save output zarr files
        
    Returns
    -------
    dict
        Results with (strip_a, strip_b) keys and (diff, mask) values
    """
    results = {}
    
    for strip_a, strip_b in strip_pairs:
        print(f"Processing: {strip_a[:40]}... - {strip_b[:40]}...")
        diff, mask = compute_dem_difference(strip_a, strip_b, zarr_dir)
        
        if diff is not None:
            results[(strip_a, strip_b)] = (diff, mask)
            
            if output_dir:
                output_path = Path(output_dir) / f"diff_{strip_a[:30]}_{strip_b[:30]}.zarr"
                _save_as_zarr(diff, mask, output_path, strip_a, strip_b)
                print(f"  Saved to {output_path}")
        else:
            print("  Failed to process pair")
    
    return results


def _save_as_zarr(diff, mask, output_path, strip_a, strip_b):
    """Save elevation difference as zarr file.
    
    Parameters
    ----------
    diff : np.ma.array
        Masked elevation difference
    mask : np.ndarray
        Valid pixel mask
    output_path : str or Path
        Output path
    strip_a, strip_b : str
        Strip names for metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    store = zarr.open(str(output_path), mode="w")
    store.create_dataset("elevation_diff", data=diff.data, compressor=zarr.Blosc())
    store.create_dataset("valid_mask", data=mask, compressor=zarr.Blosc())
    
    store.attrs["strip_recent"] = strip_a
    store.attrs["strip_old"] = strip_b
    store.attrs["description"] = f"Elevation difference: {strip_a} - {strip_b}"


# ============================================================================
# VISUALIZATION
# ============================================================================

def _secondary_axis_labels(left, right, bottom, top, n_ticks=4):
    """Compute EPSG:3413 tick positions and corresponding EPSG:4326 labels.
    
    Parameters
    ----------
    left, right, bottom, top : float
        Bounds in EPSG:3413 meters
    n_ticks : int
        Number of interior ticks
        
    Returns
    -------
    tuple
        (x_ticks, y_ticks, lon_ticks, lat_ticks)
    """
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    
    x_ticks = np.linspace(left, right, n_ticks + 2)[1:-1]
    y_ticks = np.linspace(bottom, top, n_ticks + 2)[1:-1]
    
    lon_ticks, _ = transformer.transform(x_ticks, np.full_like(x_ticks, top))
    _, lat_ticks = transformer.transform(np.full_like(y_ticks, right), y_ticks)
    
    return x_ticks, y_ticks, lon_ticks, lat_ticks


def _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks):
    """Add secondary top/right axes with lon/lat labels.
    
    Parameters
    ----------
    ax : matplotlib Axes
    x_ticks, y_ticks : ndarray
        Primary axis tick positions (EPSG:3413)
    lon_ticks, lat_ticks : ndarray
        Corresponding WGS84 coordinates
    """
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
    """Overlay shapefile features on a matplotlib Axes.
    
    Parameters
    ----------
    ax : matplotlib Axes
    shp_path : str
        Path to shapefile
    raster_crs : str
        Target CRS matching the axes data coordinates
    style : dict, optional
        Style kwargs for the features
    """
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
    tile,
    output_dir=None,
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
):
    """Plot elevation difference between two DEM strips.
    
    Creates a publication-quality figure with dual EPSG:3413/EPSG:4326 axes,
    statistics box, and optional shapefile overlay.
    
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
    tile : str
        Tile identifier
    output_dir : str, optional
        Output directory for saving plot
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
        (left, right, bottom, top) in EPSG:3413 meters
    shp_path : str, optional
        Path to shapefile for overlay
    shp_style : dict, optional
        Style for shapefile features
        
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
    main_title = f"{cbar_title} — {tile}"
    subtitle = f"{date_label_a}  −  {date_label_b}"
    strip_note = f"{strip_a[:30]}   vs   {strip_b[:30]}"
    
    interactive_mode = hasattr(sys, "ps1") or sys.flags.interactive
    
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
    else:
        raster_data = diff_masked.copy()
        crop_valid_mask = valid_mask
    
    plot_height, plot_width = raster_data.shape
    
    # Adjust extent for cropping
    if extent is not None and should_crop:
        left, bottom, right, top = extent
        pixel_width = (right - left) / total_width
        pixel_height = (top - bottom) / total_height
        extent = (
            left + crop_col_min * pixel_width,
            left + crop_col_max * pixel_width,
            bottom + (total_height - crop_row_max) * pixel_height,
            bottom + (total_height - crop_row_min) * pixel_height,
        )
    
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
    image_dir = Path(output_dir) / "dem_differences" / tile[:5] / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = image_dir / f"{tile}_diff_{strip_a[:30]}_{strip_b[:30]}.png"
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


def run_dem_difference(tile, strip_pairs, zarr_dir, output_dir=None,
                       extent=None, shp_path=None, **plot_kwargs):
    """Complete workflow for DEM difference analysis.
    
    Parameters
    ----------
    tile : str
        Tile identifier (e.g., '31_38_1_1')
    strip_pairs : list of tuples
        List of (strip_recent, strip_old) tuples
    zarr_dir : str
        Directory containing pre-computed zarr files
    output_dir : str, optional
        Output directory (defaults to config OUTPUT_DIR)
    extent : tuple, optional
        (left, right, bottom, top) in EPSG:3413
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
    print(f"DEM Difference Analysis - Tile: {tile}")
    print(f"{'='*60}")
    print(f"Number of pairs: {len(strip_pairs)}")
    print(f"Zarr directory: {zarr_dir}")
    
    # Process pairs
    results = process_multiple_pairs(strip_pairs, zarr_dir, output_dir)
    
    # Plot and collect statistics
    all_results = {}
    
    for (strip_a, strip_b), (diff, mask) in results.items():
        print(f"\n{strip_a[:40]}... - {strip_b[:40]}...:")
        print(f"  Valid pixels: {np.sum(mask):,} / {mask.size:,}")
        print(f"  Mean difference: {np.mean(diff.compressed()):.2f} m")
        print(f"  Std difference: {np.std(diff.compressed()):.2f} m")
        print(f"  Min/Max: {np.min(diff.compressed()):.2f} / "
              f"{np.max(diff.compressed()):.2f} m")
        
        # Plot
        png_path = plot_elevation_difference(
            diff, mask, strip_a, strip_b, tile,
            output_dir=output_dir,
            extent=extent,
            shp_path=shp_path,
            **plot_kwargs
        )
        
        all_results[(strip_a, strip_b)] = {
            'diff': diff,
            'mask': mask,
            'plot_path': png_path,
            'stats': {
                'valid_pixels': int(np.sum(mask)),
                'mean': float(np.mean(diff.compressed())),
                'std': float(np.std(diff.compressed())),
                'min': float(np.min(diff.compressed())),
                'max': float(np.max(diff.compressed())),
            }
        }
    
    print(f"\n{'='*60}")
    print(f"Analysis complete. {len(all_results)} pairs processed.")
    print(f"{'='*60}")
    
    return all_results