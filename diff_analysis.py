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
from typing import List, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio import warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.enums import Resampling as RioResampling
from pyproj import Transformer
import zarr
import fiona
from pathlib import Path
from shapely.geometry import shape
from matplotlib.patches import PathPatch
from shapely.ops import transform as shp_transform
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath

# Import shared utilities
from config import OUTPUT_DIR, get_output_path

# ============================================================================
# ZARR METADATA ANALYSIS
# ===========================================================================
def check_zarr_metadata(zarr_dir, tile=None, max_files=5):
    """Check metadata and extent of zarr files in a directory.
    
    Parameters
    ----------
    zarr_dir : str
        Directory containing zarr files
    tile : str, optional
        Expected tile identifier for comparison
    max_files : int
        Maximum number of files to check (to avoid overwhelming output)
        
    Returns
    -------
    dict
        Dictionary with extent info for each checked file
    """
    import zarr
    
    zarr_dir = Path(zarr_dir)
    
    if not zarr_dir.exists():
        print(f"  ✗ Directory not found: {zarr_dir}")
        return {}
    
    zarr_files = sorted(glob.glob(str(zarr_dir / "processed_*.zarr")))
    
    if not zarr_files:
        # Try alternative patterns
        zarr_files = sorted(glob.glob(str(zarr_dir / "*.zarr")))
    
    if not zarr_files:
        print(f"  ✗ No zarr files found in {zarr_dir}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"CHECKING ZARR METADATA - {zarr_dir}")
    print(f"{'='*80}")
    print(f"Total zarr files: {len(zarr_files)}")
    print(f"Checking first {min(max_files, len(zarr_files))} files...\n")
    
    results = {}
    
    for i, zf in enumerate(zarr_files[:max_files]):
        basename = os.path.basename(zf)
        print(f"{'─'*80}")
        print(f"File {i+1}: {basename[:70]}...")
        
        try:
            store = zarr.open(zf, mode="r")
            
            # Check if it's an Array or Group
            if isinstance(store, zarr.Array):
                arr = store[:]
                print(f"  Type: Array")
                print(f"  Shape: {arr.shape}")
                print(f"  Dtype: {arr.dtype}")
                
                # Check attributes
                print(f"  Attributes:")
                for key, value in store.attrs.items():
                    print(f"    {key}: {value}")
                    
                results[basename] = {
                    'shape': arr.shape,
                    'attrs': dict(store.attrs),
                }
                
            elif isinstance(store, zarr.Group):
                print(f"  Type: Group")
                keys = list(store.keys())
                print(f"  Keys: {keys}")
                
                # Check first array
                if keys:
                    first_key = keys[0]
                    arr = store[first_key][:]
                    print(f"  First array '{first_key}':")
                    print(f"    Shape: {arr.shape}")
                    print(f"    Dtype: {arr.dtype}")
                
                # Check group attributes
                print(f"  Group attributes:")
                for key, value in store.attrs.items():
                    print(f"    {key}: {value}")
                
                # Check array attributes
                if keys:
                    print(f"  Array '{first_key}' attributes:")
                    for key, value in store[first_key].attrs.items():
                        print(f"    {key}: {value}")
                
                results[basename] = {
                    'type': 'group',
                    'keys': keys,
                    'shape': arr.shape if keys else None,
                    'group_attrs': dict(store.attrs),
                    'array_attrs': dict(store[first_key].attrs) if keys else {},
                }
            
            # Look for extent/bounds/transform in attributes
            all_attrs = {}
            if isinstance(store, zarr.Array):
                all_attrs = dict(store.attrs)
            else:
                all_attrs = dict(store.attrs)
                if keys:
                    all_attrs.update(dict(store[first_key].attrs))
            
            # Common metadata keys for georeferencing
            geo_keys = ['bounds', 'extent', 'transform', 'crs', 'geotransform',
                       'left', 'right', 'bottom', 'top', 'x_min', 'x_max', 
                       'y_min', 'y_max', 'bbox', 'epsg', 'projection']
            
            found_geo = {k: v for k, v in all_attrs.items() 
                        if any(gk in str(k).lower() for gk in geo_keys)}
            
            if found_geo:
                print(f"\n  ✓ GEOREFERENCING METADATA FOUND:")
                for k, v in found_geo.items():
                    print(f"    {k}: {v}")
            else:
                print(f"\n  ⚠ NO georeferencing metadata found")
                print(f"  Available attributes: {list(all_attrs.keys())}")
            
            # If tile provided, check for tile info
            if tile:
                tile_in_attrs = any(tile in str(v) for v in all_attrs.values())
                tile_in_name = tile.replace('_', '') in basename or tile in basename
                print(f"\n  Tile match check:")
                print(f"    In attributes: {tile_in_attrs}")
                print(f"    In filename: {tile_in_name}")
            
        except Exception as e:
            print(f"  ✗ Error reading: {e}")
            results[basename] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    geo_count = sum(1 for r in results.values() 
                   if isinstance(r, dict) and not r.get('error'))
    print(f"Successfully read: {geo_count}/{min(max_files, len(zarr_files))}")
    
    # Check for extent consistency across files
    shapes = [r.get('shape') for r in results.values() if r.get('shape')]
    if shapes:
        unique_shapes = set(shapes)
        print(f"Unique shapes: {unique_shapes}")
        if len(unique_shapes) > 1:
            print(f"  ⚠ WARNING: Different shapes found - files may be from different tiles!")
    
    return results


def compare_zarr_with_tile_extent(zarr_dir, tile, tile_extent):
    """Compare zarr file shapes with expected tile extent.
    
    Parameters
    ----------
    zarr_dir : str
        Directory containing zarr files
    tile : str
        Tile identifier
    tile_extent : tuple
        (left, bottom, right, top) in EPSG:3413
        
    Returns
    -------
    bool
        True if shapes are consistent with tile extent
    """
    import zarr
    
    zarr_files = glob.glob(str(Path(zarr_dir) / "processed_*.zarr"))
    if not zarr_files:
        zarr_files = glob.glob(str(Path(zarr_dir) / "*.zarr"))
    
    if not zarr_files:
        print("No zarr files found")
        return False
    
    # Read first file
    zf = zarr_files[0]
    store = zarr.open(zf, mode="r")
    
    if isinstance(store, zarr.Array):
        shape = store[:].shape
    else:
        first_key = list(store.keys())[0]
        shape = store[first_key][:].shape
    
    height, width = shape
    
    left, bottom, right, top = tile_extent
    expected_width = (right - left) / 2  # 2m resolution
    expected_height = (top - bottom) / 2
    
    print(f"\nTile extent check for {tile}:")
    print(f"  Zarr shape: {width} × {height} pixels")
    print(f"  Expected (at 2m): {expected_width:.0f} × {expected_height:.0f} pixels")
    print(f"  Match: width={abs(width - expected_width) < 10}, height={abs(height - expected_height) < 10}")
    
    return abs(width - expected_width) < 10 and abs(height - expected_height) < 10

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
# DATE EXTRACTION
# ============================================================================

def extract_date_from_strip(strip_name):
    """Extract date from strip name."""
    parts = strip_name.split("_")
    for part in parts:
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d")
            except ValueError:
                pass
    return None


def format_date_label(date_obj):
    """Format datetime object to readable label."""
    if date_obj:
        return date_obj.strftime("%b %Y")
    return "Unknown"



# =======================================================================
# INTERNAL HELPERS
# =======================================================================
def _secondary_axis_labels(left, right, bottom, top, n_ticks=4):
    """
    Compute EPSG:3413 tick positions and their correct EPSG:4326 labels for
    the secondary axes of an imshow plot whose extent is
    (left, right, bottom, top) in EPSG:3413.

    The top secondary x-axis shows the longitude where each *vertical*
    gridline crosses the TOP edge of the plot  → transform (x_tick, top).

    The right secondary y-axis shows the latitude where each *horizontal*
    gridline crosses the RIGHT edge of the plot → transform (x_right, y_tick).

    Parameters
    ----------
    left, right, bottom, top : float  – extent in EPSG:3413 metres
    n_ticks : int – number of interior ticks on each axis (default 4)

    Returns
    -------
    x_ticks  : 1-D array  – EPSG:3413 x positions for vertical gridlines
    y_ticks  : 1-D array  – EPSG:3413 y positions for horizontal gridlines
    lon_ticks : 1-D array – longitude (°E) labels for the top x-axis
    lat_ticks : 1-D array – latitude  (°N) labels for the right y-axis
    """
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)

    x_ticks = np.linspace(left,   right, n_ticks + 2)[1:-1]   # interior only
    y_ticks = np.linspace(bottom, top,   n_ticks + 2)[1:-1]

    # Top axis: each x gridline evaluated at y = top
    lon_ticks, _ = transformer.transform(x_ticks, np.full_like(x_ticks, top))

    # Right axis: each y gridline evaluated at x = right
    _, lat_ticks = transformer.transform(np.full_like(y_ticks, right), y_ticks)

    return x_ticks, y_ticks, lon_ticks, lat_ticks


def _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks):
    """Attach secondary top/right axes with lon/lat labels to *ax*."""
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


def overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=None):
    """
    Reproject and draw all features from *shp_path* onto *ax*.

    Parameters
    ----------
    ax        : matplotlib Axes
    shp_path  : str  – path to a shapefile (or any fiona-readable vector)
    raster_crs: str  – target CRS that matches the axes data coordinates
    style     : dict – passed as kwargs to PathCollection / LineCollection /
                       Polygon patch.  Defaults to a visible red outline.
    """
    if style is None:
        style = dict(facecolor="none", edgecolor="#42d116", linewidth=1.5, zorder=6)

    patches = []
    lines   = []

    try:
        with fiona.open(shp_path) as src:
            shp_crs = src.crs_wkt if hasattr(src, "crs_wkt") else src.crs.to_wkt()
            reproj  = Transformer.from_crs(shp_crs, raster_crs, always_xy=True)

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
                        codes  = (
                            [MplPath.MOVETO]
                            + [MplPath.LINETO] * (len(coords) - 2)
                            + [MplPath.CLOSEPOLY]
                        )
                        path   = MplPath(coords, codes)
                        patches.append(PathPatch(path))
                elif gtype in ("LineString", "MultiLineString"):
                    lines_list = (
                        [geom] if gtype == "LineString" else list(geom.geoms)
                    )
                    for line in lines_list:
                        xs, ys = line.xy
                        ax.plot(xs, ys, **{
                            k: v for k, v in style.items()
                            if k not in ("facecolor",)
                        })

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
            # Print shapefile coordinates
            shp_coords = pc.get_paths()[0].vertices if patches else None
            # print(f" Shapefile coordinates (EPSG:3413): {shp_coords}")

    except Exception as e:
        print(f"  ⚠ Could not overlay shapefile {shp_path}: {e}")


# ============================================================================
# FAST DIFFERENCE FROM ZARR FILES
# ============================================================================

def compute_dem_difference_from_zarr(strip_a, strip_b, zarr_dir):
    """Compute elevation difference from pre-computed zarr files (fast).
    
    Parameters
    ----------
    strip_a, strip_b : str
        Strip identifiers (can be partial)
    zarr_dir : str
        Directory containing zarr files
        
    Returns
    -------
    diff_masked, valid_mask, metadata or (None, None, None)
    """
    zarr_dir = Path(zarr_dir)
    
    # Find zarr files
    zarr_path_a = glob.glob(str(zarr_dir / f"processed_{strip_a[:30]}*.zarr"))
    zarr_path_b = glob.glob(str(zarr_dir / f"processed_{strip_b[:30]}*.zarr"))
    
    if not zarr_path_a or not zarr_path_b:
        if not zarr_path_a:
            print(f"  Could not find zarr file for {strip_a[:30]}")
        if not zarr_path_b:
            print(f"  Could not find zarr file for {strip_b[:30]}")
        return None, None, None
    
    try:
        store_a = zarr.open(zarr_path_a[0], mode="r")
        store_b = zarr.open(zarr_path_b[0], mode="r")
    except Exception as e:
        print(f"  Error opening zarr: {e}")
        return None, None, None
    
    # Extract arrays
    try:
        arr_a = store_a[:] if isinstance(store_a, zarr.Array) else store_a[list(store_a.keys())[0]][:]
        arr_b = store_b[:] if isinstance(store_b, zarr.Array) else store_b[list(store_b.keys())[0]][:]
    except Exception as e:
        print(f"  Error reading zarr: {e}")
        return None, None, None
    
    # Ensure same shape
    if arr_a.shape != arr_b.shape:
        min_rows = min(arr_a.shape[0], arr_b.shape[0])
        min_cols = min(arr_a.shape[1], arr_b.shape[1])
        arr_a = arr_a[:min_rows, :min_cols]
        arr_b = arr_b[:min_rows, :min_cols]
    
    # Create masks
    mask_a = (arr_a != -9999) & ~np.isnan(arr_a) & (arr_a > -500) & (arr_a < 5000)
    mask_b = (arr_b != -9999) & ~np.isnan(arr_b) & (arr_b > -500) & (arr_b < 5000)
    
    valid_mask = mask_a & mask_b
    diff = arr_a.astype(np.float32) - arr_b.astype(np.float32)
    diff_masked = np.ma.array(diff, mask=~valid_mask)
    
    # Zarr files don't have georeferencing - use pixel coordinates
    metadata = {
        'bounds': None,
        'transform': None,
        'crs': None,
        'shape': diff.shape,
        'source': 'zarr',
    }
    
    return diff_masked, valid_mask, metadata


# ============================================================================
# DOWNSAMPLED DIFFERENCE FROM GEOTIFFS
# ============================================================================

def find_coregistered_dem(strip_name, archive_dir, coreg_suffix=None):
    """Find a coregistered DEM file matching a strip name."""
    patterns = []
    
    if coreg_suffix:
        patterns.append(os.path.join(archive_dir, f"*{strip_name}*{coreg_suffix}*coregistered.tif"))
    
    patterns.append(os.path.join(archive_dir, f"*{strip_name}*coregistered.tif"))
    patterns.append(os.path.join(archive_dir, f"*{strip_name}*.tif"))
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Recursive search
    for pattern in patterns:
        recursive = os.path.join(archive_dir, "**", os.path.basename(pattern))
        matches = glob.glob(recursive, recursive=True)
        if matches:
            return matches[0]
    
    return None


def compute_dem_difference_from_geotiffs(strip_a, strip_b, archive_dir,
                                         coreg_suffix=None,
                                         downsample_factor=None,
                                         max_pixels=2000):
    """Compute elevation difference from GeoTIFFs with optional downsampling.
    
    Parameters
    ----------
    strip_a, strip_b : str
        Strip identifiers
    archive_dir : str
        Archive directory
    coreg_suffix : str, optional
        Suffix for coregistered files
    downsample_factor : int, optional
        Factor to downsample by (e.g., 10 = 20m resolution from 2m).
        If None, auto-calculates based on max_pixels.
    max_pixels : int
        Maximum dimension after downsampling (if downsample_factor not set)
        
    Returns
    -------
    diff_masked, valid_mask, metadata or (None, None, None)
    """
    dem_path_a = find_coregistered_dem(strip_a, archive_dir, coreg_suffix)
    dem_path_b = find_coregistered_dem(strip_b, archive_dir, coreg_suffix)
    
    if dem_path_a is None or dem_path_b is None:
        print(f"  Could not find DEM files")
        return None, None, None
    
    print(f"  A: {os.path.basename(dem_path_a)[:50]}...")
    print(f"  B: {os.path.basename(dem_path_b)[:50]}...")
    
    try:
        with rio.open(dem_path_a) as src_a, rio.open(dem_path_b) as src_b:
            
            # Calculate overlap
            if src_a.crs != src_b.crs:
                bounds_b = warp.transform_bounds(src_b.crs, src_a.crs,
                                                *src_b.bounds)
            else:
                bounds_b = src_b.bounds
            
            bounds_a = src_a.bounds
            
            intersect_left = max(bounds_a.left, bounds_b[0])
            intersect_bottom = max(bounds_a.bottom, bounds_b[1])
            intersect_right = min(bounds_a.right, bounds_b[2])
            intersect_top = min(bounds_a.top, bounds_b[3])
            
            if intersect_left >= intersect_right or intersect_bottom >= intersect_top:
                print("  ✗ No overlap")
                return None, None, None
            
            # Calculate native resolution and size
            native_res = src_a.res[0]
            overlap_width = (intersect_right - intersect_left) / native_res
            overlap_height = (intersect_top - intersect_bottom) / native_res
            
            # Determine downsampling
            if downsample_factor is None:
                downsample_factor = max(1, int(np.ceil(max(overlap_width, overlap_height) / max_pixels)))
            
            if downsample_factor > 1:
                target_res = native_res * downsample_factor
                print(f"  Downsampling {downsample_factor}x → {target_res:.0f}m resolution "
                      f"(native: {native_res:.0f}m, size: {overlap_width:.0f}×{overlap_height:.0f})")
                
                # Read with overview/out_shape for speed
                out_width = int(overlap_width / downsample_factor)
                out_height = int(overlap_height / downsample_factor)
                
                # Read window at full res, then downsample
                window_a = src_a.window(intersect_left, intersect_bottom,
                                       intersect_right, intersect_top)
                
                # Use out_shape for efficient reading
                arr_a = src_a.read(1, window=window_a, 
                                  out_shape=(out_height, out_width),
                                  resampling=RioResampling.bilinear)
                
                if src_a.crs != src_b.crs:
                    # Reproject B to A
                    window_b = src_b.window(*warp.transform_bounds(
                        src_b.crs, src_a.crs,
                        intersect_left, intersect_bottom,
                        intersect_right, intersect_top))
                    arr_b_full = src_b.read(1, window=window_b)
                    
                    dst_transform = from_bounds(intersect_left, intersect_bottom,
                                               intersect_right, intersect_top,
                                               out_width, out_height)
                    arr_b = np.zeros((out_height, out_width), dtype=np.float32)
                    
                    reproject(
                        source=arr_b_full,
                        destination=arr_b,
                        src_transform=src_b.window_transform(window_b),
                        src_crs=src_b.crs,
                        dst_transform=dst_transform,
                        dst_crs=src_a.crs,
                        resampling=RioResampling.bilinear,
                    )
                else:
                    window_b = src_b.window(intersect_left, intersect_bottom,
                                           intersect_right, intersect_top)
                    arr_b = src_b.read(1, window=window_b,
                                      out_shape=(out_height, out_width),
                                      resampling=RioResampling.bilinear)
                
                transform = src_a.window_transform(window_a)
                # Adjust transform for downsampling
                transform = rio.Affine(
                    transform.a * downsample_factor,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e * downsample_factor,
                    transform.f,
                )
                
                bounds = (intersect_left, intersect_bottom, intersect_right, intersect_top)
                
            else:
                # Full resolution
                print(f"  Full resolution: {overlap_width:.0f}×{overlap_height:.0f} pixels")
                
                window_a = src_a.window(intersect_left, intersect_bottom,
                                       intersect_right, intersect_top)
                arr_a = src_a.read(1, window=window_a)
                
                if src_a.crs != src_b.crs:
                    # Handle reprojection...
                    return None, None, None  # Skip full-res reprojection for now
                else:
                    window_b = src_b.window(intersect_left, intersect_bottom,
                                           intersect_right, intersect_top)
                    arr_b = src_b.read(1, window=window_b)
                
                transform = src_a.window_transform(window_a)
                bounds = (intersect_left, intersect_bottom, intersect_right, intersect_top)
            
            # Ensure same shape
            if arr_a.shape != arr_b.shape:
                min_h = min(arr_a.shape[0], arr_b.shape[0])
                min_w = min(arr_a.shape[1], arr_b.shape[1])
                arr_a = arr_a[:min_h, :min_w]
                arr_b = arr_b[:min_h, :min_w]
            
            # Get nodata values
            nodata_a = src_a.nodata if src_a.nodata is not None else -9999
            nodata_b = src_b.nodata if src_b.nodata is not None else -9999
            
            # Masks
            mask_a = (arr_a != nodata_a) & ~np.isnan(arr_a) & (arr_a > -500) & (arr_a < 5000)
            mask_b = (arr_b != nodata_b) & ~np.isnan(arr_b) & (arr_b > -500) & (arr_b < 5000)
            
            valid_mask = mask_a & mask_b
            diff = arr_a.astype(np.float32) - arr_b.astype(np.float32)
            diff_masked = np.ma.array(diff, mask=~valid_mask)
            
            metadata = {
                'bounds': bounds,
                'transform': transform,
                'crs': src_a.crs,
                'shape': diff.shape,
                'source': 'geotiff',
                'downsample_factor': downsample_factor,
            }
            
            return diff_masked, valid_mask, metadata
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


# ============================================================================
# UNIFIED PROCESSING FUNCTION
# ============================================================================

def process_strip_pairs(strip_pairs, 
                        archive_dir=None,
                        zarr_dir=None,
                        coreg_suffix=None,
                        output_dir=None,
                        prefer_zarr=True,
                        downsample_factor=10,
                        max_pixels=2000,
                        **plot_kwargs):
    """Process multiple strip pairs - tries zarr first, falls back to GeoTIFF.
    
    Parameters
    ----------
    strip_pairs : list of tuples
        List of (strip_a, strip_b) tuples (partial names OK)
    archive_dir : str, optional
        Directory containing GeoTIFF DEMs (for fallback)
    zarr_dir : str, optional
        Directory containing pre-computed zarr files (fast)
    coreg_suffix : str, optional
        Suffix for coregistered files
    output_dir : str, optional
        Output directory for plots and data
    prefer_zarr : bool
        Try zarr first if both sources available
    downsample_factor : int
        Downsampling factor for GeoTIFF processing (10 = 20m from 2m)
    max_pixels : int
        Maximum pixels if auto-downsampling
    **plot_kwargs
        Passed to plot_elevation_difference()
        
    Returns
    -------
    dict
        Results with statistics and plot paths
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    print(f"\n{'='*60}")
    print(f"DEM DIFFERENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Pairs: {len(strip_pairs)}")
    
    if zarr_dir:
        print(f"Zarr directory: {zarr_dir}")
    if archive_dir:
        print(f"Archive directory: {archive_dir}")
        print(f"Downsample factor: {downsample_factor}x")
    
    all_results = {}
    
    for strip_a, strip_b in strip_pairs:
        print(f"\n{'─'*60}")
        print(f"Pair: {strip_a[:40]}...")
        print(f"      {strip_b[:40]}...")
        
        diff = mask = metadata = None
        
        # Try zarr first (fast)
        if prefer_zarr and zarr_dir:
            print("  Trying zarr files...")
            diff, mask, metadata = compute_dem_difference_from_zarr(
                strip_a, strip_b, zarr_dir
            )
            if diff is not None:
                print(f"  ✓ Zarr successful ({diff.shape})")
        
        # Fall back to GeoTIFF
        if diff is None and archive_dir:
            print("  Falling back to GeoTIFFs...")
            diff, mask, metadata = compute_dem_difference_from_geotiffs(
                strip_a, strip_b, archive_dir, coreg_suffix,
                downsample_factor=downsample_factor,
                max_pixels=max_pixels,
            )
            if diff is not None:
                print(f"  ✓ GeoTIFF successful ({diff.shape})")
        
        if diff is None:
            print("  ✗ Failed to process pair")
            continue
        
        # Statistics
        valid_pixels = np.sum(mask)
        print(f"  Valid pixels: {valid_pixels:,} / {mask.size:,}")
        if valid_pixels == 0:
            print("  No valid pixels, skipping statistics and plot")
            continue
        else:
            print(f"  Mean diff: {np.mean(diff.compressed()):.2f} m")
            print(f"  Std diff: {np.std(diff.compressed()):.2f} m")
            print(f"  Min/Max: {np.min(diff.compressed()):.2f} / {np.max(diff.compressed()):.2f} m")
            
            # Plot
            png_path = plot_elevation_difference(
                diff, mask, strip_a, strip_b,
                output_dir=output_dir,
                metadata=metadata,
                **plot_kwargs
            )
            
        all_results[(strip_a, strip_b)] = {
            'diff': diff,
            'mask': mask,
            'metadata': metadata,
            'plot_path': png_path if png_path else None,
            'stats': {
                'valid_pixels': int(np.sum(mask)),
                'mean': float(np.mean(diff.compressed())) if valid_pixels > 0 else None,
                'std': float(np.std(diff.compressed())) if valid_pixels > 0 else None,
                'min': float(np.min(diff.compressed())) if valid_pixels > 0 else None,
                'max': float(np.max(diff.compressed())) if valid_pixels > 0 else None,
                'shape': diff.shape,
            }
        }
    
    print(f"\n{'='*60}")
    print(f"Complete: {len(all_results)}/{len(strip_pairs)} pairs processed")
    print(f"{'='*60}")
    
    return all_results


def plot_elevation_difference1(
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
    auto_crop=False,  # Changed default - show full extent
    crop_threshold=0.5,
    extent=None,
    shp_path=None,
    shp_style=None,
    metadata=None,
):
    """Plot elevation difference between two DEM strips.
    
    Shows the FULL tile extent with the difference data overlaid.
    Follows the same style as plot_final_raster for consistency.
    
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
        Colormap limits (auto if None)
    max_pixels : int
        Maximum array dimension after downsampling
    auto_crop : bool
        If True, crop to valid data. If False, show full extent.
    crop_threshold : float
        Fraction of total size below which cropping triggers
    extent : tuple or None
        (left, right, bottom, top) in EPSG:3413 meters.
        If None and metadata available, uses metadata bounds.
    shp_path : str, optional
        Path to shapefile for overlay
    shp_style : dict, optional
        Style for shapefile features
    metadata : dict, optional
        Georeferencing metadata from compute function
        
    Returns
    -------
    str
        Path to saved PNG file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Extract dates for title
    date_a = extract_date_from_strip(strip_a)
    date_b = extract_date_from_strip(strip_b)
    date_label_a = date_a.strftime("%d %b %Y")
    date_label_b = date_b.strftime("%d %b %Y")
    
    # Build titles - main title with dates as subtitle
    if tile:
        main_title = f"{cbar_title} — {tile}"
    else:
        main_title = cbar_title
    
    subtitle = f"{date_label_a}  −  {date_label_b}"
    
    interactive_mode = hasattr(sys, "ps1") or sys.flags.interactive
    
    # Get extent from metadata if not provided
    if extent is None and metadata is not None:
        bounds = metadata.get('bounds')
        print(f" Using metadata bounds from diff")
        if bounds:
            extent = (bounds[0], bounds[2], bounds[1], bounds[3])  # (left, right, bottom, top)
    else:
        print(f" Using provided extent: {extent}")

    # Find valid region
    valid_rows, valid_cols = np.where(valid_mask)
    has_valid_data = len(valid_rows) > 0
    
    if not has_valid_data:
        print(f"  No valid data for {strip_a} - {strip_b}")
        return None
    
    total_height, total_width = diff_masked.shape
    
    # Auto-crop if requested AND data is sparse
    should_crop = False
    if auto_crop and extent is not None:
        row_min, row_max = valid_rows.min(), valid_rows.max() + 1
        col_min, col_max = valid_cols.min(), valid_cols.max() + 1
        valid_height = row_max - row_min
        valid_width = col_max - col_min
        
        if (valid_height / total_height < crop_threshold or 
            valid_width / total_width < crop_threshold):
            should_crop = True
            print(f"  Auto-cropping: valid region {valid_width}×{valid_height} "
                  f"/ {total_width}×{total_height}")
    
    if should_crop:
        print("Should crop to valid region...")
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
            left, right, bottom, top = extent
            pixel_width = (right - left) / total_width
            pixel_height = (top - bottom) / total_height
            extent = (
                left + crop_col_min * pixel_width,
                left + crop_col_max * pixel_width,
                bottom + (total_height - crop_row_max) * pixel_height,
                bottom + (total_height - crop_row_min) * pixel_height,
            )
            print(f"new extent after cropping: {extent}")
    else:
        raster_data = diff_masked.copy()
        crop_valid_mask = valid_mask
    
    plot_height, plot_width = raster_data.shape
    
    # Downsample if needed
    downsample_factor = max(1, int(np.ceil(max(plot_height, plot_width) / max_pixels)))
    
    if downsample_factor > 1:
        h_new = plot_height // downsample_factor
        w_new = plot_width // downsample_factor
        
        # Downsample the data
        raster_ds = (
            raster_data[:h_new * downsample_factor, :w_new * downsample_factor]
            .reshape(h_new, downsample_factor, w_new, downsample_factor)
            .mean(axis=(1, 3))
        )
        
        # Also downsample the mask for valid pixel count
        mask_ds = (
            crop_valid_mask[:h_new * downsample_factor, :w_new * downsample_factor]
            .reshape(h_new, downsample_factor, w_new, downsample_factor)
            .mean(axis=(1, 3))
        )
        n_valid_approx = int(np.sum(crop_valid_mask) * (1.0 / downsample_factor**2))
        
        print(f"  Downsampled {downsample_factor}x: "
              f"{plot_width}×{plot_height} → {raster_ds.shape[1]}×{raster_ds.shape[0]}")
        print(f"Old extent: {extent}")
        # Adjust extent for downsampling
        if extent is not None:
            left, right, bottom, top = extent
            extent = (
                left, right,
                bottom + (top - bottom) * (1 - h_new / plot_height),
                top,
            )
            print(f"new extent after downsampling: {extent}")
    else:
        raster_ds = raster_data
        n_valid_approx = int(np.sum(crop_valid_mask))
    
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
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("black")
    
    interpolation = "bilinear" if downsample_factor > 1 else "nearest"
    
    if extent is not None:
        plot_extent = (left, right, bottom, top)
    else:
        plot_extent = (0, plot_width, 0, plot_height)
    
    img = ax.imshow(
        raster_ds,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=plot_extent,
        origin="upper",
        interpolation=interpolation,
        aspect="equal",
    )
    
    # ── AXIS LABELS (following plot_final_raster style) ──
    if extent is not None:
        left, right, bottom, top = extent
        
        # Get secondary axis tick positions and labels
        x_ticks, y_ticks, lon_ticks, lat_ticks = _secondary_axis_labels(
            left, right, bottom, top
        )
        
        # Primary axis (EPSG:3413)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("X (m) - EPSG 3413", fontsize=9)
        ax.set_ylabel("Y (m) - EPSG 3413", fontsize=9)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        
        # Secondary axes (EPSG:4326)
        _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks)
        
        # Shapefile overlay
        if shp_path is not None and os.path.exists(shp_path):
            overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=shp_style)
    else:
        ax.set_xlabel("Column (pixels)", fontsize=9)
        ax.set_ylabel("Row (pixels)", fontsize=9)
        ax.tick_params(labelsize=7)
    
    # ── TITLE (main title + subtitle below, no overlap) ──
    # Main title at the top
    ax.set_title(main_title, fontweight="bold", fontsize=12, pad=20)
    
    # Subtitle as a separate text element below the main title
    # Place it just below the top of the axes, after the title
    fig.canvas.draw()  # Needed to get correct renderer
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
        va="bottom",
        fontweight="normal",
    )
    
    # ── GRID ──
    if add_grid:
        ax.grid(
            visible=True, which="both",
            color="gray", linestyle="--", linewidth=0.4, alpha=0.6,
        )
    
    # ── COLORBAR ──
    cbar = fig.colorbar(
        img, ax=ax,
        orientation="vertical",
        pad=0.08,
        fraction=0.035,
        shrink=0.8,
    )
    cbar.set_label(cbar_title, labelpad=10, rotation=270, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    
    # ── STATISTICS BOX ──
    valid_data = raster_ds.compressed()
    
    if len(valid_data) > 0:
        stats_text = (
            f"Valid pixels: {n_valid_approx:,}\n"
            f"Mean: {np.mean(valid_data):.2f} m\n"
            f"Std:  {np.std(valid_data):.2f} m\n"
            f"Min:  {np.min(valid_data):.2f} m\n"
            f"Max:  {np.max(valid_data):.2f} m"
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    
    # ── LAYOUT ──
    plt.tight_layout()
    
    # ── SAVE ──
    image_dir = Path(output_dir) / "dem_differences" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename from dates and tile
    short_a = strip_a[:25].replace('/', '_').replace(' ', '_')
    short_b = strip_b[:25].replace('/', '_').replace(' ', '_')
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
    auto_crop=False,
    crop_threshold=0.5,
    extent=None,
    shp_path=None,
    shp_style=None,
    metadata=None,
):
    """Plot elevation difference between two DEM strips.
    
    Parameters
    ----------
    ...
    extent : tuple or None
        (left, bottom, right, top) in EPSG:3413 meters.
        If None, tries to determine from tile name or metadata.
    ...
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Extract dates for title
    date_a = extract_date_from_strip(strip_a)
    date_b = extract_date_from_strip(strip_b)
    
    if date_a:
        date_label_a = date_a.strftime("%d %b %Y")
    else:
        date_label_a = strip_a[:20]
    
    if date_b:
        date_label_b = date_b.strftime("%d %b %Y")
    else:
        date_label_b = strip_b[:20]
    
    # Build titles
    if tile:
        main_title = f"{cbar_title} — {tile}"
    else:
        main_title = cbar_title
    
    subtitle = f"{date_label_a}  −  {date_label_b}"
    
    interactive_mode = hasattr(sys, "ps1") or sys.flags.interactive
    
    # ── DETERMINE EXTENT ──
    # Priority: 1) passed extent, 2) tile lookup, 3) metadata, 4) pixel coords
    
    if extent is None:
        # Try to get from tile name
        if tile:
            full_bounds = get_tile_bounds(TILE)
            left, bottom, right, top = full_bounds
            EXTENT = (left, right, bottom, top)
            extent = EXTENT
            # extent = get_tile_extent(tile)
            if extent:
                print(f"  Using extent from tile '{tile}': {extent}")
        
        # Fall back to metadata
        if extent is None and metadata is not None:
            bounds = metadata.get('bounds')
            if bounds:
                # bounds from zarr are None; from GeoTIFF they're a BoundingBox
                if hasattr(bounds, 'left'):
                    extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)
                elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
                    extent = tuple(bounds)
                print(f"  Using extent from metadata: {extent}")
    
    if extent is None:
        print(f"  ⚠ No georeferencing available - using pixel coordinates")
    
    # Find valid region
    valid_rows, valid_cols = np.where(valid_mask)
    has_valid_data = len(valid_rows) > 0
    
    if not has_valid_data:
        print(f"  No valid data for {strip_a} - {strip_b}")
        return None
    
    total_height, total_width = diff_masked.shape
    
    # Prepare data (cropping/downsampling)
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
        n_valid_approx = int(np.sum(crop_valid_mask) * (1.0 / downsample_factor**2))
        
        print(f"  Downsampled {downsample_factor}x: "
              f"{plot_width}×{plot_height} → {raster_ds.shape[1]}×{raster_ds.shape[0]}")
    else:
        raster_ds = raster_data
        n_valid_approx = int(np.sum(crop_valid_mask))
    
    # Colormap limits
    if vmin is None or vmax is None:
        valid_data = raster_ds.compressed()
        if len(valid_data) > 0:
            abs_max = np.percentile(np.abs(valid_data), 99)
            vmin = -abs_max if vmin is None else vmin
            vmax = abs_max if vmax is None else vmax
    
    # Colormap
    import matplotlib.cm as mcm
    cmap_obj = mcm.get_cmap(cmap).copy()
    cmap_obj.set_bad("black")
    cmap_obj.set_under("black")
    
    # ── CREATE FIGURE ──
    if extent is not None:
        left, right, bottom, top = extent
        data_width = right - left
        data_height = top - bottom
        data_aspect = data_width / data_height
        
        fig_width = 8
        fig_height = fig_width / data_aspect
        fig_height = max(5, min(12, fig_height))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        plot_extent = (left, right, bottom, top)
    else:
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_extent = (0, plot_width, 0, plot_height)
    
    ax.set_facecolor("black")
    
    interpolation = "bilinear" if downsample_factor > 1 else "nearest"
    
    img = ax.imshow(
        raster_ds,
        cmap=cmap_obj,
        vmin=vmin,
        vmax=vmax,
        extent=plot_extent,
        origin="upper",
        interpolation=interpolation,
        aspect="equal",
    )
    
    # ── AXIS LABELS ──
    if extent is not None:
        left, right, bottom, top = extent
        
        x_ticks, y_ticks, lon_ticks, lat_ticks = _secondary_axis_labels(
            left, right, bottom, top
        )
        
        # Primary axis (EPSG:3413)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel("X (m) - EPSG 3413", fontsize=9)
        ax.set_ylabel("Y (m) - EPSG 3413", fontsize=9)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        
        # Secondary axes (EPSG:4326)
        _add_secondary_axes(ax, x_ticks, y_ticks, lon_ticks, lat_ticks)
        
        # Shapefile overlay
        if shp_path is not None and os.path.exists(shp_path):
            overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=shp_style)
    else:
        ax.set_xlabel("Column (pixels)", fontsize=9)
        ax.set_ylabel("Row (pixels)", fontsize=9)
        ax.tick_params(labelsize=7)
    
    # ── TITLE ──
    title = main_title + subtitle
    ax.set_title(title, fontweight="bold", fontsize=12, pad=15)
    
    # Subtitle below main title
    # ax.text(
    #     0.5, 1.05,
    #     subtitle,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     color="#555555",
    #     ha="center",
    #     va="bottom",
    #     fontweight="normal",
    # )
    
    # ── GRID ──
    if add_grid:
        ax.grid(
            visible=True, which="both",
            color="gray", linestyle="--", linewidth=0.4, alpha=0.6,
        )
    
    # ── COLORBAR ──
    cbar = fig.colorbar(
        img, ax=ax,
        orientation="vertical",
        pad=0.09,
        fraction=0.035,
        shrink=0.85,
    )
    cbar.set_label(cbar_title, labelpad=9, rotation=270, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    
    # ── STATISTICS BOX ──
    valid_data = raster_ds.compressed()
    
    if len(valid_data) > 0:
        stats_text = (
            f"Valid pixels: {n_valid_approx:,}\n"
            f"Mean: {np.mean(valid_data):.2f} m\n"
            f"Std:  {np.std(valid_data):.2f} m\n"
            f"Min:  {np.min(valid_data):.2f} m\n"
            f"Max:  {np.max(valid_data):.2f} m"
        )
        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    
    plt.tight_layout()
    
    # ── SAVE ──
    image_dir = Path(output_dir) / "dem_differences" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    short_a = strip_a[:25].replace('/', '_').replace(' ', '_')
    short_b = strip_b[:25].replace('/', '_').replace(' ', '_')
    tile_str = f"{tile}_" if tile else ""
    
    png_path = image_dir / f"{tile_str}diff_{short_a}_vs_{short_b}.png"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    print(f"  ✓ PNG saved: {png_path}")
    
    if interactive_mode:
        plt.show()
    else:
        plt.close(fig)
    
    del raster_data, raster_ds, img
    gc.collect()
    
    return str(png_path)

def plot_elevation_difference2(
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
    auto_crop=False,
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
    # Print parameters kwargs
    # print(f"  Plotting parameters:")
    # print(f"    strip_a: {strip_a}")
    # print(f"    strip_b: {strip_b}")
    # print(f"    tile: {tile}")
    # print(f"    cbar_title: {cbar_title}")
    # print(f"    cmap: {cmap}")
    # print(f"    vmin: {vmin}, vmax: {vmax}")
    # print(f"    max_pixels: {max_pixels}")
    # print(f"    auto_crop: {auto_crop}, crop_threshold: {crop_threshold}")
    # print(f"    extent: {extent}")
    # print(f"    shp_path: {shp_path}")

    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Extract dates
    date_a = extract_date_from_strip(strip_a)
    date_b = extract_date_from_strip(strip_b)
    # date_label_a = format_date_label(date_a)
    # date_label_b = format_date_label(date_b)
    date_label_a = date_a.strftime("%d %b %Y")
    date_label_b = date_b.strftime("%d %b %Y")

    # Build titles
    if tile:
        main_title = f"{cbar_title} — {tile}"
    else:
        main_title = f"{cbar_title}"
    
    subtitle = f"{date_label_a}  −  {date_label_b}"
    strip_note = f"{strip_a[:43]}   vs   {strip_b[:43]}"
    
    interactive_mode = hasattr(sys, "ps1") or sys.flags.interactive
    
    # Get extent from metadata if not provided
    if extent is None and metadata is not None:
        bounds = metadata.get('bounds')
        print(f"  Metadata bounds from difference")
        if bounds:
            extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    
    # Find valid region
    valid_rows, valid_cols = np.where(valid_mask)
    if len(valid_rows) == 0:
        print(f"  No valid data for {strip_a} - {strip_b}")
        return None
    total_height, total_width = diff_masked.shape
    
    # Auto-crop
    should_crop = False
    if auto_crop:
        row_min, row_max = valid_rows.min(), valid_rows.max() + 1
        col_min, col_max = valid_cols.min(), valid_cols.max() + 1
        
        valid_height = row_max - row_min
        valid_width = col_max - col_min
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
    
    # Leave some space on top and right for titles and colorbar
    plot_height, plot_width = raster_data.shape
    # plot_height += 100
    # plot_width += 100
    
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
            overlay_shapefile(ax, shp_path, raster_crs="EPSG:3413", style=shp_style)
    else:
        ax.set_xlabel("Column (pixels)", fontsize=9)
        ax.set_ylabel("Row (pixels)", fontsize=9)
        ax.tick_params(labelsize=7)
    
    # Titles
    ax.set_title(main_title, fontweight="bold", fontsize=12, pad=20)
    # ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, fontsize=9,
    #         color="#666666", ha="center", va="bottom", fontweight="normal")
    # ax.text(0.5, 1.015, strip_note, transform=ax.transAxes, fontsize=7,
    #         color="#999999", ha="center", va="bottom", 
    #         fontweight="normal", style="italic")
    #     # Subtitle as a separate text element below the main title
    # Place it just below the top of the axes, after the title
    fig.canvas.draw()  # Needed to get correct renderer
    ax.text(
        0.5, 1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="center",
        va="bottom",
        fontweight="normal",
    )

    # Grid
    if add_grid:
        ax.grid(visible=True, which="both", color="gray", 
               linestyle="--", linewidth=0.4, alpha=0.6)
    
    # Colorbar
    cbar = fig.colorbar(img, ax=ax, orientation="vertical", pad=0.08,
                       fraction=0.035, shrink=0.8)
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
    short_a = strip_a[:25].replace('/', '_').replace(' ', '_')
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


def get_tile_bounds(TILE):
    """Get EPSG:3413 bounds for a given tile identifier."""

    # Try to load mosaic DEM
    mosaicdir = '/home/moralpom/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/mosaic/'
    try:
        tile_parts = TILE.split("_")
        supertile = f"{tile_parts[0]}_{tile_parts[1]}"
        mosaic_path = os.path.join(
            mosaicdir, f"v4.1/2m/{supertile}/{TILE}_2m_v4.1_dem.tif"
        )

        if os.path.exists(mosaic_path):
            print(f"  Loading mosaic context: {mosaic_path}")

        with rio.open(mosaic_path) as src:
            bounds = src.bounds
        return bounds
    except Exception as e:
        print(f"  Could not load mosaic for tile {TILE}: {e}")
        return None


def quick_plot_zarr(zarr_dir, tile, strip_pairs, max_files=4, downsample_to=500):
    """Quickly plot zarr files to visually check their content and extent.
    
    Parameters
    ----------
    zarr_dir : str
        Directory containing zarr files
    tile : str
        Tile identifier for title
    max_files : int
        Number of files to plot
    downsample_to : int
        Target dimension for downsampling (smaller = faster)
    """
    import zarr
    import matplotlib.pyplot as plt
    
    zarr_dir = Path(zarr_dir)
    zarr_files = sorted(glob.glob(str(zarr_dir / "processed_*.zarr")))
    
    if not zarr_files:
        zarr_files = sorted(glob.glob(str(zarr_dir / "*.zarr")))
    
    if not zarr_files:
        print(f"No zarr files found in {zarr_dir}")
        return
    
    n_files = min(max_files, len(zarr_files))
    
    # Try to also load the mosaic tile for comparison
    mosaic_path = None
    mosaic_dirs = [
        f"/home/moralpom/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/mosaic/v4.1/2m/",
    ]
    for md in mosaic_dirs:
        # Look for the tile in mosaic
        pattern = os.path.join(md, f"*{tile[:5]}*", f"*{tile}*_dem.tif")
        matches = glob.glob(pattern)
        if matches:
            mosaic_path = matches[0]
            break
    
    has_mosaic = mosaic_path is not None and os.path.exists(mosaic_path)
    
    if has_mosaic:
        n_cols = 3
        fig, axes = plt.subplots(n_files, n_cols, figsize=(15, 5 * n_files))
        if n_files == 1:
            axes = axes.reshape(1, -1)
    else:
        n_cols = 2
        fig, axes = plt.subplots(n_files, n_cols, figsize=(10, 5 * n_files))
        if n_files == 1:
            axes = axes.reshape(1, -1)
    
    print(f"\n{'='*80}")
    print(f"QUICK ZARR VISUALIZATION - {tile}")
    print(f"{'='*80}")
    # Do it for the zarr files in strip_pairs only:    
    for pair_name in strip_pairs:
        zarr_files.append()    
    zarr_files = [zf for zf in zarr_files if any(pair_name in zf for pair_name in [f"{strip_a}_vs_{strip_b}" for strip_a, strip_b in strip_pairs])]
    print(f"Zarr files: {len(zarr_files)} total, showing {n_files}")
    if has_mosaic:
        print(f"Mosaic: {os.path.basename(mosaic_path)}")
    
    for i, zf in enumerate(zarr_files[:n_files]):
        basename = os.path.basename(zf)[:50]
        print(f"\n  {i+1}. {basename}...")
        
        try:
            store = zarr.open(zf, mode="r")
            
            if isinstance(store, zarr.Array):
                arr = store[:]
            else:
                first_key = list(store.keys())[0]
                arr = store[first_key][:]
            
            print(f"      Shape: {arr.shape}, dtype: {arr.dtype}")
            print(f"      Range: [{np.nanmin(arr[arr != -9999]):.1f}, {np.nanmax(arr[arr != -9999]):.1f}]")
            
            # Downsample heavily for quick plotting
            h, w = arr.shape
            ds_factor = max(1, int(np.ceil(max(h, w) / downsample_to)))
            
            if ds_factor > 1:
                h_new = h // ds_factor
                w_new = w // ds_factor
                arr_ds = arr[:h_new * ds_factor, :w_new * ds_factor].reshape(
                    h_new, ds_factor, w_new, ds_factor
                ).mean(axis=(1, 3))
                print(f"      Downsampled {ds_factor}x: {arr.shape} → {arr_ds.shape}")
            else:
                arr_ds = arr
            
            # Mask nodata
            arr_masked = np.ma.masked_where((arr_ds == -9999) | (arr_ds < -500) | (arr_ds > 5000), arr_ds)
            
            # Plot 1: Elevation (terrain colormap)
            ax1 = axes[i, 0]
            im1 = ax1.imshow(arr_masked, cmap='terrain', aspect='equal', 
                           interpolation='bilinear')
            ax1.set_title(f"Zarr: {basename[:40]}...\n({arr.shape[1]}×{arr.shape[0]} px)")
            ax1.set_xlabel("Column (pixels)")
            ax1.set_ylabel("Row (pixels)")
            plt.colorbar(im1, ax=ax1, label='Elevation (m)', fraction=0.046)
            
            # Plot 2: Valid data mask
            ax2 = axes[i, 1]
            valid_mask = (arr_ds != -9999) & (arr_ds > -500) & (arr_ds < 5000)
            ax2.imshow(valid_mask, cmap='gray', aspect='equal', interpolation='nearest')
            ax2.set_title(f"Valid data mask\n({np.sum(valid_mask):,} / {valid_mask.size:,} pixels)")
            ax2.set_xlabel("Column (pixels)")
            
            # Plot 3: Mosaic comparison (if available)
            if has_mosaic and i == 0:  # Only show mosaic once
                try:
                    import rasterio as rio
                    with rio.open(mosaic_path) as src:
                        # Read at similar resolution
                        mosaic_h, mosaic_w = src.height, src.width
                        mosaic_ds_factor = max(1, int(np.ceil(max(mosaic_h, mosaic_w) / downsample_to)))
                        
                        if mosaic_ds_factor > 1:
                            mosaic_data = src.read(1, 
                                out_shape=(mosaic_h // mosaic_ds_factor, mosaic_w // mosaic_ds_factor),
                                resampling=rio.enums.Resampling.average)
                        else:
                            mosaic_data = src.read(1)
                        
                        mosaic_masked = np.ma.masked_equal(mosaic_data, src.nodata if src.nodata else -9999)
                        
                        ax3 = axes[i, 2]
                        im3 = ax3.imshow(mosaic_masked, cmap='terrain', aspect='equal',
                                       interpolation='bilinear')
                        ax3.set_title(f"Mosaic tile\n({src.width}×{src.height} px)\n"
                                    f"EPSG:3413 bounds:\n"
                                    f"[{src.bounds.left:.0f}, {src.bounds.bottom:.0f},\n"
                                    f" {src.bounds.right:.0f}, {src.bounds.top:.0f}]")
                        ax3.set_xlabel("Column (pixels)")
                        plt.colorbar(im3, ax=ax3, label='Elevation (m)', fraction=0.046)
                        
                        print(f"\n  MOSAIC TILE INFO:")
                        print(f"      Shape: {src.width}×{src.height}")
                        print(f"      Resolution: {src.res}")
                        print(f"      Bounds: {src.bounds}")
                        print(f"      CRS: {src.crs}")
                        
                        # Compare shapes
                        print(f"\n  COMPARISON:")
                        print(f"      Zarr shape: {arr.shape[1]}×{arr.shape[0]}")
                        print(f"      Mosaic shape: {src.width}×{src.height}")
                        print(f"      Zarr/Mosaic ratio: {arr.shape[1]/src.width:.2f} × {arr.shape[0]/src.height:.2f}")
                        
                        if abs(arr.shape[1] - src.width) < 5 and abs(arr.shape[0] - src.height) < 5:
                            print(f"      ✓ Shapes match! Zarr covers the same extent as the mosaic tile")
                        else:
                            print(f"      ⚠ Shapes differ! Zarr may be a subset or different tile")
                            
                except Exception as e:
                    print(f"      Could not read mosaic: {e}")
            
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Also plot the first zarr and mosaic side by side if both exist
    if has_mosaic and n_files >= 1:
        print(f"\n{'='*80}")
        print(f"SIDE-BY-SIDE COMPARISON (first zarr vs mosaic)")
        print(f"{'='*80}")
        
        try:
            store = zarr.open(zarr_files[0], mode="r")
            if isinstance(store, zarr.Array):
                arr = store[:]
            else:
                arr = store[list(store.keys())[0]][:]
            
            with rio.open(mosaic_path) as src:
                mosaic_data = src.read(1)
            
            # Downsample both to same size for visual comparison
            target_size = 400
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Zarr
            h, w = arr.shape
            ds = max(1, int(np.ceil(max(h, w) / target_size)))
            arr_ds = arr[::ds, ::ds]
            arr_masked = np.ma.masked_where((arr_ds == -9999) | (arr_ds < -500), arr_ds)
            im1 = ax1.imshow(arr_masked, cmap='terrain', aspect='equal')
            ax1.set_title(f"Zarr ({arr.shape[1]}×{arr.shape[0]})")
            plt.colorbar(im1, ax=ax1, fraction=0.046)
            
            # Mosaic
            h, w = mosaic_data.shape
            ds = max(1, int(np.ceil(max(h, w) / target_size)))
            mosaic_ds = mosaic_data[::ds, ::ds]
            mosaic_masked = np.ma.masked_equal(mosaic_ds, src.nodata if src.nodata else -9999)
            im2 = ax2.imshow(mosaic_masked, cmap='terrain', aspect='equal')
            ax2.set_title(f"Mosaic ({mosaic_data.shape[1]}×{mosaic_data.shape[0]})\n"
                         f"Bounds: [{src.bounds.left:.0f}, {src.bounds.bottom:.0f}, "
                         f"{src.bounds.right:.0f}, {src.bounds.top:.0f}]")
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in side-by-side: {e}")
    
    return

# Alias for backward compatibility
run_dem_difference = process_strip_pairs