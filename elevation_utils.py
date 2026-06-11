"""
ArcticDEM Utility Functions
============================
Coordinate transformations, file I/O, elevation extraction, and STAC queries.
"""

import glob
import gzip
import os
import shutil
import tarfile
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import geopandas as gpd
import numpy as np
import pystac_client
import rasterio as rio
from pyproj import Transformer
from rasterio import warp

# Import configuration
from config import (
    STAC_API_URL, COLLECTION_ID, DEFAULT_CRS, GEOGRAPHIC_CRS,
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_TYPE, MAX_CLOUD_COVER
)


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def wgs84_to_3413(lon, lat):
    """Transform WGS84 (lon, lat) to EPSG:3413 (x, y).
    
    Parameters
    ----------
    lon, lat : float
        Coordinates in decimal degrees (WGS84)
        
    Returns
    -------
    tuple
        (x, y) in EPSG:3413 (meters)
    """
    points = warp.transform('EPSG:4326', 'EPSG:3413', [lon], [lat])
    return (points[0][0], points[1][0])


def is_probably_3413(x, y):
    """Check if coordinates appear to be in EPSG:3413 (meter ranges).
    
    Parameters
    ----------
    x, y : float
        Coordinates to check
        
    Returns
    -------
    bool
        True if coordinates look like EPSG:3413
    """
    # Simple heuristic: if values outside typical lon/lat ranges
    if abs(x) > 180 or abs(y) > 90:
        return True
    # Check typical Greenland EPSG:3413 ranges
    if -1000000 < x < 1000000 and -3500000 < y < 0:
        return True
    return False


def parse_coordinate_string(coord_str):
    """Parse a coordinate string in various formats.
    
    Supports: "lon,lat", "lon lat", "lon_lat"
    
    Parameters
    ----------
    coord_str : str
        Input string
        
    Returns
    -------
    tuple
        (lon/x, lat/y) as floats
    """
    coord_str = coord_str.strip().replace(' ', ',').replace('_', ',')
    parts = [float(p.strip()) for p in coord_str.split(',') if p.strip()]
    
    if len(parts) < 2:
        raise ValueError(f"Could not parse: '{coord_str}'. Need at least 2 values.")
    
    return (parts[0], parts[1])


def parse_transect_string(coord_str):
    """Parse string with start and end coordinates for a transect.
    
    Supports: "lon1,lat1,lon2,lat2" or with spaces/underscores
    
    Parameters
    ----------
    coord_str : str
        Input string with 4 values
        
    Returns
    -------
    tuple
        ((start_lon, start_lat), (end_lon, end_lat))
    """
    coord_str = coord_str.strip().replace(' ', ',').replace('_', ',')
    parts = [float(p.strip()) for p in coord_str.split(',') if p.strip()]
    
    if len(parts) < 4:
        raise ValueError(f"Need 4 values (lon1,lat1,lon2,lat2), got {len(parts)}")
    
    return ((parts[0], parts[1]), (parts[2], parts[3]))


def transform_bounds_to_wgs84(bounds, src_crs):
    """Transform raster bounds to WGS84.
    
    Parameters
    ----------
    bounds : rasterio BoundingBox
        Bounds in source CRS
    src_crs : str or CRS
        Source coordinate system
        
    Returns
    -------
    tuple
        (west, south, east, north) in WGS84
    """
    transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
    lons, lats = transformer.transform(
        [bounds.left, bounds.right],
        [bounds.bottom, bounds.top]
    )
    return (lons[0], lats[0], lons[1], lats[1])


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def find_dem_file(base_path):
    """Find a DEM .tif file corresponding to a base path.
    
    Parameters
    ----------
    base_path : str
        Base path or pattern
        
    Returns
    -------
    str or None
        Path to the DEM file
    """
    # Try direct .tif file
    if base_path.endswith('.tif') and os.path.exists(base_path):
        return base_path
    
    # Try patterns
    patterns = [base_path + '*_dem.tif']
    if base_path.endswith('.tar.gz'):
        patterns.append(base_path[:-7] + '*_dem.tif')
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            shortened = f".../{'/'.join(matches[0].split('/')[-2:])}"
            print(f"Found .tif file: {shortened}")
            return matches[0]
    
    return None


def extract_dem_from_archive(archive_path):
    """Extract DEM from compressed archive (.gz or .tar.gz).
    
    Parameters
    ----------
    archive_path : str
        Path to compressed archive
        
    Returns
    -------
    str or None
        Path to extracted DEM file
    """
    extracted_dir = os.path.dirname(archive_path)
    shortened = f".../{'/'.join(archive_path.split('/')[-2:])}"
    
    try:
        if archive_path.endswith('.tar.gz'):
            # Extract tarball
            tar_file = archive_path[:-3]  # Remove .gz
            with gzip.open(archive_path, 'rb') as f_gz:
                with open(tar_file, 'wb') as f_tar:
                    shutil.copyfileobj(f_gz, f_tar)
            
            with tarfile.open(tar_file, 'r') as tar:
                tar.extractall(path=extracted_dir)
            
            os.remove(tar_file)
            
            # Find extracted DEM
            base = archive_path[:-7]  # Remove .tar.gz
            matches = glob.glob(f"{base}*_dem.tif")
            return matches[0] if matches else None
            
        elif archive_path.endswith('.gz'):
            # Check if it's a direct gzipped TIFF
            with gzip.open(archive_path, 'rb') as f:
                magic = f.read(4)
            
            if magic[:2] in (b'II', b'MM'):  # TIFF signature
                tif_path = archive_path[:-3]
                with gzip.open(archive_path, 'rb') as f_in, open(tif_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                print(f"Extracted: {tif_path}")
                return tif_path
        
        print(f"Unsupported format: {archive_path}")
        return None
        
    except Exception as e:
        print(f"Error extracting {shortened}: {e}")
        return None


def find_and_unzip(base_path):
    """Find DEM file, extracting from archive if needed.
    
    Parameters
    ----------
    base_path : str
        Base path to search
        
    Returns
    -------
    str or None
        Path to usable DEM file
    """
    # Try existing .tif first
    tif_path = find_dem_file(base_path)
    if tif_path:
        return tif_path
    
    # Look for archives
    patterns = [base_path + '*.gz']
    if base_path.endswith('.tar.gz'):
        patterns.append(base_path[:-7] + '*.gz')
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return extract_dem_from_archive(matches[0])
    
    print(f"No DEM file found for: {base_path}")
    return None


# ============================================================================
# ELEVATION EXTRACTION
# ============================================================================

def get_elevation_window(src, x, y, window_size=DEFAULT_WINDOW_SIZE, 
                         window_type=DEFAULT_WINDOW_TYPE):
    """Extract elevation statistics from a window around a point.
    
    Parameters
    ----------
    src : rasterio DatasetReader
        Opened raster
    x, y : float
        Coordinates in raster CRS
    window_size : int
        Window size (3=3x3, 5=5x5, etc.)
    window_type : str
        'square' for full window, 'cross' for plus-sign pattern
        
    Returns
    -------
    tuple
        (mean_elevation, std_elevation, valid_pixel_count)
    """
    row, col = src.index(x, y)
    
    if not (0 <= row < src.height and 0 <= col < src.width):
        return np.nan, np.nan, 0
    
    # Calculate window
    half = window_size // 2
    r_start = max(0, row - half)
    r_end = min(src.height, row + half + 1)
    c_start = max(0, col - half)
    c_end = min(src.width, col + half + 1)
    
    # Read window
    window_data = src.read(1, window=((r_start, r_end), (c_start, c_end)))
    
    # Apply cross pattern if requested
    if window_type == 'cross':
        cr = row - r_start
        cc = col - c_start
        mask = np.ones_like(window_data, dtype=bool)
        mask[cr, :] = False   # Horizontal
        mask[:, cc] = False   # Vertical
        mask[cr, cc] = False  # Center
        window_data = np.where(mask, np.nan, window_data)
    
    # Mask nodata and outliers
    if src.nodata is not None:
        window_data = np.where(window_data == src.nodata, np.nan, window_data)
    window_data = np.where((window_data > 5000) | (window_data < -500), np.nan, window_data)
    
    valid = window_data[~np.isnan(window_data)]
    if len(valid) == 0:
        return np.nan, np.nan, 0
    
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


def extract_elevation_profile(coords_4326, raster_path, num_samples=100):
    """Extract elevation profile between two points.
    
    Parameters
    ----------
    coords_4326 : tuple
        ((start_lon, start_lat), (end_lon, end_lat)) in WGS84
    raster_path : str
        Path to DEM file
    num_samples : int
        Number of elevation samples
        
    Returns
    -------
    tuple
        (transect_distances, elevations, total_distance, x0, y0, x1, y1)
    """
    # Transform to EPSG:3413
    origin = wgs84_to_3413(*coords_4326[0])
    end = wgs84_to_3413(*coords_4326[1])
    x0, y0 = origin
    x1, y1 = end
    
    # Handle compressed files
    if not raster_path.endswith('.tif') or not os.path.exists(raster_path):
        raster_path = find_and_unzip(raster_path)
    
    with rio.open(raster_path) as src:
        # Calculate distances
        distance = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        transect = np.linspace(0, distance, num_samples)
        
        # Sample points
        x_coords = np.linspace(x0, x1, num_samples)
        y_coords = np.linspace(y0, y1, num_samples)
        
        elevations = np.full(num_samples, np.nan)
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                if val != src.nodata:
                    elevations[i] = val
        
        return transect, elevations, distance, x0, y0, x1, y1


def read_elevation_from_compressed(raster_path, x, y, window_size=3, window_type='square'):
    """Read elevation from compressed file without full extraction.
    
    Parameters
    ----------
    raster_path : str
        Path to raster or archive
    x, y : float
        Coordinates in EPSG:3413
    window_size, window_type : as in get_elevation_window
        
    Returns
    -------
    tuple
        (mean_elevation, std_elevation, valid_pixel_count)
    """
    # Find the archive file
    gz_files = glob.glob(raster_path[:-8] + "*.gz")
    if not gz_files:
        gz_files = glob.glob(raster_path[:-18] + "*.gz")
        if not gz_files:
            # Try direct .tif
            tif_files = glob.glob(raster_path[:-18] + "*_dem.tif")
            if tif_files:
                with rio.open(tif_files[0]) as src:
                    return get_elevation_window(src, x, y, window_size, window_type)
            return np.nan, np.nan, 0
    
    archive = gz_files[0]
    
    try:
        if archive.endswith('.tar.gz'):
            with tarfile.open(archive, 'r:gz') as tar:
                dem_member = next((m for m in tar.getmembers() 
                                  if m.name.endswith('_dem.tif')), None)
                if not dem_member:
                    return np.nan, np.nan, 0
                
                with tar.extractfile(dem_member) as f:
                    with rio.MemoryFile(f.read()) as memfile:
                        with memfile.open() as src:
                            return get_elevation_window(src, x, y, window_size, window_type)
        else:
            with rio.open(archive) as src:
                return get_elevation_window(src, x, y, window_size, window_type)
    except Exception as e:
        print(f"Error reading compressed file: {e}")
        return np.nan, np.nan, 0


# ============================================================================
# STAC API QUERIES
# ============================================================================

def create_stac_client():
    """Create and return a STAC API client."""
    return pystac_client.Client.open(STAC_API_URL)


def search_arcticdem_strips(bbox, time_range, max_items=None):
    """Search for ArcticDEM strip data.
    
    Parameters
    ----------
    bbox : tuple
        (west, south, east, north) in WGS84
    time_range : str
        "YYYY-MM-DD/YYYY-MM-DD"
    max_items : int, optional
        Maximum items to return
        
    Returns
    -------
    tuple
        (GeoDataFrame, list of raw items)
    """
    client = create_stac_client()
    
    params = {
        'collections': [COLLECTION_ID],
        'bbox': bbox,
        'datetime': time_range,
    }
    if max_items:
        params['limit'] = max_items
    
    search = client.search(**params)
    items = list(search.items())
    
    if not items:
        print("No DEMs found for the specified region and time range.")
        return gpd.GeoDataFrame(), []
    
    try:
        items_gdf = gpd.GeoDataFrame.from_features(
            search.item_collection().to_dict(),
            crs="epsg:4326"
        )
    except ValueError as e:
        raise ValueError(f"Error converting results: {e}")
    
    print(f"Found {len(items)} StripDEMs")
    return items_gdf, items


def filter_strip_dems(items_gdf, max_cloud_cover=MAX_CLOUD_COVER, 
                      exclude_xtrack=True, max_items=None):
    """Filter STAC results by quality criteria.
    
    Parameters
    ----------
    items_gdf : GeoDataFrame
        Search results
    max_cloud_cover : float
        Maximum cloud fraction (0-1)
    exclude_xtrack : bool
        Exclude cross-track DEMs
    max_items : int, optional
        Downsample to this many items
        
    Returns
    -------
    GeoDataFrame
        Filtered results
    """
    filtered = items_gdf.copy()
    
    # Cloud cover filter
    if 'pgc:cloud_area_percent' in filtered.columns:
        filtered = filtered[filtered['pgc:cloud_area_percent'] < max_cloud_cover]
        print(f"After cloud filter (<{max_cloud_cover:.0%}): {len(filtered)} DEMs")
    
    # Exclude cross-track
    if exclude_xtrack and 'pgc:is_xtrack' in filtered.columns:
        filtered = filtered[filtered['pgc:is_xtrack'] == False]
        print(f"After xtrack filter: {len(filtered)} DEMs")
    
    # Downsample
    if max_items and len(filtered) > max_items:
        indices = np.linspace(0, len(filtered) - 1, max_items, dtype=int)
        filtered = filtered.iloc[indices]
        print(f"Downsampled to {len(filtered)} DEMs")
    
    return filtered


def get_dem_metadata(items_gdf):
    """Extract pairnames, geocells, and dates from STAC results.
    
    Parameters
    ----------
    items_gdf : GeoDataFrame
        
    Returns
    -------
    tuple
        (pairnames_list, geocells_list, dates_list)
    """
    pairnames = items_gdf['pgc:pairname'].tolist() if 'pgc:pairname' in items_gdf.columns else []
    geocells = items_gdf['pgc:geocell'].tolist() if 'pgc:geocell' in items_gdf.columns else []
    dates = items_gdf['datetime'].tolist() if 'datetime' in items_gdf.columns else []
    return pairnames, geocells, dates