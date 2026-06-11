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


def find_and_unzip(pathfile):
    """
    Finds a .tif file and/or unzips the .gz file if necessary.

    Args:
        pathfile (str): Path to the file without extension.

    Returns:
        str: Path to the .tif file if found or extracted, otherwise None.

    Exceptions:
        FileNotFoundError: If the .gz file or .tif file is not found.
        gzip.BadGzipFile: If the .gz file is not a valid gzip file.
        tarfile.TarError: If there is an error extracting the tar file.
        Exception: For any other exceptions, returns None and prints the error message.
    """
    if pathfile.endswith(".tar.gz"):
        tif_file_pattern = pathfile[:-7] + "*_dem.tif"
    elif pathfile.endswith("_dem.tif"):
        tif_file_pattern = pathfile
    else:
        tif_file_pattern = pathfile[:-20] + "*_dem.tif"
    # tif_file_pattern = pathfile + "*_dem.tif"
    tif_file_list = glob.glob(tif_file_pattern)
    if tif_file_list:
        shortened_path = (
            ".../"
            + tif_file_list[0].split("/")[-2]
            + "/"
            + tif_file_list[0].split("/")[-1]
        )
        print(f"Found existing .tif file: {shortened_path}")
        return tif_file_list[0]

    gz_file_list = glob.glob(pathfile[:-25] + "*.gz")
    if not gz_file_list:
        print(f"⚠ No .gz found for {pathfile}. Skipping.")
        return None

    gz_file = gz_file_list[0]
    extracted_dir = os.path.dirname(gz_file)

    shortened_path_gz = ".../" + gz_file.split("/")[-2] + "/" + gz_file.split("/")[-1]
    print(f"Unzipping .gz file: {shortened_path_gz}")

    try:
        # Check if .gz contains a .tif directly
        with gzip.open(gz_file, "rb") as f_in:
            magic = f_in.read(4)  # Read first bytes to check type

        if magic.startswith(b"II") or magic.startswith(b"MM"):  # TIFF signature
            # Extract directly to a .tif file
            tif_file = gz_file[:-3]  # Remove .gz extension
            with gzip.open(gz_file, "rb") as f_in, open(tif_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Extracted TIFF: {tif_file}")
            return tif_file

        # Otherwise, assume .tar file inside
        tar_file = gz_file[:-3]  # Remove .gz

        with gzip.open(gz_file, "rb") as f_gz:
            print("Opening gzip file")
            with open(tar_file, "wb") as f_tar:
                print("Copying gzip content to tar file")
                shutil.copyfileobj(f_gz, f_tar)

        with tarfile.open(tar_file, "r") as tar:
            print("Opening tar file and extracting...")
            tar.extractall(path=extracted_dir)
            # print(f"Extracted contents of .../{tar_file.split('/')[-2]}/{tar_file.split('/')[-1]}")

        os.remove(tar_file)

        # Refresh search for .tif file
        tif_file_list = glob.glob(f"{pathfile[:-25]}*_dem.tif")
        if tif_file_list and len(tif_file_list) < 1:
            print(f"⚠ No .tif files found: {tif_file_list}. Skipping...")
        return tif_file_list[0] if tif_file_list else None

    except FileNotFoundError as e:
        print(f"⚠ File not found: {e}")
        return None
    except gzip.BadGzipFile as e:
        print(f"⚠ Bad GZIP file: {e}")
        return None
    except tarfile.TarError as e:
        print(f"⚠ Error extracting TAR file: {e}")
        return None
    except Exception as e:
        print(f"⚠ Unexpected error: {e}\n for file {gz_file}")
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


def transform_to_3413(point_wgs84):
    """
    Transform WGS84 (lon,lat) coordinates to EPSG:3413.
    Parameters:
    ----------
    point_wgs84 : tuple
        Tuple containing two tuples: (start_point, end_point) in WGS84 (lon, lat)

    Returns:
    -------
    tuple
        Transformed coordinates in EPSG:3413
    """
    # Convert to EPSG:3413
    point = warp.transform(
        # src_crs='EPSG:3857',
        src_crs="EPSG:4326",
        dst_crs="EPSG:3413",
        xs=[point_wgs84[0]],
        ys=[point_wgs84[1]],
    )

    # Format as tuples (x,y)
    return (point[0][0], point[1][0])


def transect_from_mosaic(coords_4326, num_samples=100):
    """Get elevation profile from a mosaic using provided origin and end coordinates.

    Parameters:
    -----------
    coords_4326 : tuple
        Tuple containing two tuples: (start_point, end_point) in EPSG:4326 (lon, lat)
    num_samples : int, optional
        Number of points to sample along the profile (default: 100)

    Returns:
    --------
    dict
        Dictionary containing profile data and metadata
    """
    print(f"Finding mosaic file for coordinates {coords_4326}")
    lon_A, lat_A = coords_4326[0]
    lon_B, lat_B = coords_4326[1]

    # Search for the mosaic index
    mosaicdir = "/home/moralpom/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/mosaic/"
    mosaicindexdir = "/home/moralpom/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM_Mosaic_Index_latest_shp/"
    mosaic_index = mosaicindexdir + "ArcticDEM_Mosaic_Index_v4_1_2m.shp"

    # Read mosaic_index shapefile
    mosaic_gdf = gpd.read_file(mosaic_index)

    # Find the mosaic file that contains the point
    point_A = gpd.GeoSeries([gpd.points_from_xy([lon_A], [lat_A])[0]], crs="EPSG:4326")
    point_B = gpd.GeoSeries([gpd.points_from_xy([lon_B], [lat_B])[0]], crs="EPSG:4326")
    point_A = point_A.to_crs("EPSG:3413")
    point_B = point_B.to_crs("EPSG:3413")

    # Search GeoDataFrame for the polygon containing the point
    containing_polygons_A = mosaic_gdf[mosaic_gdf.geometry.contains(point_A.iloc[0])]
    # if containing_polygons_A.empty:
    #     raise ValueError(f"No mosaic found containing point_A {coords[0]}")
    containing_polygons_B = mosaic_gdf[mosaic_gdf.geometry.contains(point_B.iloc[0])]
    # if containing_polygons_B.empty:
    #     raise ValueError(f"No mosaic found containing point_B {coords[1]}")

    if containing_polygons_A.empty and containing_polygons_B.empty:
        raise ValueError(f"No mosaic found containing either point_A {coords_4326[0]} or point_B {coords_4326[1]}")
    if containing_polygons_A.empty:
        containing_polygons_A = containing_polygons_B
    if containing_polygons_B.empty:
        containing_polygons_B = containing_polygons_A

    # Find the mosaic tile
    tile_A = containing_polygons_A.iloc[0]["tile"]
    tile_B = containing_polygons_B.iloc[0]["tile"]
    if tile_A == tile_B:
        print(f"✓ Both points are in the same mosaic tile: {tile_A}")
        supertile = containing_polygons_A.iloc[0]["supertile"]
        tile_id = tile_A + "_2m_v4.1"

        # Construct the expected mosaic file path
        mosaic_dem = f"{mosaicdir}v4.1/2m/{supertile}/{tile_id}_dem.tif"
        if not mosaic_dem:
            raise FileNotFoundError(f"No mosaic file found for coordinates {coords_4326}.")

        rasterpath = mosaic_dem
        print(f"Using mosaic file: {rasterpath}")

        return extract_elevation_profile_compressed(coords_4326, rasterpath, num_samples)

    # This is a mess that will not work by now because the transect crosses two tiles
    else:
        print(
            f"⚠ Warning: Start and end points are in different mosaic tiles ({tile_A} & {tile_B}). Not working yet."
        )
        return None, None, None, None, None, None, None  # by now

        supertile_A = containing_polygons_A.iloc[0]["supertile"]
        tile_id_A = tile_A + "_2m_v4.1"
        mosaic_dem_A = f"{mosaicdir}v4.1/2m/{supertile_A}/{tile_id_A}_dem.tif"
        if not mosaic_dem_A:
            raise FileNotFoundError(f"No mosaic file found for coordinates {coords_4326}.")
        rasterpath_A = mosaic_dem_A

        supertile_B = containing_polygons_B.iloc[0]["supertile"]
        tile_id_B = tile_B + "_2m_v4.1"
        mosaic_dem_B = f"{mosaicdir}v4.1/2m/{supertile_B}/{tile_id_B}_dem.tif"
        if not mosaic_dem_B:
            raise FileNotFoundError(f"No mosaic file found for coordinates {coords_4326}.")
        rasterpath_B = mosaic_dem_B
        print(f"Using mosaic files: {rasterpath_A} and {rasterpath_B}")
        return extract_elevation_profile_compressed(coords_4326, rasterpath_B, num_samples)


def extract_elevation_profile_compressed(
    coords_4326, rasterpath, num_samples=100
):
    """Process elevation profile using provided origin and end coordinates from compressed files.

    Parameters:
    -----------
    coords_4326 : tuple
        Tuple containing two tuples: (start_point, end_point) in EPSG:4326 (lon, lat)
    rasterpath : str
        Path to the raster file or compressed archive
    num_samples : int, optional
        Number of points to sample along the profile (default: 100)

    Returns:
    --------
    dict
        Dictionary containing profile data and metadata
    """
    # print(f'Transforming points from EPSG:4326 to EPSG:3413.')
    origin_coords = transform_to_3413(coords_4326[0])
    end_coords = transform_to_3413(coords_4326[1])

    if origin_coords is None or end_coords is None:
        raise ValueError("Both start and end coordinates must be provided")

    # Find the compressed file
    gzfile = glob.glob(rasterpath[:-8] + "*.gz")
    if not gzfile:
        gzfile = glob.glob(rasterpath[:-18] + "*.gz")
        if not gzfile:
            gzfile = glob.glob(rasterpath[:-18] + "*_dem.tif")
            if not gzfile:
                raise ValueError("No suitable raster file found")

    def read_profile_from_memfile(src, x0, y0, x1, y1, num_samples):
        """Read elevation profile from opened raster."""
        # Validate coordinates against raster bounds
        bounds = src.bounds
        if not (
            bounds.left <= x0 <= bounds.right and bounds.bottom <= y0 <= bounds.top
        ):
            raise ValueError(f"Origin coordinates ({x0}, {y0}) outside bounds {bounds}")
        if not (
            bounds.left <= x1 <= bounds.right and bounds.bottom <= y1 <= bounds.top
        ):
            raise ValueError(f"End coordinates ({x1}, {y1}) outside bounds {bounds}")

        # Calculate the actual distance of the transect
        transect_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        transect = np.linspace(0, transect_distance, num_samples)

        # Generate world coordinates along the profile line
        # print(f"Generating {num_samples} points along profile line")
        x_coords = np.linspace(x0, x1, num_samples)
        y_coords = np.linspace(y0, y1, num_samples)

        # Convert world coordinates to pixel indices
        # rows, cols = src.index(x_coords, y_coords)  # Old way
        rows = []
        cols = []
        for i in range(num_samples):
            x_coord = x_coords[i]
            y_coord = y_coords[i]
            row, col = src.index(x_coord, y_coord)
            rows.append(row)
            cols.append(col)

        # Extract elevation values using bilinear interpolation
        profile_values = []
        for row, col in zip(rows, cols):
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                profile_values.append(np.nan if val == src.nodata else val)
            else:
                profile_values.append(np.nan)

        profile_values = np.array(profile_values)
        print(
            f"Elevation range: {np.nanmin(profile_values):.1f} to {np.nanmax(profile_values):.1f} meters"
        )

        return transect, profile_values, transect_distance, x0, y0, x1, y1

    if gzfile[0].endswith(".tar.gz"):
        with tarfile.open(gzfile[0], "r:gz") as tar:
            # Find the DEM file in the archive
            dem_member = next(
                (m for m in tar.getmembers() if m.name.endswith("_dem.tif")), None
            )
            if not dem_member:
                raise ValueError("No DEM file found in archive")

            # Extract just the needed portion to memory
            with tar.extractfile(dem_member) as f:
                with rio.MemoryFile(f.read()) as memfile:
                    with memfile.open() as src:
                        return read_profile_from_memfile(
                            src, *origin_coords, *end_coords, num_samples
                        )
    elif gzfile[0].endswith(".tif"):
        with rio.open(gzfile[0]) as src:
            return read_profile_from_memfile(
                src, *origin_coords, *end_coords, num_samples
            )
    else:
        raise ValueError("Unsupported file format")
    

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