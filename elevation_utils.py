"""
ArcticDEM Utility Functions
============================
Coordinate transformations, file I/O, elevation extraction, and STAC queries.
"""

import glob
import os
import re
import tarfile
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
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
    DEFAULT_WINDOW_SIZE, DEFAULT_WINDOW_TYPE, MAX_CLOUD_COVER, COREG_PARAMS
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
# STRIP FILE LOOKUP
# ============================================================================

# STAC item id / strip name: [SETSM_s2s041_]<pairname>_2m_[lsf_]seg<N>
_STRIP_ID_RX = re.compile(r"^(?:SETSM_s2s041_)?(?P<pair>.+?)_2m_(?:lsf_)?(?P<seg>seg\d+)$")


def coreg_suffix(coreg_mode):
    """Filename suffix between '_dem' and '_coregistered.tif' for a coreg mode.

    Built from COREG_PARAMS, e.g. for 'altim':
    "_cs2_v_999-0_dh_999-0000_['vertical_offset_mean', 'nuthkaab', 'deramp']"
    """
    params = COREG_PARAMS[coreg_mode]
    recipe = str(params['coreg_choice'].split())
    return (f"_{params['reference_data']}_v_{params['filter_vel']}-0"
            f"_dh_{params['filter_dhdt']}-0000_{recipe}")


@lru_cache(maxsize=512)
def _list_geocell(archdir, geocell):
    """Cached directory listing of one archive geocell."""
    try:
        return tuple(os.listdir(os.path.join(archdir, geocell)))
    except OSError:
        return ()


def find_strip_file(archdir, geocell, strip, coreg_mode='none'):
    """Locate the file to read for one strip.

    Parameters
    ----------
    archdir : str
        Strip archive root (contains <geocell>/ directories)
    geocell : str
        e.g. 'n75w055'
    strip : str
        Pairname ('WV01_20150701_...') or STAC item id
        ('SETSM_s2s041_WV01_20150701_..._2m_lsf_seg1'). An item id pins the
        segment; a bare pairname takes the lowest segment available.
    coreg_mode : str
        'none'  -> the strip '.tar.gz' (read in memory by open_dem), falling
                   back to an unpacked '_dem.tif'
        'altim' / 'mosaic' -> the coregistered GeoTIFF whose suffix exactly
                   matches coreg_suffix(coreg_mode). The doubled-prefix
                   variant ('_cs2_v_cs2_v_...', an upstream naming bug) is
                   accepted when the correct name is absent.
        'lsf' is optional in every name.

    Returns
    -------
    str or None
        Path to the file, or None if nothing matches
    """
    m = _STRIP_ID_RX.match(strip)
    pair, seg = (m['pair'], m['seg']) if m else (strip, None)
    seg_rx = re.escape(seg) if seg else r"seg\d+"
    head = rf"SETSM_s2s041_{re.escape(pair)}_2m_(?:lsf_)?(?P<seg>{seg_rx})"

    if coreg_mode == 'none':
        tails = [r"\.tar\.gz", r"_dem\.tif"]
    else:
        suffix = coreg_suffix(coreg_mode)
        doubled = f"_{COREG_PARAMS[coreg_mode]['reference_data']}_v{suffix}"
        tails = [re.escape(f"_dem{suffix}_coregistered.tif"),
                 re.escape(f"_dem{doubled}_coregistered.tif")]

    names = _list_geocell(archdir, geocell)
    for tail in tails:
        rx = re.compile(head + tail + "$")
        hits = sorted(((int(m['seg'][3:]), n) for n in names if (m := rx.match(n))))
        if hits:
            if seg is None and len(hits) > 1:
                print(f"  ⚠ {pair}: {len(hits)} segments available, using seg{hits[0][0]} "
                      f"(pass STAC item ids to pick the right one)")
            return os.path.join(archdir, geocell, hits[0][1])
    return None


@contextmanager
def open_dem(path):
    """Open a DEM for reading, as a context manager yielding a rasterio dataset.

    GeoTIFFs are opened directly. For a strip '.tar.gz' the '_dem.tif' member
    is streamed into memory; nothing is written to disk.
    """
    if path.endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                if member.name.endswith('_dem.tif'):
                    data = tar.extractfile(member).read()
                    break
            else:
                raise ValueError(f"No _dem.tif inside {path}")
        with rio.MemoryFile(data) as memfile, memfile.open() as src:
            yield src
    else:
        with rio.open(path) as src:
            yield src


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
        print("Row/col out of indexes")
        return np.nan, np.nan, 0
    
    # Calculate window
    half = window_size // 2
    r_start = max(0, row - half)
    r_end = min(src.height, row + half + 1)
    c_start = max(0, col - half)
    c_end = min(src.width, col + half + 1)
    
    # Read window
    window_data = src.read(1, window=((r_start, r_end), (c_start, c_end)))
    # print(f"window data (pre-filter): {window_data}")
    
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
    # print(f"window data (post-filter): {window_data}")

    valid = window_data[~np.isnan(window_data)]
    # print(f"valid (post-post-filter): {valid}")

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
        Path to DEM GeoTIFF or strip .tar.gz (read in memory)
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
    
    with open_dem(raster_path) as src:
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
    item_collection = search.item_collection()
    items = list(item_collection)
    
    if not items:
        print("No DEMs found for the specified region and time range.")
        return gpd.GeoDataFrame(), []
    
    try:
        items_gdf = gpd.GeoDataFrame.from_features(
            item_collection.to_dict(),
            crs="epsg:4326"
        )
    except ValueError as e:
        raise ValueError(f"Error converting results: {e}")
    # Item ids carry the segment (…_2m_lsf_seg1), which pairnames don't
    items_gdf['stac_id'] = [item.id for item in items]
    
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


def get_strip_ids(items_gdf):
    """STAC item ids (pairname plus segment) for find_strip_file().

    Falls back to pairnames for GeoDataFrames without a 'stac_id' column.
    """
    if 'stac_id' in items_gdf.columns:
        return items_gdf['stac_id'].tolist()
    return get_dem_metadata(items_gdf)[0]


def generate_orthogonal_transects(center_coords, half_length_m=3000):
    """Generate two perpendicular transects centered on a point.
    
    Creates a North-South transect and an East-West transect,
    both centered on the given coordinates.
    
    Parameters
    ----------
    center_coords : tuple
        (lon, lat) in WGS84 or (x, y) in EPSG:3413
    half_length_m : float
        Half-length of each transect in meters (default 3000 = 3km each way)
        
    Returns
    -------
    list of tuples
        [((start_ns, end_ns), 'N-S'), ((start_ew, end_ew), 'E-W')]
        Each transect is ((start_lon, start_lat), (end_lon, end_lat), label)
    """
    lon, lat = center_coords
    
    # Convert center to EPSG:3413 for distance calculations
    x_center, y_center = wgs84_to_3413(lon, lat)
    
    # North-South transect
    ns_start_3413 = (x_center, y_center + half_length_m)
    ns_end_3413 = (x_center, y_center - half_length_m)
    
    # East-West transect
    ew_start_3413 = (x_center - half_length_m, y_center)
    ew_end_3413 = (x_center + half_length_m, y_center)
    
    # Convert back to WGS84
    transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
    
    ns_start_lon, ns_start_lat = transformer.transform(ns_start_3413[0], ns_start_3413[1])
    ns_end_lon, ns_end_lat = transformer.transform(ns_end_3413[0], ns_end_3413[1])
    
    ew_start_lon, ew_start_lat = transformer.transform(ew_start_3413[0], ew_start_3413[1])
    ew_end_lon, ew_end_lat = transformer.transform(ew_end_3413[0], ew_end_3413[1])
    
    transects = [
        (((ns_start_lon, ns_start_lat), (ns_end_lon, ns_end_lat)), "North-South"),
        (((ew_start_lon, ew_start_lat), (ew_end_lon, ew_end_lat)), "East-West"),
    ]
    
    return transects


def read_transects_from_geoparquet(parquet_path, lat_col='lat', lon_col='lon', 
                                   label_col=None, half_length_m=3000):
    """Read transect center points from a GeoParquet file and create orthogonal transects.
    
    The file should contain points with latitude and longitude columns.
    For each point, generates N-S and E-W transects.
    
    Parameters
    ----------
    parquet_path : str
        Path to the GeoParquet file
    lat_col : str
        Name of latitude column (default: 'lat')
    lon_col : str
        Name of longitude column (default: 'lon')
    label_col : str, optional
        Name of column to use for labeling transects (e.g., 'lake_name')
        If None, uses row index
    half_length_m : float
        Half-length of each transect in meters
        
    Returns
    -------
    list of tuples
        [((start, end), label, direction), ...]
    """
    import geopandas as gpd
    
    print(f"\nReading transect points from: {parquet_path}")
    
    # Determine file type
    if parquet_path.endswith('.parquet'):
        gdf = gpd.read_parquet(parquet_path)
    elif parquet_path.endswith('.gpkg'):
        gdf = gpd.read_file(parquet_path)

    print(f"  Found {len(gdf)} points")
    print(f"  Columns: {list(gdf.columns)}")
    
    # Check if it has geometry
    if gdf.geometry is not None and not gdf.geometry.is_empty.all():
        print(f"  Using geometry column")
        # Extract coordinates from geometry
        lons = gdf.geometry.x.values
        lats = gdf.geometry.y.values
    elif lon_col in gdf.columns and lat_col in gdf.columns:
        print(f"  Using {lon_col}/{lat_col} columns")
        lons = gdf[lon_col].values
        lats = gdf[lat_col].values
    else:
        raise ValueError(f"Could not find coordinates. Expected geometry or {lon_col}/{lat_col} columns.")
    
    # Get labels
    if label_col and label_col in gdf.columns:
        labels = gdf[label_col].values
    else:
        labels = [f"Point_{i+1}" for i in range(len(gdf))]
    
    # Generate transects for each point
    all_transects = []
    
    for i, (lon, lat, label) in enumerate(zip(lons, lats, labels)):
        print(f"  {i+1}. {label}: ({lon:.4f}, {lat:.4f})")
        
        # Generate orthogonal transects
        transects = generate_orthogonal_transects((lon, lat), half_length_m)
        
        for transect_coords, direction in transects:
            all_transects.append((transect_coords, label, direction))
    
    print(f"\n  Generated {len(all_transects)} transects from {len(gdf)} points")
    
    return all_transects