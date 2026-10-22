"""
Stage 1: coarse event detection from existing 5x5-cluster elevation-history files.
Stage 2: refined volume from full-resolution zarr rasters, cropped to lake polygons,
         restricted to the event window found in Stage 1.
"""
# %%

import os
import re
import sys
from functools import lru_cache
import numpy as np
import pandas as pd
import ruptures as rpt
import xarray as xr
import rioxarray  # noqa: F401  (registers .rio accessor on xarray objects)
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import rasterio
import rasterio.features
import rasterio.transform
from shapely.geometry import shape

# Repo root on the path (same convention as the 01_/02_ notebooks), for config
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("."))
from config import OUTPUT_DIR, ARCHIVE_DIR

# All plots from this workflow go here (was: the current working directory)
VOLUME_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "volume")

# NOTE: the elevation-history files written by analysis.py contain raw
# window-mean elevations -- nothing in this repo detrends them. Stage 1 event
# detection below therefore runs on undetrended data; use simple_detrend()
# first if a long-term trend should be removed.


# ===========================================================================
# STAGE 1 -- parse existing elevation-history files, detect event windows
# ===========================================================================

def parse_elevation_history(filepath):
    """
    Parses the "Elevation History for <site_id>" text format into a DataFrame.
    Returns (metadata_dict, DataFrame[date, elev, std, valid_pixels, pairname]).
    """
    with open(filepath) as f:
        lines = f.readlines()

    meta = {}
    header_line_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Elevation History for"):
            meta["site_id"] = line.strip().split("for")[-1].strip()
        elif line.startswith("Coordinates:"):
            m = re.search(r"\(([-\d.]+).*?E,\s*([-\d.]+).*?N\)", line)
            if m:
                meta["lon"], meta["lat"] = float(m.group(1)), float(m.group(2))
        elif line.startswith("Coregistration:"):
            meta["coregistration"] = line.split(":")[-1].strip()
        elif line.startswith("Window:"):
            meta["window"] = line.split(":")[-1].strip()
        elif line.startswith("Date,"):
            header_line_idx = i
            break

    df = pd.read_csv(filepath, skiprows=header_line_idx, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Elevation (m)"]).sort_values("Date").reset_index(drop=True)
    return meta, df


def simple_detrend(df, elev_col="Elevation (m)"):
    """
    Removes a long-term linear trend (in decimal years) from the elevation series.
    Use on control/off-lake sites for noise-floor estimation, and optionally as a
    first pass on candidate sites before event detection (a real event should
    still show up as a local deviation from the linear fit).
    """
    t_years = (df["Date"] - df["Date"].min()).dt.days / 365.25
    coeffs = np.polyfit(t_years, df[elev_col], deg=1)
    trend = np.polyval(coeffs, t_years)
    return df[elev_col].values - trend, t_years.values


def estimate_noise_floor(control_filepaths, n_std=2.5):
    """
    control_filepaths: list of paths to elevation-history files from stable,
    off-lake sites (same cluster-window format).
    Returns a scalar threshold (in metres) for "real" signal vs. noise.
    """
    residual_stds = []
    for fp in control_filepaths:
        _, df = parse_elevation_history(fp)
        if len(df) < 4:
            continue
        detrended, _ = simple_detrend(df)
        residual_stds.append(np.std(detrended))
    return n_std * np.mean(residual_stds)


def detect_event_window(df, noise_floor, elev_col="Elevation (m)",
                         interp_freq_days=30, min_size=3, n_bkps=None):
    """
    Handles irregular sampling by linearly interpolating onto a uniform grid
    for change-point detection only; raw df is still used later for bounds.
    Returns list of (t_start, t_end) as pandas Timestamps.

    n_bkps: if given, uses Dynp with a FIXED number of breakpoints instead
    of Pelt's penalty-tuned approach -- e.g. if you can see 2 fill/drain
    episodes in a plot, try n_bkps=4 (each episode contributes a start and
    end breakpoint). Much easier to reason about than tuning `pen` blindly,
    and this is the recommended path if Pelt keeps returning the full range
    (a sign the penalty is too high for this series, not that there's
    nothing there). Still just a starting point -- compare against
    manual_windows_from_breakpoints() picked visually from the plot.
    """
    t0, t1 = df["Date"].min(), df["Date"].max()
    grid = pd.date_range(t0, t1, freq=f"{interp_freq_days}D")
    interp_vals = np.interp(
        grid.astype("int64"), df["Date"].astype("int64"), df[elev_col]
    )

    signal = interp_vals.reshape(-1, 1)
    if n_bkps is not None:
        algo = rpt.Dynp(model="rbf", min_size=min_size).fit(signal)
        breakpoints = algo.predict(n_bkps=n_bkps)
    else:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        pen = max(noise_floor, 0.05) * len(signal) * 0.1  # starting point, tune per-site
        breakpoints = algo.predict(pen=pen)

    windows = []
    prev = 0
    for bp in breakpoints:
        windows.append((grid[prev], grid[min(bp, len(grid) - 1)]))
        prev = bp
    return windows  # inspect visually -- pick the window(s) matching a real event


def robust_event_bounds(df, window, elev_col="Elevation (m)", low_pct=5, high_pct=95):
    """window: (t_start, t_end) as returned by detect_event_window."""
    mask = (df["Date"] >= window[0]) & (df["Date"] <= window[1])
    segment = df.loc[mask, elev_col]
    if len(segment) < 2:
        return None, None  # not enough points in this window at cluster level
    return np.percentile(segment, low_pct), np.percentile(segment, high_pct)


# ===========================================================================
# STAGE 2 -- refined volume from full-resolution zarr rasters + lake polygon
# ===========================================================================

# Folder naming, confirmed consistent across all strips:
#   processed_<platform>_<YYYYMMDD>_<catalogID>_dt.zarr
STRIP_DATE_PATTERN = re.compile(r"processed_[A-Za-z0-9]+_(\d{8})_")

TILE_ZARR_BASE = "/home/moralpom/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/tile_zarrs"
MEGATILE_SUBDIR = "megatile"  # tile_zarrs/megatile/<tile_id>_<cs2|mosaic>, kept with outliers
# Stacks rebuilt after the October 2026 GLOBE fixes carry "_rs" (resampled
# placement), e.g. cs2_..._deramp_rs_std30_mean20_outliersout_50m
_RS_TAG_RX = re.compile(r"_rs(_|$)")
# GLOBE's stacking logbook: one row per FINISHED stack (store_dir = its folder)
STACK_LOGBOOK = "/home/moralpom/luna/CPOM/moralpom/globe/data/stacking_tracking.duckdb"


def _finished_stack_dirs():
    """
    Folders of the stacks GLOBE has finished writing, from its DuckDB
    logbook. A "_rs" folder not listed there is still being written (or
    the run died), so it isn't used. Returns None if the logbook can't be
    read (e.g. locked by a running stacking job).
    """
    try:
        import duckdb
        with duckdb.connect(STACK_LOGBOOK, read_only=True) as con:
            rows = con.sql("SELECT store_dir FROM stack_runs").fetchall()
        return {os.path.normpath(r[0]) for r in rows if r[0]}
    except Exception as exc:
        print(f"warning: can't read the stacking logbook ({exc}); "
              f"can't tell finished _rs stacks from ones still being written")
        return None

# NOTE: the subfolder name is NOT consistent across tiles -- different tiles
# have been processed with different reference/filtering pipelines, e.g.
# "cs2_v_999_..." (coregistered to CryoSat-2) vs "mosaic_v_0_..." (coregistered
# to the ArcticDEM mosaic), with/without "_outliersout_", with/without "_50m"
# (confirmed elsewhere: that suffix is a leftover naming artifact, NOT actual
# resolution -- native res is still 2x2 m regardless). These are genuinely
# different processing choices, not just naming variants of the same data --
# deliberately NOT auto-picking one for you when more than one is present.
# No single default subfolder constant anymore; use list_tile_zarr_subfolders()
# to see what's there, and pass your choice explicitly to build_strip_index().


def list_tile_zarr_subfolders(tile_id, base_dir=TILE_ZARR_BASE):
    """
    Lists the processing-variant subfolders actually present for this tile
    (e.g. cs2- vs mosaic-referenced, with/without outlier removal). Use this
    to see your options before choosing one for build_strip_index().

    Also lists this tile's stacks in the shared megatile folder
    (megatile/<tile_id>_<cs2|mosaic>), returned as "megatile/<name>" so the
    name can be passed on as a subfolder like any other. Those were made
    for a colleague and keep outliers (they match the "outliersin" stacks),
    and they aren't updated when a tile is restacked. Empty folders,
    folders flagged "incomplete" in their name, and "_rs" stacks that
    GLOBE's logbook doesn't list as finished are skipped.
    """
    def has_strips(path):
        return any(e.name.endswith(".zarr") for e in os.scandir(path))

    names = []
    tile_dir = os.path.join(base_dir, tile_id)
    if os.path.isdir(tile_dir):
        names += [e.name for e in os.scandir(tile_dir) if e.is_dir()]
    mega_dir = os.path.join(base_dir, MEGATILE_SUBDIR)
    if os.path.isdir(mega_dir):
        names += [f"{MEGATILE_SUBDIR}/{e.name}" for e in os.scandir(mega_dir)
                  if e.is_dir() and e.name.startswith(f"{tile_id}_")]
    def path_of(n):
        return (os.path.join(base_dir, n) if n.startswith(f"{MEGATILE_SUBDIR}/")
                else os.path.join(base_dir, tile_id, n))

    names = [n for n in names if "incomplete" not in n and has_strips(path_of(n))]
    if any(_RS_TAG_RX.search(n) for n in names):
        finished = _finished_stack_dirs()
        if finished is not None:
            unfinished = [n for n in names if _RS_TAG_RX.search(n)
                          and os.path.normpath(path_of(n)) not in finished]
            for n in unfinished:
                print(f"note: skipping {tile_id}/{n}: not in the stacking "
                      f"logbook, so still being written or unfinished")
            names = [n for n in names if n not in unfinished]
    if not names:
        raise FileNotFoundError(f"No zarr stacks found for tile {tile_id} in "
                                f"{tile_dir} or {mega_dir}")
    return sorted(names)


def select_preferred_subfolder(tile_id, base_dir=TILE_ZARR_BASE):
    """
    Applies your stated preferences to the raw subfolder list:
    0. "nocoreg" (no coregistration) is dropped whenever a cs2/mosaic
       folder exists, and cs2/mosaic are recognised anywhere in the
       folder name (e.g. "12_39_1_2_cs2").
    1. "outliersout" (filters strips with SD > 50 m) is ALWAYS preferred
       over the equivalent non-outliersout variant of the same reference
       pipeline (cs2 or mosaic) -- collapsed automatically, no ambiguity.
       Rebuilt "_rs" stacks come before older ones whatever their outlier
       setting, since the older stacks predate the GLOBE stacking fixes.
       Then the tile's own folders, and the megatile copy (which keeps
       outliers) only when the tile folder has no stack of that reference.
    2. cs2 (CryoSat-2-referenced) vs mosaic (ArcticDEM-mosaic-referenced)
       has NO fixed preference -- your stated rule depends on the tile's
       proximity to the coastline/ice-sheet edge (enough exposed bedrock
       for reliable mosaic coregistration vs. relying on altimetry alone
       further inland). That's geographic judgement this function doesn't
       have, so if both remain after step 1, it still raises rather than
       guessing -- you decide per tile based on that heuristic.

    Returns the single resolved subfolder name if unambiguous; raises
    listing the remaining choice(s) otherwise.
    """
    available = list_tile_zarr_subfolders(tile_id, base_dir)

    def reference_type(name):
        # matched anywhere in the name: folders like "12_39_1_2_cs2" or
        # "zarr_coreg_cs2_..." are cs2 too
        if "nocoreg" in name:
            return "nocoreg"
        if "cs2" in name:
            return "cs2"
        if "mosaic" in name:
            return "mosaic"
        return name  # unrecognised naming -- treat as its own group, don't guess

    by_reference = {}
    for name in available:
        by_reference.setdefault(reference_type(name), []).append(name)

    # Uncoregistered data is only a last resort: drop it whenever a
    # coregistered (cs2/mosaic) alternative exists
    if "nocoreg" in by_reference and ({"cs2", "mosaic"} & set(by_reference)):
        by_reference.pop("nocoreg")

    collapsed = []
    for ref, names in by_reference.items():
        names = sorted(names, key=lambda n: (not _RS_TAG_RX.search(n),
                                             "outliersout" not in n,
                                             n.startswith(f"{MEGATILE_SUBDIR}/")))
        collapsed.append(names[0])

    if len(collapsed) == 1:
        chosen = collapsed[0]
        if "nocoreg" in chosen:
            print(f"WARNING: tile {tile_id}'s only available subfolder "
                  f"({chosen}) has NO coregistration applied -- this is a "
                  f"substantively noisier product than the cs2/mosaic-"
                  f"referenced data used elsewhere. Results for this tile "
                  f"should be treated as lower-confidence unless a "
                  f"coregistered alternative genuinely doesn't exist.")
        if not _RS_TAG_RX.search(chosen) and "nocoreg" not in chosen:
            print(f"note: tile {tile_id} has no rebuilt (_rs) stack yet; "
                  f"using the older {chosen}")
        if chosen.startswith(f"{MEGATILE_SUBDIR}/"):
            print(f"note: tile {tile_id} uses the megatile stack {chosen}, "
                  f"which keeps outliers (no outliersout stack exists)")
        return chosen

    raise ValueError(
        f"tile {tile_id}: outliersout preference applied, but still need a "
        f"cs2-vs-mosaic choice based on proximity to the ice-sheet edge -- "
        f"pick explicitly from: {collapsed}"
    )


def get_tile_zarr_dir(tile_id, subfolder, base_dir=TILE_ZARR_BASE):
    """
    subfolder: REQUIRED, no default -- see list_tile_zarr_subfolders(tile_id)
    to see what's available for this specific tile before choosing. Picking
    silently would be a real methodological choice (which reference/filter
    pipeline the strips were processed with), not just a naming detail.
    """
    if subfolder.startswith(f"{MEGATILE_SUBDIR}/"):
        path = os.path.join(base_dir, subfolder)
    else:
        path = os.path.join(base_dir, tile_id, subfolder)
    if not os.path.isdir(path):
        available = list_tile_zarr_subfolders(tile_id, base_dir)
        raise FileNotFoundError(
            f"{path} doesn't exist. Available subfolders for tile {tile_id}: "
            f"{available}"
        )
    return path


def build_strip_index(tile_id, subfolder=None):
    """
    Scans a tile's zarr directory and returns {pd.Timestamp: [zarr_paths]}.
    Uses site->tile_id mapping you already maintain elsewhere -- pass the
    tile_id for the site you're processing.
    A date maps to a list because more than one strip can share an acquisition
    date (different catalog IDs); keep all of them, filtering happens later.

    subfolder: which processing-variant folder to read strips from (e.g.
    "cs2_v_999_dh_999_vertical_offset_mean_nuthkaab_deramp_outliersout_50m").
    If omitted, uses select_preferred_subfolder() -- auto-resolves the
    outliersout preference, but still raises if a cs2-vs-mosaic choice
    remains (that depends on the tile's proximity to the ice-sheet edge,
    which this function doesn't know).
    """
    if subfolder is None:
        subfolder = select_preferred_subfolder(tile_id)
        print(f"using subfolder: {subfolder}")

    tile_dir = get_tile_zarr_dir(tile_id, subfolder)
    strip_index = {}
    for entry in os.scandir(tile_dir):
        if not (entry.is_dir() and entry.name.endswith(".zarr")):
            continue
        match = STRIP_DATE_PATTERN.search(entry.name)
        if not match:
            print(f"warning: couldn't parse date from {entry.name}, skipping")
            continue
        date = pd.to_datetime(match.group(1), format="%Y%m%d")
        strip_index.setdefault(date, []).append(entry.path)
    return strip_index


def strips_in_window(strip_index, window):
    """window: (t_start, t_end) as returned by detect_event_window."""
    t0, t1 = window
    paths = []
    for date, paths_for_date in strip_index.items():
        if t0 <= date <= t1:
            paths.extend(paths_for_date)
    return paths


def load_lake_polygon(polygon_path, site_id, target_crs=None):
    """
    polygon_path: shapefile/geojson with a lake-boundary polygon per site.
    site_id: matches the identifier used in your delineation (e.g. "11_39_1_1").
    """
    gdf = gpd.read_file(polygon_path)
    lake = gdf[gdf["site_id"] == site_id]
    if lake.empty:
        raise ValueError(f"No polygon found for site {site_id}")
    if target_crs is not None:
        lake = lake.to_crs(target_crs)
    return lake


def get_site_lonlat_from_polygon(lake_gdf, tile_crs=None):
    """
    Returns (lon, lat) from the polygon's centroid -- use this instead of
    relying on an elevation-history file's parsed coordinates (meta["lon"]/
    meta["lat"]), which some candidates won't have. Can fully replace that
    dependency for validate_georeferencing() and anywhere else a site's
    approximate coordinates are needed.

    IMPORTANT: centroid must be computed in a PROJECTED (metric) CRS, not
    geographic (lon/lat degrees) -- a centroid computed directly in degrees
    can be meaningfully distorted, especially for larger/irregular polygons.
    If lake_gdf isn't already in a projected CRS, pass tile_crs to reproject
    first (same tile_crs from get_tile_transform(tile_id)).
    """
    if tile_crs is not None and lake_gdf.crs != tile_crs:
        proj_gdf = lake_gdf.to_crs(tile_crs)
    else:
        proj_gdf = lake_gdf

    if proj_gdf.crs is not None and proj_gdf.crs.is_geographic:
        raise ValueError(
            "lake_gdf is in a geographic CRS (degrees) -- pass tile_crs= "
            "to reproject before computing a centroid, or the centroid "
            "location can be distorted."
        )

    centroid_proj = proj_gdf.geometry.centroid.iloc[0]
    centroid_4326 = (gpd.GeoDataFrame(geometry=[centroid_proj], crs=proj_gdf.crs)
                      .to_crs("EPSG:4326").geometry.iloc[0])
    return centroid_4326.x, centroid_4326.y  # lon, lat


# ---------------------------------------------------------------------------
# Georeferencing -- the zarr arrays themselves have no CRS/transform in
# .attrs (confirmed empty). Origin comes from the PGC mosaic tile index.
#
# IMPORTANT: the index polygon bounds are ~100 m larger per edge than the
# actual 25000x25000 px @ 2 m array (50,200 m span vs. 50,000 m). This looks
# like a buffer/overlap baked into the index geometry, not a real data
# extent. The function below derives the actual buffer from the discrepancy
# between the index bounds and the known array shape/resolution -- rather
# than hardcoding "100 m" -- so it self-corrects if the buffer differs for
# other tiles. Still: run validate_georeferencing() below on a known site
# before trusting this on lakes you haven't already checked by eye.
# ---------------------------------------------------------------------------

TILE_INDEX_PATH = (
    "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/mosaic/"
    "ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
)
TILE_CRS = "EPSG:3413"
TILE_RESOLUTION = 2.0
TILE_SHAPE = (25000, 25000)  # (rows, cols) -- confirmed from zarr diagnostic


@lru_cache(maxsize=None)
def get_tile_transform(tile_id, index_path=TILE_INDEX_PATH,
                        resolution=TILE_RESOLUTION, shape=TILE_SHAPE):
    """
    Returns (transform, crs) as an affine.Affine + CRS string, with the
    index polygon's buffer automatically stripped based on the known array
    shape, rather than a hardcoded 100 m assumption.

    Cached per tile: every crop calls this, and reading the mosaic index
    shapefile each time dominated run time.
    """
    import affine

    gdf = gpd.read_file(index_path)
    row = gdf[gdf["tile"] == tile_id]
    if row.empty:
        raise ValueError(f"tile_id {tile_id!r} not found in index (column 'tile')")

    minx, miny, maxx, maxy = row.total_bounds
    nrows, ncols = shape

    # Derive the symmetric buffer from the mismatch between index bounds and
    # the true data extent (ncols * resolution), instead of assuming a fixed
    # value -- keeps this correct even if the buffer differs per tile/release.
    index_width = maxx - minx
    true_width = ncols * resolution
    buffer = (index_width - true_width) / 2.0

    origin_x = minx + buffer
    origin_y = maxy - buffer  # top-left corner (north-up raster convention)

    transform = affine.Affine(resolution, 0, origin_x, 0, -resolution, origin_y)
    crs_epsg = row["epsg"].iloc[0] if "epsg" in row.columns else 3413
    return transform, f"EPSG:{int(crs_epsg)}"


def zarr_path_from_pairname(tile_id, pairname, subfolder=None, base_dir=TILE_ZARR_BASE):
    """
    Maps a history-file Pairname (e.g. 'WV01_20241007_1020010104A74400_
    1020010104A87A00') directly to its zarr folder, since the naming
    convention matches. Use this for validation -- it guarantees the strip
    actually has valid data at the site (it's literally where that history
    row's elevation value came from).

    subfolder: same requirement as build_strip_index() -- if omitted, uses
    select_preferred_subfolder() (auto-resolves outliersout preference,
    still raises on a genuine cs2-vs-mosaic choice).
    """
    if subfolder is None:
        subfolder = select_preferred_subfolder(tile_id, base_dir)
    return os.path.join(get_tile_zarr_dir(tile_id, subfolder, base_dir),
                         f"processed_{pairname}_dt.zarr")


def validate_georeferencing(zarr_path, tile_id, site_lon, site_lat,
                             expected_elev=None, tol_m=200, nodata=-9999.0):
    """
    Empirical sanity check -- run this BEFORE trusting crop_zarr_to_polygon()
    on any new tile. Converts a known site's (lon, lat) to the tile CRS,
    computes the corresponding pixel row/col, reads a small window there,
    and prints it for a by-eye comparison against expected_elev.

    Use zarr_path_from_pairname() to get a path GUARANTEED to cover the site
    (matched from the history file's own Pairname column) -- an arbitrary
    strip very likely misses the site entirely (most strips only cover part
    of a tile), which shows up as all-nodata and looks like a transform bug
    when it isn't.

    nodata: sentinel for missing pixels in these zarr stores (confirmed
    -9999.0, NOT NaN) -- must be masked before any statistics.
    """
    import zarr
    from pyproj import Transformer

    transform, crs = get_tile_transform(tile_id)
    to_tile_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_tile_crs.transform(site_lon, site_lat)

    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)

    z = zarr.open(zarr_path, mode="r")
    half = 5
    window = z[max(row - half, 0):row + half, max(col - half, 0):col + half]
    window = np.where(window == nodata, np.nan, window)

    print(f"site (lon={site_lon}, lat={site_lat}) -> tile CRS x={x:.1f}, y={y:.1f}")
    print(f"-> pixel row={row}, col={col} (array shape {z.shape})")
    print(f"local window values (nodata masked):\n{window}")
    valid_frac = np.mean(~np.isnan(window))
    print(f"valid fraction in window: {valid_frac:.0%}")
    if valid_frac == 0:
        print("all-nodata -- this strip doesn't cover the site; use "
              "zarr_path_from_pairname() with a row known to have valid data")
        return
    print(f"local window mean: {np.nanmean(window):.2f} m")
    if expected_elev is not None:
        diff = abs(np.nanmean(window) - expected_elev)
        flag = "OK" if diff < tol_m else "MISMATCH -- check transform/tile_id"
        print(f"expected ~{expected_elev:.2f} m, diff={diff:.2f} m [{flag}]")


def crop_zarr_to_polygon(zarr_path, lake_gdf, tile_id, nodata=-9999.0):
    """
    Opens the raw zarr array via dask, attaches coordinates/CRS derived from
    get_tile_transform(), masks the nodata sentinel to NaN, then crops to
    the polygon. Only run this after validate_georeferencing() has confirmed
    alignment for this tile.

    nodata masking happens BEFORE cropping and BEFORE any downstream
    statistics -- confirmed these zarr stores use -9999.0 as a literal fill
    value, not NaN, so every quantile/mean/notnull() call downstream would
    silently treat missing pixels as extreme valid readings otherwise.
    """
    import zarr
    import dask.array as dsa

    z = zarr.open(zarr_path, mode="r")
    darr = dsa.from_zarr(z)
    darr = dsa.where(darr == nodata, np.nan, darr)

    transform, crs = get_tile_transform(tile_id)
    nrows, ncols = darr.shape
    xs = transform.c + (np.arange(ncols) + 0.5) * transform.a
    ys = transform.f + (np.arange(nrows) + 0.5) * transform.e

    da = xr.DataArray(darr, dims=("y", "x"), coords={"x": xs, "y": ys})
    da = da.rio.write_crs(crs)

    if lake_gdf.crs != crs:
        lake_gdf = lake_gdf.to_crs(crs)

    minx, miny, maxx, maxy = lake_gdf.total_bounds
    da = da.rio.clip_box(minx, miny, maxx, maxy)
    da = da.rio.clip(lake_gdf.geometry, lake_gdf.crs, drop=True)
    return da  # still lazy


def build_polygon_mean_timeseries(strip_index, lake_gdf, tile_id, nodata=-9999.0):
    """
    Streams through every strip in strip_index (typically build_strip_index()'s
    full output, NOT window-filtered), crops each to the lake polygon, and
    records per-strip aggregate stats. Doesn't hold all strips in memory at
    once -- each crop is computed and discarded before moving to the next, so
    memory use stays at "one small polygon crop" regardless of strip count.

    This is a much cleaner signal than the 5x5-cluster history file (uses
    every pixel in the polygon, not just 25), and std_elev here is a useful
    diagnostic for error bars / spotting bad strips -- but note it's SPATIAL
    std across the polygon at that epoch (mixes real within-lake surface
    variation with strip noise), not a formal measurement-uncertainty value;
    be precise about that distinction if it ends up in a figure/Methods text.

    Returns a DataFrame sorted by date: date, mean_elev, std_elev,
    n_valid_pixels, n_total_pixels, path.
    """
    records = []
    for date, paths_for_date in sorted(strip_index.items()):
        for zp in paths_for_date:
            try:
                da = crop_zarr_to_polygon(zp, lake_gdf, tile_id, nodata=nodata)
                vals = da.values  # triggers compute -- only for this small crop
            except Exception as e:
                print(f"skipping {zp}: {e}")
                continue
            n_valid = int(np.sum(~np.isnan(vals)))
            if n_valid == 0:
                continue
            records.append({
                "date": date,
                "mean_elev": float(np.nanmean(vals)),
                "std_elev": float(np.nanstd(vals)),
                "n_valid_pixels": n_valid,
                "n_total_pixels": int(vals.size),
                "path": zp,
            })
    if not records:
        raise ValueError("No strips produced valid data over this polygon")
    return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)


def build_lake_buffer_ring(lake_gdf, buffer_dist_m=None, inner_gap_m=0, tile_crs=None):
    """
    Returns a ring (annulus) polygon around the lake: an outer buffer minus
    an inner buffer.

    buffer_dist_m: if None (default), uses Fan et al. 2023's ADAPTIVE width
    -- the radius of a circle with the same area as the lake polygon --
    rather than a fixed distance for every lake regardless of size. This
    matches their confirmed approach (they tested a fixed 2km ring and
    found it performed inconsistently, since most lakes are far smaller
    than 2km across; the adaptive width keeps the ring's footprint
    comparable in scale to the lake itself). Pass a number to override
    with a fixed width instead.

    inner_gap_m: Fan et al. use NO gap (0, the default) -- the ring starts
    right at the polygon edge. A non-zero value is a deliberate deviation
    from their method, protecting against delineation-boundary uncertainty
    at the cost of matching their approach less exactly -- your call, but
    know which one you're doing.

    IMPORTANT: buffer_dist_m/inner_gap_m are in METRES, so lake_gdf must be
    in a projected (metric) CRS before this runs -- buffering a geographic
    (lon/lat degrees) CRS by "500" silently buffers by 500 DEGREES, produces
    a garbage/invalid ring geometry, and surfaces later as a confusing
    "cannot convert float NaN to integer" error inside rio.clip_box, not
    here. If tile_crs is given and lake_gdf isn't already in it, reprojects
    first rather than trusting the caller got this right.
    """
    if tile_crs is not None and lake_gdf.crs != tile_crs:
        lake_gdf = lake_gdf.to_crs(tile_crs)
    elif lake_gdf.crs is not None and lake_gdf.crs.is_geographic:
        raise ValueError(
            "lake_gdf is in a geographic CRS (degrees) -- pass tile_crs= "
            "to reproject before buffering, or reproject lake_gdf yourself "
            "first. Buffering in degrees will silently produce garbage."
        )

    if buffer_dist_m is None:
        area = float(lake_gdf.geometry.area.iloc[0])
        buffer_dist_m = float(np.sqrt(area / np.pi))  # equivalent-circle radius

    outer = lake_gdf.geometry.buffer(buffer_dist_m)
    inner = lake_gdf.geometry.buffer(inner_gap_m) if inner_gap_m > 0 else lake_gdf.geometry
    ring = outer.difference(inner)
    return gpd.GeoDataFrame(geometry=ring, crs=lake_gdf.crs)


def build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id,
                                       buffer_dist_m=None, inner_gap_m=0,
                                       nodata=-9999.0, return_diagnostics=False,
                                       ring_gap_frac=None, ring_width_frac=1.0):
    """
    Per strip: crops both the lake polygon and its surrounding buffer ring,
    and records the ring's mean as a local background estimate. "corrected"
    = lake_mean - buffer_mean isolates the lake-specific signal from
    regional elevation-change trends/noise, matching Fan et al. 2023's
    background-subtraction approach -- more robust than a single distant
    control site, since the buffer ring shares the same strip footprint
    geometry and local conditions as the lake itself.

    Streams strip-by-strip like build_polygon_mean_timeseries(); same
    memory profile (small, discarded per strip).

    NOTE on coverage: crop_zarr_to_polygon() crops against the full TILE
    grid, not the strip's own footprint, so it essentially never raises
    just because a narrow strip doesn't reach the polygon -- it just leaves
    those pixels NaN. The real exclusion happens below (n_lake==0 or
    n_buf==0), which is now logged, not silent -- see return_diagnostics.

    return_diagnostics: if True, returns (ts_df, diagnostics_df) instead of
    just ts_df. diagnostics_df has ONE ROW PER STRIP CONSIDERED (kept or
    not), with a "status" column ("kept", "zero_lake_coverage",
    "zero_buffer_coverage", "zero_both", "exception") plus n_valid_lake/
    n_valid_buffer counts -- use this to actually see which strips got
    dropped and why, instead of guessing.

    ring_gap_frac: if given, use the gapped background ring instead
    (build_background_ring: starts ring_gap_frac x r outside the polygon,
    ring_width_frac x r wide), keeping lake area outside the hand-drawn
    outline out of the background. None = Fan et al. ring (no gap).
    """
    _, tile_crs = get_tile_transform(tile_id)
    if lake_gdf.crs != tile_crs:
        lake_gdf = lake_gdf.to_crs(tile_crs)  # do this ONCE, before buffering
    if ring_gap_frac is not None:
        buffer_gdf = build_background_ring(lake_gdf, ring_gap_frac, ring_width_frac,
                                           tile_crs=tile_crs)
    else:
        buffer_gdf = build_lake_buffer_ring(lake_gdf, buffer_dist_m, inner_gap_m,
                                             tile_crs=tile_crs)

    records = []
    diagnostics = []
    for date, paths_for_date in sorted(strip_index.items()):
        for zp in paths_for_date:
            try:
                lake_vals = crop_zarr_to_polygon(zp, lake_gdf, tile_id, nodata).values
                buf_vals = crop_zarr_to_polygon(zp, buffer_gdf, tile_id, nodata).values
            except Exception as e:
                print(f"skipping {zp}: {e}")
                diagnostics.append({"date": date, "path": zp, "status": "exception",
                                     "n_valid_lake": None, "n_valid_buffer": None})
                continue

            n_lake = int(np.sum(~np.isnan(lake_vals)))
            n_buf = int(np.sum(~np.isnan(buf_vals)))
            if n_lake == 0 or n_buf == 0:
                status = ("zero_both" if n_lake == 0 and n_buf == 0
                          else "zero_lake_coverage" if n_lake == 0
                          else "zero_buffer_coverage")
                diagnostics.append({"date": date, "path": zp, "status": status,
                                     "n_valid_lake": n_lake, "n_valid_buffer": n_buf})
                continue

            lake_mean = float(np.nanmean(lake_vals))
            buf_mean = float(np.nanmean(buf_vals))
            diagnostics.append({"date": date, "path": zp, "status": "kept",
                                 "n_valid_lake": n_lake, "n_valid_buffer": n_buf})
            records.append({
                "date": date,
                "lake_mean": lake_mean,
                "buffer_mean": buf_mean,
                "corrected": lake_mean - buf_mean,
                "lake_std": float(np.nanstd(lake_vals)),
                "buffer_std": float(np.nanstd(buf_vals)),
                "n_valid_lake": n_lake,
                "n_valid_buffer": n_buf,
                "path": zp,
            })

    diagnostics_df = pd.DataFrame.from_records(diagnostics).sort_values("date").reset_index(drop=True)
    status_counts = diagnostics_df["status"].value_counts().to_dict()
    print(f"strip status breakdown: {status_counts}")

    if not records:
        raise ValueError("No strips produced valid lake+buffer data -- check "
                          "diagnostics_df (return_diagnostics=True) for why")
    ts_df = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)

    if return_diagnostics:
        return ts_df, diagnostics_df
    return ts_df



def estimate_noise_floor_from_buffer(ts_df, n_mad=2.5, col="corrected"):
    """
    Robust noise-floor estimate from the buffer-corrected series itself --
    replaces the separate distant-control-site approach for confirmed lakes.
    Uses MAD (median absolute deviation) scaled to be std-equivalent for
    normal data, since a plain std would be inflated by the real event
    itself; MAD is far more robust to that one large excursion dominating
    the estimate.
    """
    x = ts_df[col].values
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    robust_std = 1.4826 * mad
    return n_mad * robust_std


def manual_window(start_date, end_date):
    """
    Bypass automatic detection entirely -- pass dates you've picked visually
    from the plot below, e.g. manual_window("2015-06-01", "2016-09-01").
    Returns a (t_start, t_end) tuple in the same format detect_event_window*
    functions return, so it drops straight into strips_in_window().
    """
    return pd.Timestamp(start_date), pd.Timestamp(end_date)


def manual_windows_from_breakpoints(*dates):
    """
    Given a sorted list of dates marking episode boundaries (picked visually
    from the plot), returns consecutive (start, end) window pairs -- e.g.
    manual_windows_from_breakpoints("2013-06-01", "2015-03-01", "2019-01-01")
    returns [(2013-06-01, 2015-03-01), (2015-03-01, 2019-01-01)], one window
    per episode. Simplest reliable path given the automatic detector keeps
    returning the full range -- probably faster than further penalty-tuning
    under time pressure.
    """
    ts = sorted(pd.Timestamp(d) for d in dates)
    if len(ts) < 2:
        raise ValueError("Need at least 2 dates to form a window")
    return list(zip(ts[:-1], ts[1:]))


def detect_event_window_from_series(dates, values, noise_floor,
                                     interp_freq_days=30, min_size=3, n_bkps=None):
    """
    Same change-point logic as detect_event_window(), generalised to take a
    plain (dates, values) pair -- use this on build_buffer_corrected_timeseries()'s
    output ("date", "corrected" columns), which should be far less noisy than
    the cluster-based series and give the detector a much better chance.
    n_bkps: see detect_event_window() -- try this if Pelt keeps returning
    the full range.
    """
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "elev": values})
    return detect_event_window(df, noise_floor, elev_col="elev",
                                interp_freq_days=interp_freq_days,
                                min_size=min_size, n_bkps=n_bkps)


def build_before_after_diff_map(strip_index, lake_gdf, tile_id,
                                 before_window, after_window,
                                 min_valid_per_group=2, background_correct=True,
                                 buffer_dist_m=None, inner_gap_m=0,
                                 nodata=-9999.0, verbose=True):
    """
    Alternative to refined_volume_from_zarr()'s per-pixel-independent-
    percentile map: instead of each pixel drawing its high/low from
    whichever strips happen to cover it (which produces incoherent,
    grainy-looking maps when strips have residual mis-coregistration
    noise), every pixel here is compared using the SAME TWO STRIP GROUPS
    -- a "before" period and an "after" period you pick manually (e.g.
    from eyeballing the buffer-corrected timeseries or transect plot).
    diff = mean(after-group strips) - mean(before-group strips), per pixel.

    This won't fix underlying strip-to-strip mis-registration noise, but
    it removes the incoherence caused by different pixels being built from
    different, uncontrolled strip subsets -- so a real, spatially coherent
    signal (if present) should show up more clearly against the noise
    floor than in the percentile-based map.

    before_window, after_window: (t_start, t_end) tuples, e.g. from
    manual_window() -- pick two periods you're confident sit on either
    side of the fill/drain event, not necessarily adjacent to each other.

    min_valid_per_group: minimum strips covering a pixel within EACH group
    for that pixel to count -- 1 lets more pixels through but each mean
    could be a single noisy strip; 2+ is more robust but lowers coverage.
    Same tradeoff as refined_volume_from_zarr's coverage filtering.

    background_correct: if True, also computes the same before/after mean
    difference over a buffer ring (build_lake_buffer_ring) and subtracts
    it from the lake's diff map -- same rationale as
    build_buffer_corrected_timeseries, removing regional background trend
    between the two periods rather than just within-lake noise.

    Returns a dict with "diff_map" (2D DataArray, drop straight into
    plot_pixel_change_map) plus before/after coverage fractions and strip
    counts per group.
    """
    def _group_mean_map(paths, crop_gdf):
        cropped = []
        for zp in paths:
            try:
                cropped.append(crop_zarr_to_polygon(zp, crop_gdf, tile_id, nodata))
            except Exception as e:
                print(f"skipping {zp}: {e}")
        if len(cropped) < min_valid_per_group:
            raise ValueError(
                f"Only {len(cropped)} usable strips in this group, need "
                f">= {min_valid_per_group}"
            )
        stack = xr.concat(cropped, dim="strip")
        valid_count = stack.notnull().sum(dim="strip")
        has_enough = valid_count >= min_valid_per_group
        coverage = float(has_enough.compute().mean())
        if coverage == 0:
            max_overlap = int(valid_count.max().compute())
            raise ValueError(
                f"No pixel has >= {min_valid_per_group} overlapping strips "
                f"in this group (best pixel has {max_overlap}) -- with only "
                f"{len(cropped)} strips, narrow ArcticDEM swaths may simply "
                f"not stack {min_valid_per_group}-deep anywhere over this "
                f"polygon. Try min_valid_per_group=1, or widen the window "
                f"to pull in more strips."
            )
        mean_map = stack.mean(dim="strip", skipna=True).where(has_enough)
        coverage = float(has_enough.compute().mean())
        return mean_map, coverage, len(cropped)

    before_paths = strips_in_window(strip_index, before_window)
    after_paths = strips_in_window(strip_index, after_window)

    before_mean, before_cov, n_before = _group_mean_map(before_paths, lake_gdf)
    after_mean, after_cov, n_after = _group_mean_map(after_paths, lake_gdf)

    # Subtract on raw numpy values, NOT the xarray objects directly --
    # after_mean - before_mean can silently return all-NaN if the two
    # coordinate arrays differ by even a tiny float amount (e.g. from two
    # independently-run .rio.clip() calls on paper-identical geometry):
    # xarray aligns by coordinate LABEL before subtracting, and a failed
    # label match gives an empty-looking result with no error raised.
    before_vals = before_mean.compute().values
    after_vals = after_mean.compute().values
    diff_vals = after_vals - before_vals
    diff_map = xr.DataArray(diff_vals, dims=("y", "x"),
                             coords={"x": after_mean["x"].values,
                                     "y": after_mean["y"].values})

    result = {
        "diff_map": diff_map,
        "before_coverage": before_cov,
        "after_coverage": after_cov,
        "n_strips_before": n_before,
        "n_strips_after": n_after,
    }

    if background_correct:
        _, tile_crs = get_tile_transform(tile_id)
        lake_proj = lake_gdf if lake_gdf.crs == tile_crs else lake_gdf.to_crs(tile_crs)
        buffer_gdf = build_lake_buffer_ring(lake_proj, buffer_dist_m, inner_gap_m,
                                             tile_crs=tile_crs)
        buf_before_mean, _, _ = _group_mean_map(before_paths, buffer_gdf)
        buf_after_mean, _, _ = _group_mean_map(after_paths, buffer_gdf)
        # same raw-numpy fix as above -- avoid xarray label-alignment
        buf_before_vals = buf_before_mean.compute().values
        buf_after_vals = buf_after_mean.compute().values
        background_diff = float(np.nanmean(buf_after_vals - buf_before_vals))
        diff_map = diff_map - background_diff  # DataArray - python float: safe, no alignment
        result["diff_map"] = diff_map
        result["background_diff_m"] = background_diff

    if verbose:
        print(f"before: {n_before} strips, {before_cov:.0%} coverage | "
              f"after: {n_after} strips, {after_cov:.0%} coverage")
        if background_correct:
            print(f"background (buffer ring) diff subtracted: "
                  f"{result['background_diff_m']:.3f} m")
        valid = diff_map.values[~np.isnan(diff_map.values)]
        if valid.size > 0:
            print(f"diff map: min={valid.min():.2f}, max={valid.max():.2f}, "
                  f"mean={valid.mean():.2f}, median={np.median(valid):.2f} m")

    return result


def full_range_dz_map(strip_index, lake_gdf, tile_id, pixel_area=4.0,
                       low_pct=5, high_pct=95, verbose=True):
    """
    Companion to the "full range, unwindowed" polygon timeseries plot --
    same per-pixel-independent-percentile method as refined_volume_from_zarr,
    but run across EVERY strip in strip_index rather than one narrowed
    window. Shows the total observed spread per pixel over the whole
    record, for visual comparison against long-term trends in the
    timeseries (not a single-event volume estimate -- treat it as a
    diagnostic map, not a number to report).
    """
    all_paths = [p for paths in strip_index.values() for p in paths]
    full_window = (min(strip_index.keys()), max(strip_index.keys()))
    return refined_volume_from_zarr(all_paths, lake_gdf, tile_id, pixel_area,
                                     low_pct=low_pct, high_pct=high_pct,
                                     window=full_window, verbose=verbose)


def full_range_diff_map(strip_index, lake_gdf, tile_id, split_fraction=0.2,
                         min_valid_per_group=2, background_correct=True,
                         buffer_dist_m=None, inner_gap_m=0, verbose=True):
    """
    Companion diff map spanning the WHOLE record, not one episode: splits
    the full available date range into an early group (first split_fraction
    of the span) and a late group (last split_fraction), then runs the
    same before/after mean-difference method. Useful for spotting a
    longer-term net change alongside the full timeseries plot, distinct
    from any one detected fill/drain episode.

    split_fraction: 0.2 = first/last 20% of the total date range. Widen if
    either group ends up with too few strips (check n_strips_before/after
    in the printed output).
    """
    t_min, t_max = min(strip_index.keys()), max(strip_index.keys())
    span = t_max - t_min
    before_window = (t_min, t_min + span * split_fraction)
    after_window = (t_max - span * split_fraction, t_max)
    if verbose:
        print(f"full-range split: before={before_window}, after={after_window}")
    return build_before_after_diff_map(strip_index, lake_gdf, tile_id,
                                        before_window, after_window,
                                        min_valid_per_group=min_valid_per_group,
                                        background_correct=background_correct,
                                        buffer_dist_m=buffer_dist_m,
                                        inner_gap_m=inner_gap_m, verbose=verbose)


def refined_volume_from_zarr(strip_paths_in_window, lake_gdf, tile_id, pixel_area,
                              low_pct=5, high_pct=95, window=None, verbose=True):
    """
    strip_paths_in_window: list of zarr paths for strips whose acquisition dates
    fall inside the Stage-1 event window (filter this list using your existing
    strip-date index before calling).

    Mechanically: for each pixel independently, take the [low_pct, high_pct]
    range of THAT pixel's own values across the strips with enough data (>=2
    valid epochs) -- NOT interpolated for pixels with insufficient data, just
    excluded (NaN, contributes 0 to the sum). Multiply each pixel's range by
    pixel_area and sum. This is NOT area_total x mean_dz -- it's a per-pixel
    volume computed independently, then summed; coverage_fraction tells you
    what share of the polygon's pixels actually contributed.

    window: optional (t_start, t_end) -- if given and duration > 0, also
    computes an approximate m3/yr rate (total volume / window duration in
    years) for comparison against literature RATES. Note this is a total
    divided by duration, not a linear-fit slope over a fixed observation
    period like some published rates use -- same units, not the same
    quantity; state that distinction if comparing directly.

    Returns a dict now (not a bare tuple) -- see keys below -- including the
    2D per-pixel dz_range map (dz_map) for use with plot_pixel_change_map().
    """
    cropped_arrays = []
    for zp in strip_paths_in_window:
        try:
            cropped_arrays.append(crop_zarr_to_polygon(zp, lake_gdf, tile_id))
        except Exception as e:
            print(f"skipping {zp}: {e}")

    if len(cropped_arrays) < 2:
        raise ValueError("Need at least 2 strips in-window for a dz range")

    stack = xr.concat(cropped_arrays, dim="strip")  # (strip, y, x), still lazy

    valid_count = stack.notnull().sum(dim="strip")
    has_enough = (valid_count >= 2)

    dz_low = stack.quantile(low_pct / 100, dim="strip", skipna=True)
    dz_high = stack.quantile(high_pct / 100, dim="strip", skipna=True)
    dz_range = (dz_high - dz_low).where(has_enough)

    # .compute() triggers actual reads -- only for the cropped region, not full tiles
    dz_range_vals = dz_range.compute()
    has_enough_vals = has_enough.compute()
    coverage_fraction = float(has_enough_vals.mean())

    n_total_pixels = int(has_enough_vals.size)
    n_contributing_pixels = int(has_enough_vals.values.sum())
    nominal_polygon_area = n_total_pixels * pixel_area
    contributing_area = n_contributing_pixels * pixel_area

    volume = float(np.nansum(dz_range_vals.values)) * pixel_area

    # "Rounded up" volume -- extrapolates the measured (covered-pixels-only)
    # volume to the full nominal polygon area, assuming uncovered pixels
    # behave like the average of the covered ones. Mathematically identical
    # to mean-filling the missing pixels then summing over the full area:
    # scaling by 1/coverage_fraction. Reasonable if missing coverage is
    # roughly scattered through the polygon; less reasonable if missing
    # pixels cluster systematically (e.g. mostly at the lake edge or mostly
    # in the deepest/most-active center) -- worth a quick look at dz_map's
    # spatial pattern of NaNs before leaning on this for a headline number.
    extrapolated_volume = (volume / coverage_fraction
                            if coverage_fraction > 0 else np.nan)

    result = {
        "volume_m3": volume,
        "extrapolated_volume_m3": extrapolated_volume,
        "coverage_fraction": coverage_fraction,
        "n_strips": len(cropped_arrays),
        "nominal_polygon_area_m2": nominal_polygon_area,
        "contributing_area_m2": contributing_area,
        "n_total_pixels": n_total_pixels,
        "n_contributing_pixels": n_contributing_pixels,
        "dz_map": dz_range_vals,  # 2D DataArray -- for plot_pixel_change_map()
    }

    if window is not None:
        duration_years = (window[1] - window[0]).total_seconds() / (365.25 * 24 * 3600)
        result["duration_years"] = duration_years
        result["volume_rate_m3_per_yr"] = volume / duration_years if duration_years > 0 else np.nan
        result["extrapolated_volume_rate_m3_per_yr"] = (
            extrapolated_volume / duration_years if duration_years > 0 else np.nan
        )

    if verbose:
        print(f"volume (covered pixels only): {volume:.2e} m3 | "
              f"extrapolated (full polygon): {extrapolated_volume:.2e} m3")
        print(f"coverage: {coverage_fraction:.0%} "
              f"({n_contributing_pixels}/{n_total_pixels} px) | n_strips={len(cropped_arrays)}")
        print(f"nominal polygon area: {nominal_polygon_area:.2e} m2 | "
              f"contributing area: {contributing_area:.2e} m2")
        if window is not None:
            print(f"window duration: {duration_years:.2f} yr | "
                  f"implied rate: {result['volume_rate_m3_per_yr']:.2e} m3/yr "
                  f"(total/duration -- NOT the same quantity as a linear-fit "
                  f"slope rate; see docstring)")

    return result


def sliding_windows(strip_index, window_months=9, step_months=None):
    """
    Generates fixed-length, overlapping windows across the full available
    date range -- fully automatic, no need to trust detected event windows
    or know event timing in advance. Use with build_persistence_mask's
    mode="consecutive" to handle events of very different duration (a fast
    drainage vs. a slow multi-year fill) fairly -- see that function's
    docstring for why a fixed set of large periods can't do this well.

    CAUTION if strip acquisitions cluster seasonally (e.g. no optical
    stereo coverage during polar night): a fixed CALENDAR-time window does
    NOT correspond to a fixed AMOUNT of data -- some windows land mostly
    within one rich imaging season, others straddle a season boundary and
    catch only sparse edges of two. That can distort persistence in both
    modes. Consider sliding_windows_by_count() instead if this applies to
    your data -- it sidesteps the issue by defining windows in terms of
    strip count rather than calendar time.

    step_months: defaults to window_months/2 (50% overlap) if not given.
    Fractional values are fine (uses a day-based approximation internally,
    not pd.DateOffset(months=...), which only accepts whole months).
    """
    step_months = step_months or window_months / 2
    window_delta = pd.Timedelta(days=window_months * 30.44)  # avg month length
    step_delta = pd.Timedelta(days=step_months * 30.44)

    t_min, t_max = min(strip_index.keys()), max(strip_index.keys())
    windows = []
    start = t_min
    while start < t_max:
        end = start + window_delta
        windows.append((start, min(end, t_max)))
        start = start + step_delta
    return windows


def sliding_windows_by_count(strip_index, strips_per_window=20, step_strips=None):
    """
    Alternative to sliding_windows() for seasonally-clustered strip
    archives (e.g. optical stereo DEMs with no polar-night coverage):
    defines windows by a fixed NUMBER of strips rather than a fixed
    calendar duration, so every window carries comparable statistical
    power regardless of how strip density varies across the year or
    across the archive's history (typically sparser early on, denser
    more recently, and near-zero in winter).

    Windows will vary in calendar duration -- that's the point, not a
    bug: a window during a dense imaging season might span a few weeks,
    while one straddling a data-sparse period might span many months, to
    both collect the same number of strips.

    step_strips: defaults to strips_per_window/2 (50% overlap by strip
    count, not by time) if not given.
    """
    all_dates = sorted(strip_index.keys())
    n = len(all_dates)
    step_strips = int(step_strips or strips_per_window / 2)
    if step_strips < 1:
        step_strips = 1

    windows = []
    i = 0
    while i < n:
        j = min(i + strips_per_window, n) - 1
        windows.append((all_dates[i], all_dates[j]))
        if j == n - 1:
            break
        i += step_strips
    return windows


def _nanpercentile_axis0(stack, q):
    """
    np.nanpercentile(stack, q, axis=0) with the same (linear) interpolation,
    but vectorised: sort along axis 0 (NaNs sort last), then interpolate
    between the two ranks around q for each pixel's own valid count.
    np.nanpercentile loops internally and took over 9 minutes for one
    15 x 3200 x 3100 window; this takes seconds. Pixels with no valid value
    return NaN (without numpy's All-NaN warning).
    """
    s = np.sort(stack, axis=0)
    n = np.sum(~np.isnan(stack), axis=0)
    pos = (q / 100.0) * np.maximum(n - 1, 0)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
    frac = pos - lo
    v_lo = np.take_along_axis(s, lo[None], axis=0)[0]
    v_hi = np.take_along_axis(s, hi[None], axis=0)[0]
    out = v_lo + (v_hi - v_lo) * frac
    return np.where(n > 0, out, np.nan)


def _clean_mask(mask, morph_closing_px=3, morph_opening_px=1):
    """
    Opening (removes speckle) then closing (fills small gaps), as applied to
    the persistence candidates. As before, opening only runs when closing
    is enabled. Returns a uint8 0/1 mask.
    """
    mask = mask.astype("uint8")
    if morph_closing_px > 0:
        from scipy.ndimage import binary_closing, binary_opening
        struct = np.ones((3, 3), dtype=bool)
        if morph_opening_px > 0:
            mask = binary_opening(mask.astype(bool), structure=struct,
                                  iterations=morph_opening_px).astype("uint8")
        mask = binary_closing(mask.astype(bool), structure=struct,
                              iterations=morph_closing_px).astype("uint8")
    return mask


def _grid_transform(ref_coords):
    """Affine transform of the crop grid from its pixel-centre coordinates."""
    x_vals, y_vals = ref_coords
    res_x = float(x_vals[1] - x_vals[0])
    res_y = float(y_vals[1] - y_vals[0])  # typically negative (north-up)
    return rasterio.transform.from_origin(
        x_vals[0] - res_x / 2, y_vals[0] - res_y / 2, res_x, -res_y
    )


def _mask_to_polygons(mask, ref_coords, min_area_m2=0.0):
    """
    Vectorises a 0/1 mask on the crop grid. Returns (polygons, n_raw):
    the pieces at least min_area_m2 in size, and how many there were
    before that filter.
    """
    mask = mask.astype("uint8")
    shapes = rasterio.features.shapes(mask, mask=(mask == 1),
                                      transform=_grid_transform(ref_coords))
    polygons = [shape(geom) for geom, val in shapes if val == 1]
    n_raw = len(polygons)
    if min_area_m2 > 0:
        polygons = [p for p in polygons if p.area >= min_area_m2]
    return polygons, n_raw


def build_persistence_mask(strip_index, search_gdf, tile_id, pixel_area,
                            lake_gdf=None, periods=None, n_periods=6, threshold_m=None,
                            threshold_n_mad=3.0, mode="fraction",
                            min_persistence_frac=0.5, min_consecutive=2,
                            min_area_m2=400.0, morph_closing_px=3, morph_opening_px=1,
                            low_pct=5, high_pct=95, min_valid_per_period=2,
                            nodata=-9999.0, verbose=True, keep_strip_cache=False):
    """
    Semi-automated candidate boundary, following the persistence-across-
    windows idea (a pixel only counts as "lake" if it repeatedly shows
    signal across multiple INDEPENDENT time periods, not just once).

    search_gdf: a generously large search region -- NOT your current
    manual polygon. Pass something like your current polygon buffered
    outward by 2-3x its own equivalent-circle radius, so the algorithm has
    room to find a boundary larger (or smaller) than what you drew by
    hand. This function proposes a candidate for you to compare against
    your manual delineation -- it does not replace your judgment, and a
    saturated or wildly inconsistent RESULT here is itself informative
    (e.g. saturated everywhere -> search region may still be too small;
    highly inconsistent between periods -> real activity may be smaller/
    more localized than the search region, or threshold is too low).

    lake_gdf: your existing manual polygon, used ONLY to define a
    background reference region (search_gdf minus lake_gdf) for
    auto-deriving threshold_m -- see threshold_m below. Not required if
    you pass threshold_m explicitly.

    periods: explicit list of (start, end) windows. Three options, in
    order of preference:
      1. Your own detected event windows, if you trust them.
      2. sliding_windows(strip_index) -- fixed-length, overlapping,
         fully automatic, no trust in event detection required. Use
         with mode="consecutive" (see below) -- this combination is the
         recommended default when you don't yet trust detected windows,
         since a fixed small number of large periods (n_periods fallback)
         forces a single time-scale onto events that may be very
         different durations (a fast drainage vs. a slow multi-year
         fill), which a global fraction-based persistence check can't
         treat fairly -- a brief event only ever touches a small share
         of a long record's periods no matter how real it is.
      3. Leave as None -> falls back to n_periods equal chunks. Weakest
         option: equal chunks can straddle a real event and dilute its
         signal, and do nothing to distinguish event duration from noise.
    NOTE: do NOT use a single period spanning the entire record -- that
    collapses this function back into a single dz_map computation with no
    persistence check at all (equivalent to full_range_dz_map), silently
    undoing the whole point of this method.

    mode: "fraction" (default, backward-compatible) requires a pixel to
    exceed threshold_m in >= min_persistence_frac of ALL periods checked.
    Works fine with a small number of large, deliberately-chosen periods
    (e.g. your own detected windows), but penalizes short real events
    when periods span a long record.
    "consecutive" instead requires the LONGEST UNBROKEN RUN of
    overlapping periods exceeding threshold_m to be >= min_consecutive.
    Scales naturally to event duration -- a slow fill produces a long
    consecutive run, a fast drainage a short one, and isolated single-
    strip noise (which shouldn't persist across neighbouring overlapping
    windows) produces no meaningful run at all. Use this with
    sliding_windows() periods.

    threshold_m: per-pixel dz value (from refined_volume_from_zarr's raw,
    UNCORRECTED percentile-spread method -- no buffer-ring subtraction
    happens inside this function) a pixel must exceed to count as "active"
    in a given period.

    IMPORTANT: do NOT reuse noise_floor from estimate_noise_floor_from_
    buffer() here -- that's computed on the buffer-CORRECTED,
    polygon-AVERAGED timeseries, a different quantity at a different
    level of correction/averaging than the raw per-pixel dz this function
    thresholds. Using it directly is an apples-to-oranges comparison and
    can produce either zero hits (if raw noise is actually higher than
    the corrected estimate) or many spurious scattered hits (if the
    reverse) -- both were observed in initial testing.

    If threshold_m is left as None AND lake_gdf is provided, it's instead
    derived automatically, per period, from the SAME dz_map already being
    computed -- specifically from pixels in search_gdf but outside
    lake_gdf (an implicit background region, free since no extra crop is
    needed), using threshold_n_mad x a robust MAD-based spread of those
    background pixels' dz values, pooled across all periods. This keeps
    the threshold in the same raw, uncorrected units as what it's
    actually being compared against.

    min_persistence_frac: (mode="fraction" only) fraction of periods a
    pixel must exceed threshold_m in. Be cautious going below ~0.4-0.5
    with a small number of periods (e.g. 6) -- at 0.2 (1-2 of 6 periods),
    this is barely testing persistence at all.

    min_consecutive: (mode="consecutive" only) minimum run length of
    consecutive overlapping periods required. With 50%-overlap sliding
    windows, min_consecutive=2 means "showed up in two neighbouring
    windows," a fairly low bar meant to catch even brief real events
    while still excluding single-window noise; raise it if still noisy.

    min_area_m2: drops candidate polygon pieces smaller than this after
    thresholding (default 400 m^2 = 100 pixels at 2x2 m). IMPORTANT: the
    persistence check tests each pixel's behaviour over TIME but never
    requires neighbouring pixels to agree in SPACE -- a single pixel
    that crosses threshold in enough periods (from noise, a strip-edge
    artifact, or unrelated rough terrain elsewhere in a large search_gdf)
    counts exactly the same as a genuine multi-pixel lake feature. This
    is very likely why you're seeing hundreds of disjoint pieces even
    with a well-calibrated threshold -- most of them are probably single-
    or few-pixel specks, not real candidates. Raise min_area_m2 if you're
    still seeing many pieces after this filter; lower it (or set to 0)
    if you suspect it's dropping something real and small.

    morph_closing_px / morph_opening_px: applied to the raw boolean mask
    BEFORE vectorizing, at the pixel-grid level (fixes the same spatial-
    incoherence problem min_area_m2 only partly addresses). Opening
    (erode-then-dilate) first removes small isolated speckle noise;
    closing (dilate-then-erode) then fills small internal voids and
    bridges small gaps between nearby fragments into one coherent
    contour -- this is what turns a speckled pixel cloud into the kind
    of filled, contiguous shape you're actually after. Both are in
    PIXELS, not metres (multiply by pixel size, e.g. 2 m, to reason
    about metres). Set either to 0 to disable. These defaults (3 px
    closing, 1 px opening) are untuned starting guesses -- increase
    morph_closing_px if fragments still aren't merging into one piece;
    decrease it if the shape is ballooning past what the data supports.

    Returns a dict with "candidate_gdf" (GeoDataFrame, may contain more
    than one disjoint polygon -- inspect all of them, don't just take the
    largest), "persistence_map" (2D DataArray, fraction of periods each
    pixel exceeded threshold), and "threshold_m_used" (the value actually
    applied, whether passed or auto-derived).
    """
    if threshold_m is None and lake_gdf is None:
        raise ValueError(
            "Need either threshold_m explicitly, or lake_gdf to "
            "auto-derive one from the search region's own background "
            "pixels (in the same raw units this function thresholds)."
        )

    if periods is None:
        print("WARNING: periods=None uses equal time-chunks, which can "
              "straddle a real event and dilute its persistence signal "
              "-- pass your own detected event windows if you have them")
        t_min, t_max = min(strip_index.keys()), max(strip_index.keys())
        edges = pd.date_range(t_min, t_max, periods=n_periods + 1)
        periods = list(zip(edges[:-1], edges[1:]))

    if lake_gdf is not None and lake_gdf.crs != search_gdf.crs:
        lake_gdf = lake_gdf.to_crs(search_gdf.crs)

    period_dz = []
    periods_used = []  # periods that passed the coverage checks, in order

    # Cache: crop each UNIQUE strip once, reuse across every period it
    # appears in. With 50%-overlap sliding windows, most strips belong to
    # ~2 periods -- this was previously being recropped from scratch each
    # time, which is likely the main cause of slow runs (compounded by
    # search_gdf being much larger than your original lake polygon).
    all_paths_needed = set()
    period_paths = {}
    for period in periods:
        paths = strips_in_window(strip_index, period)
        period_paths[period] = paths
        all_paths_needed.update(paths)

    if verbose:
        print(f"cropping {len(all_paths_needed)} unique strips once "
              f"(vs. {sum(len(p) for p in period_paths.values())} total "
              f"period-strip references without caching)")

    strip_cache = {}
    ref_coords = None
    for zp in all_paths_needed:
        try:
            da = crop_zarr_to_polygon(zp, search_gdf, tile_id, nodata)
        except Exception as e:
            print(f"skipping {zp}: [{type(e).__name__}] {e}")
            continue
        if ref_coords is None:
            ref_coords = (da["x"].values, da["y"].values)
        vals = da.values
        if np.isnan(vals).all():
            continue  # strip doesn't reach the search region: nothing to keep
        strip_cache[zp] = vals

    # Whole-record per-pixel statistics from the same crops, for the
    # SD-based outline (sd_outline): standard deviation of elevation over
    # every strip with data at the pixel, and how many strips that was
    if strip_cache:
        import warnings
        full = np.stack(list(strip_cache.values()), axis=0)
        n_valid_map = np.sum(~np.isnan(full), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            sd_map = np.nanstd(full, axis=0)
        del full
    else:
        n_valid_map, sd_map = None, None

    for period in periods:
        paths = [p for p in period_paths[period] if p in strip_cache]
        if len(paths) < min_valid_per_period:
            print(f"period {period}: only {len(paths)} usable strips, skipping")
            continue

        stack_p = np.stack([strip_cache[p] for p in paths], axis=0)
        valid_count = np.sum(~np.isnan(stack_p), axis=0)
        has_enough = valid_count >= min_valid_per_period
        if not has_enough.any():
            print(f"period {period}: no pixel has >= {min_valid_per_period} "
                  f"overlapping strips, skipping")
            continue

        dz_low = _nanpercentile_axis0(stack_p, low_pct)
        dz_high = _nanpercentile_axis0(stack_p, high_pct)
        dz_vals = np.where(has_enough, dz_high - dz_low, np.nan)

        period_dz.append(dz_vals)
        periods_used.append(period)

    if len(period_dz) < 2:
        raise ValueError(
            f"Only {len(period_dz)} usable period(s) -- need at least "
            f"2 for a meaningful persistence check. Widen periods or lower "
            f"min_valid_per_period."
        )

    if threshold_m is None:
        # background mask: True where inside search_gdf but OUTSIDE lake_gdf
        x_vals, y_vals = ref_coords
        res_x = float(x_vals[1] - x_vals[0])
        res_y = float(y_vals[1] - y_vals[0])
        transform = rasterio.transform.from_origin(
            x_vals[0] - res_x / 2, y_vals[0] - res_y / 2, res_x, -res_y
        )
        lake_raster = rasterio.features.geometry_mask(
            lake_gdf.geometry, out_shape=(len(y_vals), len(x_vals)), transform=transform,
            invert=True  # True = inside lake_gdf
        )
        bg_vals = np.concatenate([dz[~lake_raster & ~np.isnan(dz)] for dz in period_dz])
        if bg_vals.size < 20:
            raise ValueError(
                "Too few background pixels to derive a threshold -- "
                "widen search_gdf relative to lake_gdf"
            )
        med = np.median(bg_vals)
        mad = np.median(np.abs(bg_vals - med))
        threshold_m = med + threshold_n_mad * 1.4826 * mad
        if verbose:
            print(f"auto-derived threshold_m = {threshold_m:.3f} m "
                  f"(background median {med:.3f}, MAD-scaled spread "
                  f"{1.4826 * mad:.3f}, n_mad={threshold_n_mad})")

    period_masks = [(dz >= threshold_m) for dz in period_dz]
    stack = np.stack(period_masks, axis=0)  # (n_periods, y, x), chronological order

    if mode == "consecutive":
        # longest unbroken run of True along the period axis, per pixel
        run = np.zeros_like(stack, dtype=int)
        run[0] = stack[0].astype(int)
        for i in range(1, stack.shape[0]):
            run[i] = np.where(stack[i], run[i - 1] + 1, 0)
        persistence_metric = run.max(axis=0)
        candidate_mask = (persistence_metric >= min_consecutive).astype("uint8")
        metric_name = "max consecutive periods"
    elif mode == "fraction":
        persistence_metric = stack.mean(axis=0)  # fraction of periods exceeding threshold
        candidate_mask = (persistence_metric >= min_persistence_frac).astype("uint8")
        metric_name = "fraction of periods"
    else:
        raise ValueError(f"mode must be 'fraction' or 'consecutive', got {mode!r}")

    candidate_mask = _clean_mask(candidate_mask, morph_closing_px, morph_opening_px)

    persistence_da = xr.DataArray(persistence_metric, dims=("y", "x"),
                                   coords={"x": ref_coords[0], "y": ref_coords[1]})

    polygons, n_before_area_filter = _mask_to_polygons(candidate_mask, ref_coords,
                                                       min_area_m2)

    if verbose:
        criterion = (f"consecutive >= {min_consecutive}" if mode == "consecutive"
                     else f"fraction >= {min_persistence_frac:.0%}")
        if min_area_m2 > 0:
            print(f"area filter (>= {min_area_m2:.0f} m2) dropped "
                  f"{n_before_area_filter - len(polygons)} of {n_before_area_filter} "
                  f"raw piece(s)")
        print(f"{len(period_dz)} usable periods ({metric_name}), "
              f"{len(polygons)} candidate polygon piece(s) found "
              f"at {criterion}")
        if len(polygons) == 0:
            hint = "min_consecutive" if mode == "consecutive" else "min_persistence_frac"
            print(f"no pixels met the persistence threshold -- try lowering "
                  f"threshold_n_mad or {hint}")
        elif len(polygons) > 1:
            print("MULTIPLE disjoint pieces -- inspect all of them, the "
                  "true lake may be smaller/differently shaped than "
                  "expected, or this may indicate several distinct "
                  "features rather than one lake")

    candidate_gdf = gpd.GeoDataFrame(geometry=polygons, crs=search_gdf.crs)
    return {"candidate_gdf": candidate_gdf, "persistence_map": persistence_da,
            "threshold_m_used": threshold_m,
            # per-period "active" masks (before morphology/area filtering),
            # one per entry of periods_used, for per-window outlines/envelopes
            "period_masks": stack, "periods_used": periods_used,
            "ref_coords": ref_coords,
            "sd_map": sd_map, "n_valid_map": n_valid_map,
            # cropped strips {zarr path: 2D array}, only if keep_strip_cache
            # (reused by build_epoch_differences; can be several GB)
            "strip_cache": strip_cache if keep_strip_cache else None,
            "n_strips_with_data": len(strip_cache),
            "cleaning": {"morph_closing_px": morph_closing_px,
                         "morph_opening_px": morph_opening_px,
                         "min_area_m2": min_area_m2},
            "mode": mode,
            "criterion": (min_consecutive if mode == "consecutive"
                          else min_persistence_frac)}


def persistence_outlines(result, crs, superlake_min_area_m2=0.0):
    """
    Turns a build_persistence_mask() result into polygons:

    - period_gdf: one row per outline piece per period (window), from that
      period's "active" mask (pixel dz spread above threshold_m_used),
      cleaned like the candidates (opening/closing, min_area_m2 of
      build_persistence_mask). Columns: period_index, period_start,
      period_end, area_m2, in_superlake.
    - superlake_gdf: one row, the union of the period pieces at least
      superlake_min_area_m2 in size -- the area that was clearly active in
      at least one window. Small pieces (typically noise scattered over the
      search region) are kept in period_gdf (in_superlake=False) but left
      out of the superlake. Columns: n_periods, n_pieces, area_m2.

    Either may be empty if nothing survives.
    """
    c = result["cleaning"]
    rows = []
    for k, (t0, t1) in enumerate(result["periods_used"]):
        mask = _clean_mask(result["period_masks"][k], c["morph_closing_px"],
                           c["morph_opening_px"])
        polys, _ = _mask_to_polygons(mask, result["ref_coords"], c["min_area_m2"])
        for p in polys:
            rows.append({"period_index": k,
                         "period_start": pd.Timestamp(t0).strftime("%Y-%m-%d"),
                         "period_end": pd.Timestamp(t1).strftime("%Y-%m-%d"),
                         "area_m2": float(p.area),
                         "in_superlake": bool(p.area >= superlake_min_area_m2),
                         "geometry": p})
    if rows:
        period_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    else:
        period_gdf = gpd.GeoDataFrame(
            {"period_index": [], "period_start": [], "period_end": [],
             "area_m2": [], "in_superlake": []}, geometry=[], crs=crs)
    kept = period_gdf[period_gdf["in_superlake"].astype(bool)] if len(period_gdf) else period_gdf
    if len(kept):
        geom = kept.geometry.union_all()
        superlake_gdf = gpd.GeoDataFrame(
            {"n_periods": [int(kept["period_index"].nunique())],
             "n_pieces": [len(getattr(geom, "geoms", [geom]))],
             "area_m2": [float(geom.area)]}, geometry=[geom], crs=crs)
    else:
        superlake_gdf = gpd.GeoDataFrame(
            {"n_periods": [], "n_pieces": [], "area_m2": []}, geometry=[], crs=crs)
    return period_gdf, superlake_gdf


def build_background_ring(lake_gdf, gap_frac=0.5, width_frac=1.0, tile_crs=None):
    """
    Background ring for thresholds, kept away from the lake: it starts
    gap_frac x r outside the manual polygon and is width_frac x r wide,
    where r is the polygon's equivalent-circle radius. The gap leaves room
    for lake area outside the hand-drawn outline, so it doesn't inflate the
    background statistics. gap_frac=0, width_frac=1 is the Fan et al. 2023
    ring (build_lake_buffer_ring).

    Note the search region used by build_persistence_mask is the polygon
    buffered by 1.5 r, so gap_frac + width_frac <= 1.5 keeps the ring
    inside the cropped data.
    """
    if tile_crs is not None and lake_gdf.crs != tile_crs:
        lake_gdf = lake_gdf.to_crs(tile_crs)
    if lake_gdf.crs is not None and lake_gdf.crs.is_geographic:
        raise ValueError("lake_gdf must be in a projected (metric) CRS")
    r = float(np.sqrt(lake_gdf.geometry.area.iloc[0] / np.pi))
    inner = lake_gdf.geometry.buffer(gap_frac * r) if gap_frac > 0 else lake_gdf.geometry
    outer = lake_gdf.geometry.buffer((gap_frac + width_frac) * r)
    return gpd.GeoDataFrame(geometry=outer.difference(inner), crs=lake_gdf.crs)


def _ring_mask(ring_gdf, ref_coords, shape):
    """Boolean mask of the ring on the crop grid."""
    return rasterio.features.geometry_mask(
        ring_gdf.geometry, out_shape=shape, transform=_grid_transform(ref_coords),
        invert=True)


def sd_outline(result, lake_gdf, n_mad=3.0, min_valid=3, min_area_m2=0.0,
               ring_gap_frac=0.5, ring_width_frac=1.0, min_threshold_m=None,
               sensitivity_thresholds_m=(2.0, 3.0, 4.0, 5.0), fixed_threshold_m=None):
    """
    Outline from the whole-record per-pixel standard deviation of elevation
    (the criterion used for the manual QGIS polygons), independent of any
    windowing.

    Threshold: median + n_mad x 1.4826 x MAD of the SD values inside the
    background ring (build_background_ring: starts ring_gap_frac x r outside
    the manual polygon, ring_width_frac x r wide), raised to
    min_threshold_m if that is given and higher. Pixels with fewer than
    min_valid strips are ignored. The mask is cleaned like the persistence
    candidates; pieces smaller than min_area_m2 are dropped.

    sensitivity_thresholds_m: fixed thresholds at which the outline area is
    also computed (cleaned the same way), to see how much the outline
    depends on the threshold choice.

    fixed_threshold_m: if given, the outline uses exactly this threshold
    instead of the ring-based one (the ring is still returned).

    Returns (outline_gdf, threshold_m, ring_gdf, sensitivity_df).
    sensitivity_df: threshold_m, area_m2 (one row per threshold, the
    ring-based one included).
    """
    sd = result["sd_map"]
    ref = result["ref_coords"]
    crs = result["candidate_gdf"].crs
    sd = np.where(result["n_valid_map"] >= min_valid, sd, np.nan)
    ring_gdf = build_background_ring(lake_gdf, ring_gap_frac, ring_width_frac, tile_crs=crs)
    ring_vals = sd[_ring_mask(ring_gdf, ref, sd.shape) & np.isfinite(sd)]
    if ring_vals.size < 20:
        raise ValueError("Too few valid pixels in the background ring to set "
                         "an SD threshold")
    med = np.median(ring_vals)
    mad = np.median(np.abs(ring_vals - med))
    threshold = float(med + n_mad * 1.4826 * mad)
    if min_threshold_m is not None:
        threshold = max(threshold, float(min_threshold_m))
    if fixed_threshold_m is not None:
        threshold = float(fixed_threshold_m)
    c = result["cleaning"]
    sd_filled = np.nan_to_num(sd, nan=-np.inf)
    min_piece = max(min_area_m2, c["min_area_m2"])

    def outline_at(thr):
        mask = _clean_mask(sd_filled >= thr, c["morph_closing_px"], c["morph_opening_px"])
        polys, _ = _mask_to_polygons(mask, ref, min_piece)
        return polys

    polys = outline_at(threshold)
    outline_gdf = gpd.GeoDataFrame({"area_m2": [float(p.area) for p in polys]},
                                   geometry=polys, crs=crs)
    sens = [{"threshold_m": threshold, "area_m2": float(sum(p.area for p in polys)),
             "ring_based": True}]
    for thr in sensitivity_thresholds_m or ():
        sens.append({"threshold_m": float(thr),
                     "area_m2": float(sum(p.area for p in outline_at(thr))),
                     "ring_based": False})
    return outline_gdf, threshold, ring_gdf, pd.DataFrame(sens)


def export_persistence_gpkg(gpkg_path, lake_name, tile_id, lake_gdf, search_gdf,
                            result, period_gdf, superlake_gdf,
                            sd_outline_gdf=None, ring_gdf=None,
                            epoch_outlines_gdf=None, epoch_superlake_gdf=None):
    """
    Appends one lake's outlines to a GeoPackage (EPSG:3413), one layer per
    kind, every row tagged with lake_name and tile_id:

    - manual_outline: the hand-drawn polygon used as the starting point
    - search_region: the area the persistence search ran over
    - background_ring: the gapped ring used for the SD and epoch thresholds
    - persistence_candidates: pieces passing the persistence criterion
    - period_outlines: per-window outlines (in_superlake flags the big ones)
    - superlake: union of the large per-window outlines
    - sd_outline: whole-record standard-deviation outline
    - epoch_outlines: drain/fill pieces per interval between epochs
    - superlake_epochs: union of the epoch outlines

    Empty layers for this lake are skipped. Safe to call lake by lake: an
    interrupted run keeps every lake written so far.

    The GeoPackage is built in a local temporary copy and then copied to
    gpkg_path: SQLite can't lock files on the luna network share
    ("Failed to start transaction"), so writing there directly fails.
    """
    import getpass
    import shutil
    import tempfile
    local = os.path.join(tempfile.gettempdir(),
                         f"{getpass.getuser()}_{os.path.basename(gpkg_path)}")
    if os.path.exists(gpkg_path) and os.path.getsize(gpkg_path) > 0:
        shutil.copyfile(gpkg_path, local)  # continue from what's already saved
    elif os.path.exists(local):
        os.remove(local)
    crs = search_gdf.crs
    tags = {"lake_name": lake_name, "tile_id": tile_id,
            "threshold_m": float(result["threshold_m_used"]),
            "mode": result["mode"], "criterion": float(result["criterion"])}
    layers = {
        "manual_outline": gpd.GeoDataFrame(geometry=list(lake_gdf.to_crs(crs).geometry), crs=crs),
        "search_region": gpd.GeoDataFrame(geometry=list(search_gdf.geometry), crs=crs),
        "background_ring": (gpd.GeoDataFrame(geometry=list(ring_gdf.geometry), crs=crs)
                            if ring_gdf is not None else gpd.GeoDataFrame(geometry=[], crs=crs)),
        "persistence_candidates": gpd.GeoDataFrame(
            {"area_m2": [float(g.area) for g in result["candidate_gdf"].geometry]},
            geometry=list(result["candidate_gdf"].geometry), crs=crs),
        "period_outlines": period_gdf,
        "superlake": superlake_gdf,
        "sd_outline": (sd_outline_gdf if sd_outline_gdf is not None
                       else gpd.GeoDataFrame(geometry=[], crs=crs)),
        "epoch_outlines": (epoch_outlines_gdf if epoch_outlines_gdf is not None
                           else gpd.GeoDataFrame(geometry=[], crs=crs)),
        "superlake_epochs": (epoch_superlake_gdf if epoch_superlake_gdf is not None
                             else gpd.GeoDataFrame(geometry=[], crs=crs)),
    }
    for layer, gdf in layers.items():
        if len(gdf) == 0:
            continue
        gdf = gdf.copy()
        for key, val in tags.items():
            gdf[key] = val
        gdf.to_file(local, layer=layer, driver="GPKG",
                    mode="a" if os.path.exists(local) else "w")
    shutil.copyfile(local, gpkg_path)
    print(f"GeoPackage updated: {gpkg_path} ({lake_name})")


def persistence_qc(result, lake_gdf, search_gdf, pixel_size=TILE_RESOLUTION,
                   superlake_gdf=None, sd_outline_gdf=None, sd_threshold_m=None,
                   sd_sensitivity_df=None, epoch_res=None):
    """
    One-row summary of a build_persistence_mask() result against the manual
    polygon, for comparing lakes/settings in a table. Areas in m2, in the
    search_gdf CRS (projected).

    touches_search_edge: the candidate reaches the edge of the search
    region, i.e. the search region may be too small to contain it.
    superlake_* / sd_*: the superlake (persistence_outlines) and the SD
    outline (sd_outline), if given, compared with the manual polygon.
    """
    manual = lake_gdf.to_crs(search_gdf.crs).geometry.union_all()
    search = search_gdf.geometry.union_all()
    manual_area = float(manual.area)

    def compare(gdf, prefix):
        if gdf is None:
            return {}
        if len(gdf) == 0:
            return {f"{prefix}_area_m2": 0.0, f"{prefix}_to_manual": 0.0,
                    f"{prefix}_iou_manual": 0.0, f"{prefix}_covers_manual": 0.0,
                    f"{prefix}_n_pieces": 0}
        g = gdf.geometry.union_all()
        return {f"{prefix}_area_m2": float(g.area),
                f"{prefix}_to_manual": float(g.area) / manual_area if manual_area else np.nan,
                f"{prefix}_iou_manual": float(g.intersection(manual).area / g.union(manual).area),
                f"{prefix}_covers_manual": float(g.intersection(manual).area) / manual_area,
                f"{prefix}_n_pieces": len(getattr(g, "geoms", [g]))}

    cand_gdf = result["candidate_gdf"]
    cand = cand_gdf.geometry.union_all() if len(cand_gdf) else None
    pieces = sorted((g.area for g in cand_gdf.geometry), reverse=True)
    cand_area = float(sum(pieces))
    row = {
        "manual_area_m2": manual_area,
        "n_strips_with_data": result.get("n_strips_with_data"),
        "n_periods_used": len(result["periods_used"]),
        "threshold_m_used": float(result["threshold_m_used"]),
        "mode": result["mode"],
        "criterion": result["criterion"],
        "candidate_area_m2": cand_area,
        "candidate_iou_manual": (float(cand.intersection(manual).area / cand.union(manual).area)
                                 if cand is not None else 0.0),
        "candidate_n_pieces": len(pieces),
        "candidate_touches_search_edge": (bool(cand.intersects(search.boundary.buffer(pixel_size)))
                                          if cand is not None else False),
    }
    row.update(compare(superlake_gdf, "superlake"))
    if sd_threshold_m is not None:
        row["sd_threshold_m"] = float(sd_threshold_m)
    row.update(compare(sd_outline_gdf, "sd"))
    if sd_sensitivity_df is not None:
        for _, r in sd_sensitivity_df[~sd_sensitivity_df["ring_based"]].iterrows():
            row[f"sd_area_m2_at_{r['threshold_m']:g}m"] = r["area_m2"]
    if epoch_res is not None:
        row["n_epochs"] = len(epoch_res["epochs_df"])
        row["n_intervals"] = len(epoch_res["intervals_df"])
        row.update(compare(epoch_res["superlake_gdf"], "superlake_epochs"))
    return row


def plot_persistence_diagnostics(result, lake_gdf, search_gdf, lake_name=None,
                                 output_dir=None, max_period_panels=24,
                                 period_gdf=None, superlake_gdf=None,
                                 sd_outline_gdf=None, sd_threshold_m=None,
                                 ring_gdf=None, panel_max_px=600,
                                 epoch_superlake_gdf=None):
    """
    Two figures for one build_persistence_mask() result. Every map shows
    the manual polygon (black) and the search region (grey dashed).

    1. persistence_diag_<name>.png, three panels:
       a) persistence metric on its true scale, pass threshold marked red
          on the colour bar, candidates in red;
       b) number of periods each pixel was active in (raw), with the
          superlake (union of large period outlines) in blue;
       c) whole-record per-pixel standard deviation of elevation, with the
          background ring (cyan dashed) that sets its threshold and the
          resulting SD outline (green).
    2. persistence_periods_<name>.png -- one small panel per period: raw
       active pixels (orange), period outline pieces that are part of the
       superlake (blue) and smaller ones left out (grey). Panels are
       downsampled (nearest) to panel_max_px; outlines are exact.

    Drawn with imshow, not pcolormesh: a 3000 x 3000 crop as a mesh used
    enough memory per panel to stall the run.

    Returns (diag_path, periods_path).
    """
    output_dir = output_dir or VOLUME_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    slug = lake_name or "site"
    crs = search_gdf.crs
    manual = lake_gdf.to_crs(crs)
    cand = result["candidate_gdf"]
    pmap = result["persistence_map"]
    x, y = pmap["x"].values, pmap["y"].values
    hx, hy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    extent = (x.min() - hx, x.max() + hx, y.min() - hy, y.max() + hy)  # rows run north->south
    masks = result["period_masks"]
    periods = result["periods_used"]
    n = len(periods)

    def overlay(ax):
        search_gdf.boundary.plot(ax=ax, color="grey", linestyle="--", linewidth=1)
        manual.boundary.plot(ax=ax, color="black", linewidth=1.5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")

    # -- Figure 1 --
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    if result["mode"] == "consecutive":
        vmax, label = n, "Longest run of consecutive active periods"
    else:
        vmax, label = 1.0, "Fraction of periods active"
    data = np.ma.masked_invalid(pmap.values.astype(float))
    im = ax1.imshow(data, extent=extent, origin="upper", cmap="viridis",
                    vmin=0, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046)
    cb.set_label(label)
    cb.ax.axhline(result["criterion"], color="red", linewidth=2)
    overlay(ax1)
    if len(cand):
        cand.boundary.plot(ax=ax1, color="red", linewidth=1)
    ax1.set_title(f"a) Persistence ({result['mode']}, pass >= {result['criterion']}, "
                  f"{n} periods)\nred = persistence candidates")

    count = masks.sum(axis=0).astype(float)
    count[count == 0] = np.nan
    im2 = ax2.imshow(np.ma.masked_invalid(count), extent=extent, origin="upper",
                     cmap="magma_r", vmin=1, vmax=max(n, 1), interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046)
    cb2.set_label("Number of periods active (raw)")
    overlay(ax2)
    if superlake_gdf is not None and len(superlake_gdf):
        superlake_gdf.boundary.plot(ax=ax2, color="blue", linewidth=1.2)
    ax2.set_title(f"b) Activity count, threshold {result['threshold_m_used']:.2f} m"
                  f"\nblue = superlake (union of large period outlines)")

    sd = result.get("sd_map")
    if sd is not None:
        sd_plot = np.where(result["n_valid_map"] >= 3, sd, np.nan)
        finite = sd_plot[np.isfinite(sd_plot)]
        vmax_sd = float(np.percentile(finite, 98)) if finite.size else 1.0
        im3 = ax3.imshow(np.ma.masked_invalid(sd_plot), extent=extent, origin="upper",
                         cmap="viridis", vmin=0, vmax=vmax_sd, interpolation="nearest")
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046)
        cb3.set_label("Standard deviation of elevation, whole record (m)")
        if sd_threshold_m is not None:
            cb3.ax.axhline(min(sd_threshold_m, vmax_sd), color="lime", linewidth=2)
    overlay(ax3)
    if ring_gdf is not None:
        ring_gdf.boundary.plot(ax=ax3, color="cyan", linestyle="--", linewidth=1)
    if sd_outline_gdf is not None and len(sd_outline_gdf):
        sd_outline_gdf.boundary.plot(ax=ax3, color="lime", linewidth=1.2)
    if epoch_superlake_gdf is not None and len(epoch_superlake_gdf):
        epoch_superlake_gdf.boundary.plot(ax=ax3, color="magenta", linewidth=1.2)
    thr_txt = f"{sd_threshold_m:.2f} m" if sd_threshold_m is not None else "n/a"
    ax3.set_title(f"c) Whole-record SD, threshold {thr_txt} (from cyan ring)"
                  f"\ngreen = SD outline, magenta = epoch superlake")
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x (m) - EPSG:3413"); ax.set_ylabel("y (m) - EPSG:3413")
    fig.suptitle(f"Lake outline diagnostics -- {slug}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    diag_path = os.path.join(output_dir, f"persistence_diag_{slug}.png")
    fig.savefig(diag_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # -- Figure 2: one panel per period --
    step = max(1, int(np.ceil(max(masks.shape[1:]) / panel_max_px)))
    shown = min(n, max_period_panels)
    ncols = 6
    nrows = int(np.ceil(shown / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= shown:
            ax.axis("off"); continue
        small = masks[k][::step, ::step]
        ax.imshow(np.where(small, 1.0, np.nan), extent=extent, origin="upper",
                  cmap="autumn", vmin=0, vmax=1, interpolation="nearest")
        overlay(ax)
        if period_gdf is not None and len(period_gdf):
            sel = period_gdf[period_gdf["period_index"] == k]
            big = sel[sel["in_superlake"].astype(bool)]
            small_p = sel[~sel["in_superlake"].astype(bool)]
            if len(small_p):
                small_p.boundary.plot(ax=ax, color="grey", linewidth=0.4)
            if len(big):
                big.boundary.plot(ax=ax, color="blue", linewidth=1)
        t0, t1 = periods[k]
        ax.set_title(f"{pd.Timestamp(t0):%Y-%m-%d} to {pd.Timestamp(t1):%Y-%m-%d}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    extra = f" (first {shown} of {n})" if n > shown else ""
    fig.suptitle(f"Active pixels per period{extra} -- {slug}\n"
                 f"orange = raw active pixels, blue = outline kept in superlake, "
                 f"grey = small outline left out",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    periods_path = os.path.join(output_dir, f"persistence_periods_{slug}.png")
    fig.savefig(periods_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots saved: {diag_path}, {periods_path}")
    return diag_path, periods_path


# some stacks (12_38_2_2) have a doubled "processed_processed_" prefix
_ZARR_PAIR_RX = re.compile(r"^(?:processed_)+(?P<pair>.+?)(?:_dt)?\.zarr$")


def _pair_key(name):
    """
    Matching key between catalogue pairnames and zarr folder names: older
    zarr folders keep only <sensor>_<date>_<first catalogue ID>, e.g.
    processed_W1W1_20180223_10200100717B9C00_dt.zarr for pairname
    W1W1_20180223_10200100717B9C00_<second ID>. Rebuilt (_rs) stacks keep
    the full pairname, so match on that whenever the zarr name has it.
    """
    return "_".join(name.split("_")[:3])


def _pair_in(pair, keys):
    """True if catalogue pairname `pair` is in `keys` (full or short names)."""
    return pair in keys or _pair_key(pair) in keys


def coverage_gap_report(lake_gdf, tile_id, strip_index, ts_diag, subfolder=None,
                        time_range="2009-01-01/2026-12-31", min_gap_days=365):
    """
    Why does the lake's elevation time series have long gaps?

    For every ArcticDEM strip whose catalogue footprint (PGC STAC) touches
    the lake polygon, records how far it got through the data chain:
      1. in_catalogue  -- listed by PGC (always True here)
      2. raw_in_archive -- the strip .tar.gz is in ARCHIVE_DIR
      3. coreg_in_archive -- the coregistered GeoTIFF matching this tile's
         zarr reference (cs2 -> COREG_PARAMS['altim'], mosaic ->
         COREG_PARAMS['mosaic']) is in ARCHIVE_DIR
      4. in_zarr -- the strip is in the tile's zarr folder
      5. zarr_data_over_lake -- it has valid pixels over the lake and its
         background ring (status "kept" in ts_diag, the diagnostics from
         build_buffer_corrected_timeseries(..., return_diagnostics=True))

    Gaps are intervals of at least min_gap_days between consecutive
    "kept" strips (plus before the first / after the last). For each gap
    the counts at every stage are given, and the first stage that drops
    to zero is the likely reason: "no strips acquired", "not downloaded",
    "not coregistered", "coregistered but not in zarr stack", or "in zarr
    but no valid data over the lake".

    Returns (strips_df, gaps_df).
    """
    from elevation_utils import search_arcticdem_strips, find_strip_file

    if subfolder is None:
        subfolder = select_preferred_subfolder(tile_id)
    ref = "mosaic" if "mosaic" in subfolder else "altim"  # cs2 -> altim

    lake4326 = lake_gdf.to_crs("EPSG:4326")
    geom = lake4326.geometry.union_all()
    w, s_, e, n = geom.bounds
    items_gdf, _ = search_arcticdem_strips((w, s_, e, n), time_range)
    if len(items_gdf) == 0:
        return pd.DataFrame(), pd.DataFrame()
    items_gdf = items_gdf[items_gdf.intersects(geom)].copy()

    zarr_pairs = {}
    for paths in strip_index.values():
        for p in paths:
            m = _ZARR_PAIR_RX.match(os.path.basename(p))
            if m:
                zarr_pairs[m["pair"]] = p
    kept_pairs = set()
    for _, r in ts_diag.iterrows():
        m = _ZARR_PAIR_RX.match(os.path.basename(r["path"]))
        if m and r["status"] == "kept":
            kept_pairs.add(m["pair"])

    rows = []
    for _, it in items_gdf.iterrows():
        pair, geocell, sid = it["pgc:pairname"], it["pgc:geocell"], it["stac_id"]
        rows.append({
            "date": pd.Timestamp(it["datetime"]).tz_localize(None).normalize(),
            "pairname": pair, "stac_id": sid, "geocell": geocell,
            "is_xtrack": bool(it.get("pgc:is_xtrack", False)),
            "cloud_fraction": it.get("pgc:cloud_area_percent", np.nan),
            "raw_in_archive": find_strip_file(ARCHIVE_DIR, geocell, sid, "none") is not None,
            "coreg_in_archive": find_strip_file(ARCHIVE_DIR, geocell, sid, ref) is not None,
            "in_zarr": _pair_in(pair, zarr_pairs),
            "zarr_data_over_lake": _pair_in(pair, kept_pairs),
        })
    strips_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    kept_dates = sorted(strips_df.loc[strips_df["zarr_data_over_lake"], "date"].unique())
    t_lo = pd.Timestamp(time_range.split("/")[0])
    t_hi = pd.Timestamp(time_range.split("/")[1])
    edges = [t_lo] + list(kept_dates) + [t_hi]
    stages = ["raw_in_archive", "coreg_in_archive", "in_zarr", "zarr_data_over_lake"]
    reasons = {"n_catalogue": "no strips acquired",
               "raw_in_archive": "not downloaded to archive",
               "coreg_in_archive": "not coregistered",
               "in_zarr": "coregistered but not in zarr stack",
               "zarr_data_over_lake": "in zarr but no valid data over lake"}
    gaps = []
    for a, b in zip(edges[:-1], edges[1:]):
        if (b - a).days < min_gap_days:
            continue
        inside = strips_df[(strips_df["date"] > a) & (strips_df["date"] < b)]
        counts = {"n_catalogue": len(inside), **{k: int(inside[k].sum()) for k in stages}}
        reason = next((reasons[k] for k in ["n_catalogue"] + stages if counts[k] == 0),
                      "data present (check filtering)")
        gaps.append({"gap_start": a.date(), "gap_end": b.date(),
                     "gap_days": (b - a).days, **counts, "likely_reason": reason})
    return strips_df, pd.DataFrame(gaps)


def build_epoch_differences(strip_cache, strip_index, lake_gdf, ref_coords, crs,
                            epoch_days=30, ring_gap_frac=0.5, ring_width_frac=1.0,
                            n_mad=3.0, min_abs_dz_m=None, min_ring_frac=0.2,
                            min_lake_frac=0.5,
                            min_piece_m2=0.0, cleaning=None, pixel_area=None,
                            verbose=True):
    """
    Option B: outlines from differences between consecutive acquisition
    epochs, which adapts to how dense or sparse each lake's record is.

    1. Strip offsets: each cropped strip has its median over the background
       ring (build_background_ring) subtracted, so heights are relative to
       the local background (the same idea as the buffer-corrected time
       series). Strips covering less than min_ring_frac of the ring are
       skipped, since their offset can't be estimated, and so are strips
       covering less than min_lake_frac of the manual polygon: partial
       "sliver" strips at the edge of the search region carry no lake
       information and their offsets are unreliable, which showed up as
       uniform multi-metre steps and strip-edge seams.
    2. Epochs: strips are grouped in date order; an epoch holds every strip
       within epoch_days of its first strip. Per pixel, the epoch value is
       the median of its strips. Bursts of acquisitions (e.g. six strips in
       one week) become one epoch instead of several noisy intervals.
    3. Intervals: for each epoch i >= 1 and each pixel it observes,
       dz = epoch[i] - (the most recent EARLIER epoch that observed that
       pixel). Comparing against the last observation per pixel, not just
       epoch i-1, keeps a partial-coverage epoch in between from hiding a
       change in the pixels it doesn't cover. The "from" date can therefore
       differ between pixels of one interval; intervals_df lists the range.
    4. Threshold per interval: n_mad x robust SD (1.4826 x MAD) of dz over
       the background ring, raised to min_abs_dz_m if given. "drain" pixels:
       dz - ring median <= -threshold; "fill" pixels: >= +threshold.
    5. Outlines: each drain/fill mask is cleaned like the persistence
       candidates (cleaning dict) and pieces under min_piece_m2 are dropped.
       The epoch superlake is the union of all pieces, both signs.

    strip_cache: {zarr path: 2D array on the crop grid}, from
    build_persistence_mask(..., keep_strip_cache=True).

    Returns a dict:
      epochs_df    -- epoch_index, start, end, n_strips
      intervals_df -- interval_index, epoch dates, sigma_m, threshold_m,
                      valid_frac (share of the search grid with dz),
                      drain_area_m2, fill_area_m2
      outlines_gdf -- one row per piece: interval_index, interval_start,
                      interval_end, sign ("drain"/"fill"), area_m2,
                      mean_dz_m, volume_m3 (sum of dz x pixel area)
      superlake_gdf -- union of all pieces (one row)
      dz_maps      -- list of 2D dz arrays, one per interval
      ring_gdf     -- the background ring used
    """
    cleaning = cleaning or {"morph_closing_px": 3, "morph_opening_px": 1,
                            "min_area_m2": 400.0}
    shape = next(iter(strip_cache.values())).shape
    if pixel_area is None:
        pixel_area = abs((ref_coords[0][1] - ref_coords[0][0]) *
                         (ref_coords[1][1] - ref_coords[1][0]))
    ring_gdf = build_background_ring(lake_gdf, ring_gap_frac, ring_width_frac, tile_crs=crs)
    ring = _ring_mask(ring_gdf, ref_coords, shape)
    n_ring = int(ring.sum())
    lake_mask = _ring_mask(lake_gdf.to_crs(crs), ref_coords, shape)
    n_lake = int(lake_mask.sum())

    # 1. per-strip offset removal
    path_date = {p: d for d, paths in strip_index.items() for p in paths}
    corrected, skipped_ring, skipped_lake = [], 0, 0
    for path, arr in strip_cache.items():
        if np.isfinite(arr[lake_mask]).sum() < max(1, min_lake_frac * n_lake):
            skipped_lake += 1
            continue
        ring_vals = arr[ring]
        ok = np.isfinite(ring_vals)
        if ok.sum() < max(1, min_ring_frac * n_ring):
            skipped_ring += 1
            continue
        offset = float(np.median(ring_vals[ok]))
        corrected.append((path_date[path], (arr - offset).astype("float32")))
    corrected.sort(key=lambda t: t[0])
    if verbose:
        print(f"epoch differences: {len(corrected)} strips usable; skipped "
              f"{skipped_lake} covering < {min_lake_frac:.0%} of the lake, "
              f"{skipped_ring} covering < {min_ring_frac:.0%} of the ring")

    # 2. epochs
    epochs, current = [], []
    for date, arr in corrected:
        if current and (date - current[0][0]).days > epoch_days:
            epochs.append(current)
            current = []
        current.append((date, arr))
    if current:
        epochs.append(current)
    epoch_maps, epoch_rows = [], []
    for k, ep in enumerate(epochs):
        stack = np.stack([a for _, a in ep], axis=0)
        epoch_maps.append(stack[0] if len(ep) == 1 else _nanpercentile_axis0(stack, 50))
        epoch_rows.append({"epoch_index": k, "start": ep[0][0], "end": ep[-1][0],
                           "n_strips": len(ep)})
    epochs_df = pd.DataFrame(epoch_rows)
    if verbose:
        print(f"{len(epochs)} epochs (strips within {epoch_days} days grouped), "
              f"{max(len(epochs) - 1, 0)} intervals")

    # 3-5. intervals, each epoch against the last earlier observation per pixel
    interval_rows, piece_rows, dz_maps = [], [], []
    min_piece = max(min_piece_m2, cleaning["min_area_m2"])
    last_val = epoch_maps[0].copy() if epoch_maps else None
    last_idx = np.where(np.isfinite(last_val), 0, -1) if epoch_maps else None
    for e in range(1, len(epoch_maps)):
        cur = epoch_maps[e]
        dz = cur - last_val  # NaN where either is missing
        from_idx = np.where(np.isfinite(dz), last_idx, -1)
        dz_maps.append(dz)
        valid = np.isfinite(dz)
        used_from = sorted(set(from_idx[valid].tolist())) if valid.any() else []
        t1 = epoch_rows[e]
        t0_first = epoch_rows[used_from[0]] if used_from else epoch_rows[e - 1]
        t0_last = epoch_rows[used_from[-1]] if used_from else epoch_rows[e - 1]
        k = e - 1
        bg = dz[ring & valid]
        row = {"interval_index": k,
               "from_epoch_start": t0_first["start"].date(), "from_epoch_end": t0_last["end"].date(),
               "to_epoch_start": t1["start"].date(), "to_epoch_end": t1["end"].date(),
               "n_from_epochs": len(used_from),
               "valid_frac": float(valid.mean()),
               "sigma_m": np.nan, "threshold_m": np.nan,
               "drain_area_m2": 0.0, "fill_area_m2": 0.0}
        # update the per-pixel last observation with this epoch
        seen = np.isfinite(cur)
        last_val = np.where(seen, cur, last_val)
        last_idx = np.where(seen, e, last_idx)
        if bg.size < 20:
            interval_rows.append(row)  # too little background to set a threshold
            continue
        med = float(np.median(bg))
        sigma = float(1.4826 * np.median(np.abs(bg - med)))
        thr = n_mad * sigma
        if min_abs_dz_m is not None:
            thr = max(thr, float(min_abs_dz_m))
        row.update(sigma_m=sigma, threshold_m=thr)
        rel = np.nan_to_num(dz - med, nan=0.0)
        for sign, mask in (("drain", rel <= -thr), ("fill", rel >= thr)):
            mask = _clean_mask(mask, cleaning["morph_closing_px"], cleaning["morph_opening_px"])
            polys, _ = _mask_to_polygons(mask, ref_coords, min_piece)
            for p in polys:
                pm = rasterio.features.geometry_mask(
                    [p], out_shape=shape, transform=_grid_transform(ref_coords), invert=True)
                vals = dz[pm & valid]
                froms = from_idx[pm & valid]
                f0 = epoch_rows[int(froms.min())] if froms.size else t0_first
                f1 = epoch_rows[int(froms.max())] if froms.size else t0_last
                piece_rows.append({
                    "interval_index": k,
                    "interval_start": f0["end"].strftime("%Y-%m-%d"),
                    "interval_start_latest": f1["end"].strftime("%Y-%m-%d"),
                    "interval_end": t1["start"].strftime("%Y-%m-%d"),
                    "sign": sign, "area_m2": float(p.area),
                    "mean_dz_m": float(vals.mean()) if vals.size else np.nan,
                    "volume_m3": float(vals.sum() * pixel_area) if vals.size else np.nan,
                    "geometry": p})
                row[f"{sign}_area_m2"] += float(p.area)
        interval_rows.append(row)

    intervals_df = pd.DataFrame(interval_rows)
    if piece_rows:
        outlines_gdf = gpd.GeoDataFrame(piece_rows, geometry="geometry", crs=crs)
        geom = outlines_gdf.geometry.union_all()
        superlake_gdf = gpd.GeoDataFrame(
            {"n_intervals": [int(outlines_gdf["interval_index"].nunique())],
             "n_pieces": [len(getattr(geom, "geoms", [geom]))],
             "area_m2": [float(geom.area)]}, geometry=[geom], crs=crs)
    else:
        outlines_gdf = gpd.GeoDataFrame(
            {"interval_index": [], "interval_start": [], "interval_start_latest": [],
             "interval_end": [], "sign": [],
             "area_m2": [], "mean_dz_m": [], "volume_m3": []}, geometry=[], crs=crs)
        superlake_gdf = gpd.GeoDataFrame(
            {"n_intervals": [], "n_pieces": [], "area_m2": []}, geometry=[], crs=crs)
    if verbose:
        n_active = int((intervals_df[["drain_area_m2", "fill_area_m2"]].sum(axis=1) > 0).sum()) if len(intervals_df) else 0
        print(f"{n_active} interval(s) with drain/fill outlines, "
              f"epoch superlake {superlake_gdf['area_m2'].sum() / 1e6:.2f} km2")
    return {"epochs_df": epochs_df, "intervals_df": intervals_df,
            "outlines_gdf": outlines_gdf, "superlake_gdf": superlake_gdf,
            "dz_maps": dz_maps, "ring_gdf": ring_gdf}


def plot_epoch_differences(epoch_res, lake_gdf, search_gdf, ref_coords, lake_name=None,
                           output_dir=None, max_panels=30, panel_max_px=500):
    """
    One small panel per interval between consecutive epochs: dz (blue =
    lowering, red = rising, scale +/- 3 x that interval's threshold),
    drain outlines (blue), fill outlines (red), manual polygon (black),
    background ring (cyan dashed). Saved as epochs_<name>.png. Panels are
    downsampled (nearest) to panel_max_px; outlines are exact.
    """
    output_dir = output_dir or VOLUME_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    slug = lake_name or "site"
    crs = search_gdf.crs
    manual = lake_gdf.to_crs(crs)
    x, y = ref_coords
    hx, hy = abs(x[1] - x[0]) / 2, abs(y[1] - y[0]) / 2
    extent = (x.min() - hx, x.max() + hx, y.min() - hy, y.max() + hy)
    dz_maps = epoch_res["dz_maps"]
    iv = epoch_res["intervals_df"]
    ol = epoch_res["outlines_gdf"]
    n = len(dz_maps)
    if n == 0:
        print("no intervals to plot")
        return None
    shown = min(n, max_panels)
    step = max(1, int(np.ceil(max(dz_maps[0].shape) / panel_max_px)))
    ncols = 6
    nrows = int(np.ceil(shown / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= shown:
            ax.axis("off"); continue
        row = iv.iloc[k]
        lim = 3 * row["threshold_m"] if np.isfinite(row["threshold_m"]) else 5.0
        ax.imshow(np.ma.masked_invalid(dz_maps[k][::step, ::step]), extent=extent,
                  origin="upper", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
        search_gdf.boundary.plot(ax=ax, color="grey", linestyle="--", linewidth=0.8)
        epoch_res["ring_gdf"].boundary.plot(ax=ax, color="cyan", linestyle="--", linewidth=0.6)
        manual.boundary.plot(ax=ax, color="black", linewidth=1.2)
        if len(ol):
            sel = ol[ol["interval_index"] == k]
            for sign, col in (("drain", "blue"), ("fill", "red")):
                part = sel[sel["sign"] == sign]
                if len(part):
                    part.boundary.plot(ax=ax, color=col, linewidth=1)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        thr_txt = f"thr {row['threshold_m']:.1f} m" if np.isfinite(row["threshold_m"]) else "no thr"
        ax.set_title(f"{row['from_epoch_end']} -> {row['to_epoch_start']}\n"
                     f"{thr_txt}, data {row['valid_frac']:.0%}", fontsize=7)
    extra = f" (first {shown} of {n})" if n > shown else ""
    fig.suptitle(f"Epoch-to-epoch elevation change{extra} -- {slug}\n"
                 f"blue/red shading = lowering/rising (scale +/-3x threshold); "
                 f"outlines: blue = drain, red = fill", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"epochs_{slug}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")
    return path


def _append_gpkg_layer(gpkg_path, layer, gdf):
    """Append gdf to a layer of gpkg_path, via a local copy (luna can't host SQLite writes)."""
    import getpass
    import shutil
    import tempfile
    if gdf is None or len(gdf) == 0:
        return
    local = os.path.join(tempfile.gettempdir(),
                         f"{getpass.getuser()}_{os.path.basename(gpkg_path)}")
    if os.path.exists(gpkg_path) and os.path.getsize(gpkg_path) > 0:
        shutil.copyfile(gpkg_path, local)
    elif os.path.exists(local):
        os.remove(local)
    gdf.to_file(local, layer=layer, driver="GPKG",
                mode="a" if os.path.exists(local) else "w")
    shutil.copyfile(local, gpkg_path)


def contour_timeseries(strip_cache, strip_index, contour_gdf, ring_gdf, ref_coords,
                       min_ring_frac=0.2, min_contour_frac=0.0):
    """
    Background-corrected elevation time series over any contour (manual
    polygon, SD outline, ...), from the already-cropped strips.

    Per strip: contour_mean = mean elevation over the contour's pixels with
    data; ring_median = median over the background ring minus the contour
    (so a contour reaching into the ring never counts as background);
    corrected = contour_mean - ring_median. Strips covering less than
    min_ring_frac of the ring or min_contour_frac of the contour are left
    out.

    Returns a DataFrame: date, contour_mean, ring_median, corrected,
    contour_coverage, n_contour_px.
    """
    shape = next(iter(strip_cache.values())).shape
    crs = contour_gdf.crs
    cmask = _ring_mask(contour_gdf, ref_coords, shape)
    rmask = _ring_mask(ring_gdf.to_crs(crs), ref_coords, shape) & ~cmask
    n_c, n_r = int(cmask.sum()), int(rmask.sum())
    path_date = {p: d for d, paths in strip_index.items() for p in paths}
    rows = []
    for path, arr in strip_cache.items():
        cv = arr[cmask]; rv = arr[rmask]
        cok = np.isfinite(cv); rok = np.isfinite(rv)
        if n_c == 0 or cok.sum() == 0 or cok.sum() < min_contour_frac * n_c:
            continue
        if rok.sum() < max(1, min_ring_frac * n_r):
            continue
        cm = float(cv[cok].mean()); rm = float(np.median(rv[rok]))
        rows.append({"date": path_date[path], "contour_mean": cm, "ring_median": rm,
                     "corrected": cm - rm, "contour_coverage": float(cok.mean()),
                     "n_contour_px": int(cok.sum())})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["date", "contour_mean", "ring_median", "corrected", "contour_coverage", "n_contour_px"])


def contour_volumes(epoch_res, contour_gdf, ref_coords, pixel_area=TILE_RESOLUTION ** 2):
    """
    Volume change per epoch interval inside a contour, from the epoch
    dz maps (build_epoch_differences, offset-corrected, each pixel against
    its last earlier observation):

    - mean_dz_m: mean dz over contour pixels with data
    - volume_measured_m3: sum of dz x pixel area over those pixels
    - volume_m3: mean_dz_m x full contour area (assumes uncovered pixels
      changed like covered ones; check coverage)
    - coverage: share of the contour with dz in that interval
    - cumulative_volume_m3: running sum of volume_m3

    Returns a DataFrame, one row per interval.
    """
    dz_maps = epoch_res["dz_maps"]
    if not dz_maps:
        return pd.DataFrame()
    cmask = _ring_mask(contour_gdf, ref_coords, dz_maps[0].shape)
    area = float(contour_gdf.geometry.union_all().area)
    rows = []
    for k, dz in enumerate(dz_maps):
        vals = dz[cmask]
        ok = np.isfinite(vals)
        iv = epoch_res["intervals_df"].iloc[k]
        mean_dz = float(vals[ok].mean()) if ok.any() else np.nan
        rows.append({"interval_index": k,
                     "from_epoch_end": iv["from_epoch_end"], "to_epoch_start": iv["to_epoch_start"],
                     "coverage": float(ok.mean()) if vals.size else 0.0,
                     "mean_dz_m": mean_dz,
                     "volume_measured_m3": float(vals[ok].sum() * pixel_area) if ok.any() else 0.0,
                     "volume_m3": mean_dz * area if ok.any() else 0.0,
                     "contour_area_m2": area})
    df = pd.DataFrame(rows)
    df["cumulative_volume_m3"] = df["volume_m3"].fillna(0).cumsum()
    return df


def plot_contour_comparison(ts_by_contour, vol_by_contour, lake_name=None, output_dir=None,
                            min_coverage=0.5):
    """
    Two panels per lake for several contours (manual polygon, SD outlines
    at different thresholds, ...):
    top -- background-corrected mean elevation time series (no outlier
    removal); bottom -- cumulative volume change from the epoch
    differences, plotted at each interval's end date (intervals with
    contour coverage below min_coverage drawn as open markers).
    Saved as contours_<name>.png.
    """
    output_dir = output_dir or VOLUME_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    slug = lake_name or "site"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (name, ts) in enumerate(ts_by_contour.items()):
        if ts is None or len(ts) == 0:
            continue
        c = colors[k % len(colors)]
        ax1.plot(pd.to_datetime(ts["date"]), ts["corrected"], "o-", color=c, ms=4, lw=1, label=name)
        vol = vol_by_contour.get(name)
        if vol is not None and len(vol):
            t = pd.to_datetime(vol["to_epoch_start"])
            cum = vol["cumulative_volume_m3"] / 1e6
            ax2.plot(t, cum, "-", color=c, lw=1.2, label=name)
            good = vol["coverage"] >= min_coverage
            ax2.plot(t[good], cum[good], "o", color=c, ms=5)
            ax2.plot(t[~good], cum[~good], "o", mfc="none", color=c, ms=5)
    ax1.set_ylabel("Contour mean - background ring median (m)")
    ax1.set_title(f"Background-corrected elevation, per contour -- {slug}", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5); ax1.legend(fontsize=8)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_ylabel("Cumulative volume change (10$^6$ m$^3$)")
    ax2.set_title(f"Cumulative volume change from epoch differences "
                  f"(open markers: contour coverage < {min_coverage:.0%})", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.5); ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = os.path.join(output_dir, f"contours_{slug}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")
    return path


# BedMachine Greenland (Morlighem et al.) ICE THICKNESS, in metres on the
# EPSG:3413 150 m grid: either the netCDF (BedMachineGreenland-v*.nc, which
# has a "thickness" variable) or a single-band thickness GeoTIFF. NOT the
# bed GeoTIFF: bed is geoid-referenced, ArcticDEM is ellipsoidal, and the
# difference (tens of metres) would swamp a 10 m thickness estimate.
BEDMACHINE_PATH = ("/home/moralpom/luna/CPOM/archive/RESOURCES/surface_discrimination_masks/"
                   "greenland/bedmachine_v6/BedMachineGreenland-v6.nc")


def lake_ice_thickness(lake_gdf, bedmachine_path=None):
    """
    Rough ice thickness at a lake from BedMachine Greenland: the mean
    "thickness" over the grid cells whose centres fall inside the polygon
    (or the nearest cell to the centroid if none do), plus the value at
    the centroid. Rounded to 10 m -- BedMachine's own uncertainty is much
    larger than that in most places.

    Returns {"ice_thickness_m", "ice_thickness_centroid_m", "bed_error_m"}
    (NaN if no file). bed_error_m is BedMachine's mean "errbed" over the
    polygon (netCDF only) -- the thickness uncertainty is of that order.
    """
    path = bedmachine_path or BEDMACHINE_PATH
    if not path or not os.path.exists(path):
        return {"ice_thickness_m": np.nan, "ice_thickness_centroid_m": np.nan,
                "bed_error_m": np.nan}
    lake = lake_gdf.to_crs("EPSG:3413")
    if path.lower().endswith((".tif", ".tiff")):
        import rasterio.mask
        with rasterio.open(path) as src:
            geom = lake.to_crs(src.crs).geometry.union_all()
            data, _ = rasterio.mask.mask(src, [geom], crop=True, all_touched=False,
                                         filled=False)
            vals = data[0].compressed().astype(float)
            if src.nodata is not None:
                vals = vals[vals != src.nodata]
            c = geom.centroid
            centroid_val = float(next(src.sample([(c.x, c.y)]))[0])
        vals = vals[np.isfinite(vals)]
        mean_val = float(vals.mean()) if vals.size else centroid_val
        return {"ice_thickness_m": round(mean_val, -1),
                "ice_thickness_centroid_m": round(centroid_val, -1), "bed_error_m": np.nan}
    minx, miny, maxx, maxy = lake.total_bounds
    pad = 500.0
    with xr.open_dataset(path) as ds:
        th = ds["thickness"]
        ys_desc = bool(ds["y"][0] > ds["y"][-1])
        ysl = slice(maxy + pad, miny - pad) if ys_desc else slice(miny - pad, maxy + pad)
        sub = th.sel(x=slice(minx - pad, maxx + pad), y=ysl).load()
        err = (ds["errbed"].sel(x=slice(minx - pad, maxx + pad), y=ysl).load()
               if "errbed" in ds else None)
        c = lake.geometry.union_all().centroid
        centroid_val = float(th.sel(x=c.x, y=c.y, method="nearest"))
    xx, yy = np.meshgrid(sub["x"].values, sub["y"].values)
    pts = gpd.GeoSeries(gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:3413")
    inside = pts.within(lake.geometry.union_all()).values.reshape(xx.shape)
    vals = sub.values[inside].astype(float)
    vals = vals[np.isfinite(vals)]
    mean_val = float(vals.mean()) if vals.size else centroid_val
    err_val = np.nan
    if err is not None:
        ev = err.values[inside].astype(float)
        ev = ev[np.isfinite(ev)]
        err_val = round(float(ev.mean()), -1) if ev.size else np.nan
    return {"ice_thickness_m": round(mean_val, -1),
            "ice_thickness_centroid_m": round(centroid_val, -1), "bed_error_m": err_val}


def add_d_over_h(row, ice_thickness_m):
    """
    Adds equivalent diameters d = 2 sqrt(A / pi) and d/H to a QC row, for
    every outline area present (manual polygon, SD outline, superlakes).
    d/H well above ~1 means the lake is wide compared with the ice above it
    (surface signal close to the basal change); around or below 1, the
    surface signal is broader and damped compared with the lake itself.
    """
    out = dict(row)
    out["ice_thickness_m"] = ice_thickness_m
    for key, label in (("manual_area_m2", "manual"), ("sd_area_m2", "sd"),
                       ("superlake_area_m2", "superlake"),
                       ("superlake_epochs_area_m2", "superlake_epochs")):
        a = row.get(key)
        if a is None or not np.isfinite(a):
            continue
        d = 2.0 * np.sqrt(a / np.pi)
        out[f"d_{label}_m"] = round(d, -1)
        out[f"d_over_H_{label}"] = (round(d / ice_thickness_m, 2)
                                   if ice_thickness_m and np.isfinite(ice_thickness_m) else np.nan)
    return out


def plot_lake_vs_background(ts_df, lake_name=None, output_path=None,
                            ring_desc="ring 0.5-1.5 r outside the manual polygon",
                            errorbars=True):
    """
    Absolute elevation of the lake (mean over the polygon) and of the
    background ring, per strip, in one plot -- no trend line, no outlier
    removal. Error bars: +/- 1 spatial standard deviation of each area's
    pixels at that date (a spread across the area, not a formal measurement
    uncertainty).

    ts_df: build_buffer_corrected_timeseries() output (date, lake_mean,
    buffer_mean, lake_std, buffer_std).
    """
    df = ts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    fig, ax = plt.subplots(figsize=(14, 7))
    for col, std, label, color in (
            ("buffer_mean", "buffer_std", f"Background ({ring_desc})", "grey"),
            ("lake_mean", "lake_std", "Lake (manual polygon mean)", "#e8432a")):
        ax.plot(df["date"], df[col], "-", color=color, lw=2, zorder=2)
        if errorbars and std in df:
            ax.errorbar(df["date"], df[col], yerr=df[std], fmt="o", ms=8, color=color,
                        mec="white", ecolor=color, elinewidth=1, capsize=3, alpha=0.9,
                        label=label, zorder=3)
        else:
            ax.plot(df["date"], df[col], "o", ms=8, color=color, mec="white", label=label,
                    zorder=3)
    ax.set_ylabel("Elevation (m)", fontsize=13)
    ax.set_xlabel("Date", fontsize=13)
    ax.grid(True, ls=":", alpha=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="best", fontsize=11)
    name = lake_name or "site"
    ax.set_title(f"Lake vs. background elevation history -- {name}\n"
                 f"{len(df)} strips, error bars = +/-1 spatial SD", fontsize=15,
                 fontweight="bold")
    plt.tight_layout()
    if output_path is None:
        os.makedirs(VOLUME_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(VOLUME_OUTPUT_DIR, f"lake_vs_background_{name}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


def plot_pixel_change_map(dz_map, lake_gdf=None, output_path=None,
                           lake_name=None, pixel_area=None, show_volume=False,
                           downsample_factor=None, vmax_percentile=98,
                           nodata_color="lightgrey", signed=True):
    """
    Zoomed-in map of per-pixel values, either from refined_volume_from_zarr()
    ("dz_map") or build_before_after_diff_map() ("diff_map").

    signed: set this based on which one you're plotting --
      - diff_map: signed=True (default). diff = after_mean - before_mean can
        genuinely be positive (filling) or negative (draining) -- diverging
        colormap (RdBu_r) centred at zero is correct here.
      - dz_map: pass signed=False. dz_map = 95th_percentile - 5th_percentile
        per pixel is a SPREAD, not a signed change -- it is structurally
        always >= 0 (a percentile subtraction can't go negative), so the
        blue half of a diverging colormap is always empty/unused there.
        signed=False switches to a sequential colormap starting at 0,
        which is the correct choice for a magnitude, and visually keeps
        dz_map and diff_map from being confused for the same kind of
        quantity when shown side by side.

    downsample_factor: if given (e.g. 4), block-averages the map by that
    factor in both x and y before plotting -- for a rough visual sketch on
    a large polygon, not for anything you'd read exact values off. Uses
    NaN-aware averaging, so a downsampled cell is only NaN if ALL pixels
    inside it were NaN.

    vmax_percentile: color scale is set from this percentile of |data|
    (signed=True) or of data (signed=False), not the raw max -- a single
    outlier pixel can otherwise stretch the scale so far that every real,
    small value renders as near-white/near-background and the plot looks
    empty even with valid data underneath.

    NaN cells (no contributing data) are rendered in nodata_color rather
    than left transparent -- transparent-on-white can also look like "no
    data" even when it's really "no data AND that's worth seeing".
    """
    da = dz_map
    if downsample_factor is not None and downsample_factor > 1:
        da = da.coarsen(x=downsample_factor, y=downsample_factor,
                         boundary="trim").mean()

    data = da.values
    x, y = da["x"].values, da["y"].values

    if show_volume and pixel_area is not None:
        # NOTE: if downsampled, each cell now covers downsample_factor^2
        # original pixels -- scale accordingly so this stays a per-ORIGINAL-
        # pixel volume figure, not per-downsampled-block:
        cell_area = pixel_area * (downsample_factor ** 2 if downsample_factor else 1)
        data = data * cell_area
        cbar_label = ("Volume change per pixel (m$^3$)" if signed
                      else "Volume magnitude per pixel (m$^3$)")
    else:
        cbar_label = "Elevation change (m)" if signed else "Elevation change magnitude (m)"

    valid = data[~np.isnan(data)]
    if valid.size == 0:
        print("map is entirely NaN after any downsampling -- nothing to plot")
        return None

    data_masked = np.ma.masked_invalid(data)

    if signed:
        vmax = np.percentile(np.abs(valid), vmax_percentile)
        if vmax == 0:
            vmax = np.max(np.abs(valid)) or 1.0
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad(color=nodata_color)
        vmin_plot, vmax_plot = -vmax, vmax
    else:
        if np.any(valid < 0):
            print("warning: signed=False but negative values found -- "
                  "is this really a magnitude quantity (e.g. dz_map)? "
                  "check you didn't pass diff_map here by mistake.")
        vmax = np.percentile(valid, vmax_percentile)
        if vmax == 0:
            vmax = np.max(valid) or 1.0
        cmap = plt.get_cmap("Reds").copy()  # sequential, starts at 0 -- no blue half wasted
        cmap.set_bad(color=nodata_color)
        vmin_plot, vmax_plot = 0, vmax

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(x, y, data_masked, cmap=cmap, vmin=vmin_plot, vmax=vmax_plot,
                        shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)

    # Lock axis limits to the pixel map's own extent BEFORE any overlay --
    # if lake_gdf is in a mismatched CRS (e.g. still geographic degrees
    # instead of the tile's projected CRS), plotting its boundary on these
    # axes can otherwise blow the autoscaled view out to a huge range,
    # squeezing the actual mesh into an invisible sliver. Re-locking after
    # the overlay call guards against this regardless of root cause.
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min(y_min, y_max), max(y_min, y_max))

    if lake_gdf is not None:
        if lake_gdf.crs is not None and lake_gdf.crs.is_geographic:
            print("warning: lake_gdf is in a geographic CRS (degrees) -- "
                  "its boundary overlay will not align with this map "
                  "(which is in projected metres). Reproject lake_gdf to "
                  "the tile CRS before calling for a correct overlay; "
                  "skipping the overlay this time to protect the axis view.")
        else:
            lake_gdf.boundary.plot(ax=ax, color="black", linewidth=1.5)
            ax.set_xlim(x_min, x_max)  # re-lock in case the overlay reset it
            ax.set_ylim(min(y_min, y_max), max(y_min, y_max))

    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_aspect("equal")
    title = "Per-pixel elevation change"
    if downsample_factor:
        title += f" (downsampled {downsample_factor}x{downsample_factor})"
    if lake_name:
        title = f"{title} -- {lake_name}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path is None:
        name_slug = lake_name or "site"
        os.makedirs(VOLUME_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(VOLUME_OUTPUT_DIR, f"pixel_change_map_{name_slug}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


def plot_polygon_timeseries(ts_df, window=None, output_path=None, lake_name=None,
                             site_coords_4326=None, use_corrected=True,
                             remove_outliers=True):
    """
    Analogous to plot_elevation_history() but for the full-polygon
    buffer-corrected mega-timeseries from build_buffer_corrected_timeseries(),
    rather than the 5x5-cluster pseudo-pixel series -- built to mirror that
    function's structure/style closely for direct visual comparison.

    window: optional (t_start, t_end) -- restricts the plot to one episode,
    e.g. the output of manual_window()/manual_windows_from_breakpoints().
    Leave as None to see the full available range (useful for picking
    breakpoints visually in the first place).

    use_corrected: plot the buffer-corrected series (recommended -- isolates
    lake signal from regional background) vs. raw lake_mean. Error bars use
    sqrt(lake_std^2 + buffer_std^2) when using the corrected series (a
    simple, slightly-conservative combination of both regions' spatial
    std -- not a rigorous propagated uncertainty, since it ignores pixel
    counts and any spatial correlation between the two regions).
    """
    df = ts_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Filename years come from the input selection (the window, or the full
    # series), not from whatever survives the outlier cut below
    if window is not None:
        years = f"{pd.Timestamp(window[0]).year}-{pd.Timestamp(window[1]).year}"
        df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])]
    else:
        years = f"{df['date'].min().year}-{df['date'].max().year}"
    df = df.sort_values("date").reset_index(drop=True)

    if use_corrected:
        y_col, y_label = "corrected", "Buffer-corrected elevation change (m)"
        yerr = np.sqrt(df["lake_std"] ** 2 + df["buffer_std"] ** 2)
    else:
        y_col, y_label = "lake_mean", "Elevation (m)"
        yerr = df["lake_std"]

    valid = df[y_col].notna()
    df, yerr = df[valid].reset_index(drop=True), yerr[valid].reset_index(drop=True)
    if len(df) == 0:
        print("No valid data to plot")
        return None

    # remove outliers, same logic as plot_elevation_history -- CAUTION: a
    # real drainage/fill event can itself look like a statistical outlier
    # relative to a quieter series. If a dramatic event you know exists
    # from other evidence (e.g. a transect) isn't showing up here, try
    # remove_outliers=False first before assuming it's a coverage gap --
    # this filter would silently delete exactly that point.
    if remove_outliers:
        median, std = df[y_col].median(), df[y_col].std()
        keep = (df[y_col] <= median + 2 * std) & (df[y_col] >= median - 2 * std)
        n_dropped = (~keep).sum()
        if n_dropped > 0:
            print(f"remove_outliers dropped {n_dropped} point(s) -- rerun with "
                  f"remove_outliers=False if you suspect a real event got cut")
        df, yerr = df[keep].reset_index(drop=True), yerr[keep].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.errorbar(df["date"], df[y_col], yerr=yerr,
                fmt="o", markersize=6, color="royalblue",
                ecolor="gray", elinewidth=1.5, capsize=3,
                alpha=1.0, markeredgecolor="white",
                label="Polygon mean ± combined SD" if use_corrected
                      else "Polygon mean ± SD")

    if len(df) > 3:
        dates_num = mdates.date2num(df["date"])
        z = np.polyfit(dates_num, df[y_col], 1)
        p = np.poly1d(z)
        slope = z[0] * 365.25
        trend_color = "red" if slope < -0.5 else "green" if slope > 0.5 else "orange"
        ax.plot(df["date"], p(dates_num), "--", color=trend_color,
                linewidth=1.5, alpha=0.6, label=f"Trend: {slope:.2f} m/yr")

    mean_val = df[y_col].mean()
    ax.axhline(y=mean_val, color="gray", linestyle=":", linewidth=1, alpha=0.5,
               label=f"Mean: {mean_val:.1f}m")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xlabel("Date", fontsize=12)

    title_loc = (f"({site_coords_4326[0]:.3f}\u00b0E, {site_coords_4326[1]:.3f}\u00b0N)"
                 if site_coords_4326 else "")
    name_part = f"{lake_name} - " if lake_name else ""
    coverage_part = f"n={len(df)} strips, median coverage {df['n_valid_lake'].median():.0f}px"
    ax.set_title(f"Polygon Elevation History {title_loc}\n"
                 f"{name_part}{coverage_part}", fontsize=14, fontweight="bold")

    ax.legend(loc="best", fontsize=9, frameon=True, fancybox=True)
    plt.tight_layout()

    if output_path is None:
        name_slug = lake_name or "site"
        os.makedirs(VOLUME_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(VOLUME_OUTPUT_DIR,
                                   f"polygon_elevation_history_{name_slug}_{years}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


#%% 
# ===========================================================================
# Example end-to-end usage
# ===========================================================================
if __name__ == "__main__":
    # polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_august26.gpkg"
    # polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_01sept26.gpkg"
    polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_02sept26.gpkg"
    index_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/mosaic/ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
    gdf = gpd.read_file(index_path)
    lake_list_gdf = gpd.read_file(polygon_path)
    selection = lake_list_gdf[lake_list_gdf.index.isin([1, 9, 10, 11, 13, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31])]
    lake_names = ['Flade Isblink', 'Sermeq SW', 'Sermeq Centre', 'Sermeq NE', 'Inuppaat Quuaat 01 (Edge)', 'Upernavik W',\
                'Upernavik E', 'Usulluup Sermia (Big)', 'Inuppaat Quuaat 02 (Usu)', 'Candidate #5', 'Candidate #8',\
                'Candidate #7', 'Candidate #6', 'Sermeq (Jade)', 'Candidate #9', 'Candidate #4' ]
    selection['lake_name'] = lake_names
    print("CRS:", gdf.crs)
    print("columns:", list(gdf.columns))
    print()
    print(gdf.head(3))
    print()

    import gc
    SUPERLAKE_MIN_FRAC = 0.10  # min outline piece = 10% of the manual polygon area
    RING_GAP_FRAC = 0.5        # background ring starts 0.5 r outside the manual polygon
    RING_WIDTH_FRAC = 1.0      # ... and is 1 r wide (r = equivalent-circle radius)
    SD_MIN_THRESHOLD_M = None  # floor for the SD-outline threshold (m); None = ring-based only
    EPOCH_DAYS = 30            # strips within 30 days of an epoch's first strip form one epoch
    EPOCH_MIN_ABS_DZ_M = None  # floor for the epoch-difference threshold (m)
    SD_FIXED_THRESHOLDS_M = (2.0, 3.0, 5.0)  # extra SD contours compared per lake
    # Each run writes to its own folder: files from earlier runs that are
    # open elsewhere (Windows locks them) are never touched
    from datetime import datetime
    run_dir = os.path.join(VOLUME_OUTPUT_DIR, f"persistence_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"output folder: {run_dir}")
    gpkg_path = os.path.join(run_dir, "persistence_outlines.gpkg")
    qc_path = os.path.join(run_dir, "persistence_qc.csv")
    for old in (gpkg_path, os.path.join(run_dir, "coverage_strips.csv"),
                os.path.join(run_dir, "coverage_gaps.csv"),
                os.path.join(run_dir, "epochs.csv"), os.path.join(run_dir, "epoch_intervals.csv"),
                os.path.join(run_dir, "contour_volumes.csv"),
                os.path.join(run_dir, "contour_timeseries.csv")):
        if os.path.exists(old):
            os.remove(old)  # regenerated from scratch each run
    qc_rows = []

    # Process lakes in order of strips in their tile (fewest first), so
    # finished lakes appear early and the slowest ones (e.g. Flade) come last
    def _n_strips(subtile):
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                return sum(map(len, build_strip_index(subtile[:9]).values()))
        except Exception:
            return 0  # fails fast anyway (recorded as skipped)
    selection = selection.assign(_n_strips=selection["subtile"].map(_n_strips))
    selection = selection.sort_values("_n_strips", kind="stable")
    # Optional subset: VOLUME_LAKES="Candidate #8;Flade Isblink" runs only those
    only = [n.strip() for n in os.environ.get("VOLUME_LAKES", "").split(";") if n.strip()]
    if only:
        selection = selection[selection["lake_name"].isin(only)]
    print("processing order:", list(zip(selection["lake_name"], selection["_n_strips"])))
    for a in range(len(selection))[:]:
        lake_list_name = selection.iloc[a]["lake_name"]
        try:
            lake_gdf = selection.iloc[[a]]
            lakename = lake_gdf.lake_name.values[0]
            subtile = lake_gdf["subtile"].iloc[0]  # pull out the scalar string
            # Try to find the row matching your tile_id, checking likely column names
            tile_id = subtile[:9]  # Adjust the slicing based on your actual tile_id format
            site_id = tile_id

            # lake_gdf = lake_list_gdf[lake_list_gdf['subtile'] == tile_id]
            # lake_gdf = lake_list_gdf[lake_list_gdf.index == 13]
            lake_gdf = lake_gdf.to_crs("EPSG:3413")  # ensure same CRS for cropping
            lon, lat = get_site_lonlat_from_polygon(lake_gdf)
            try:
                strip_index = build_strip_index(tile_id)
                # print(f'strip index OK for {subtile}')
            except Exception as e:
                # strip_index = build_strip_index(tile_id, subfolder="mosaic_v_0_dh_0_vertical_offset_mean_nuthkaab_deramp_*50m")
                print(f'No strip index for {subtile}: {e}')
                qc_rows.append({"lake_name": lake_list_name, "status": "skipped",
                                "reason": f'no strip index: {e}'})
                pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
                continue

            # Buffer-corrected mega-timeseries (replaces the distant-control-site
            # noise floor for confirmed/delineated lakes -- see build_buffer_corrected_
            # timeseries docstring). Same streaming approach, same memory profile.
            # ts_df = build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id)
            ts_df, diag = build_buffer_corrected_timeseries(
                strip_index, lake_gdf, tile_id, return_diagnostics=True,
                ring_gap_frac=RING_GAP_FRAC, ring_width_frac=RING_WIDTH_FRAC)
            # Manual-polygon time series (gapped ring), no outlier removal:
            # the 2-sigma filter can delete real drainage points
            ts_df.to_csv(os.path.join(run_dir, f"timeseries_manual_{lakename}.csv"), index=False)
            try:
                plot_polygon_timeseries(
                    ts_df, lake_name=f"{lakename} (manual polygon)", remove_outliers=False,
                    site_coords_4326=(lon, lat),
                    output_path=os.path.join(run_dir, f"timeseries_manual_{lakename}.png"))
                plot_lake_vs_background(
                    ts_df, lake_name=lakename,
                    output_path=os.path.join(run_dir, f"lake_vs_background_{lakename}.png"))
            except Exception as e:
                print(f"WARNING: time-series plot failed for {lakename}: {e}")
            try:
                strips_df, gaps_df = coverage_gap_report(lake_gdf, tile_id, strip_index, diag)
                for df, name in ((strips_df, "coverage_strips.csv"), (gaps_df, "coverage_gaps.csv")):
                    if len(df):
                        df.insert(0, "lake_name", lakename)
                        path = os.path.join(run_dir, name)
                        df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
                print(f"coverage gaps:\n{gaps_df.to_string(index=False) if len(gaps_df) else 'none'}")
            except Exception as e:
                print(f"WARNING: coverage gap report failed for {lakename}: {e}")
            # print(f"DIAGNOSTICS: {diag[diag.status != 'kept']}")
            print(diag['status'].value_counts())
            print(f"mega-timeseries: {len(ts_df)} usable strips, "
                f"median lake coverage {ts_df['n_valid_lake'].median():.0f} px")

            if len(ts_df) < 3:
                print(f"Not enough usable strips for {lake_gdf.lake_name} to detect an event, skipping")
                qc_rows.append({"lake_name": lake_list_name, "status": "skipped",
                                "reason": "fewer than 3 usable strips"})
                pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
                continue

            noise_floor = estimate_noise_floor_from_buffer(ts_df)
            print(f"noise floor (from buffer ring, robust): {noise_floor:.3f} m")
    
            refined_windows = detect_event_window_from_series(
                ts_df["date"], ts_df["corrected"], noise_floor, n_bkps=4
            )
            print(f"candidate windows (n_bkps=4): {refined_windows}")

            # 1. Build a generous search region -- e.g. 2.5x your current polygon's radius
            current_radius = np.sqrt(lake_gdf.geometry.area.iloc[0] / np.pi)
            search_gdf = gpd.GeoDataFrame(
                geometry=lake_gdf.geometry.buffer(current_radius * 1.5), crs=lake_gdf.crs
            )
            # 2. Plot the persistence map FIRST, before trusting any polygon it derives
            auto_periods = sliding_windows_by_count(strip_index, strips_per_window=15, step_strips=7)
            result = build_persistence_mask(
                strip_index, search_gdf, tile_id, pixel_area=4.0,
                lake_gdf=lake_gdf,
                periods=auto_periods,  # or leave as None for n_periods split
                # threshold_m=noise_floor * 2.5,      # tune this -- see caveats below
                min_persistence_frac=0.5,
                keep_strip_cache=True,  # reused by the epoch differences below
            )
            print(result["threshold_m_used"])
            # 3. Outlines (per period + envelope), saved before plotting so a
            #    plotting failure can't lose them; QC table rewritten per lake
            # Minimum piece size for the superlake and the SD outline, as a
            # fraction of the manual polygon's area (drops scattered noise)
            min_piece_m2 = SUPERLAKE_MIN_FRAC * float(lake_gdf.geometry.area.iloc[0])
            period_gdf, superlake_gdf = persistence_outlines(
                result, search_gdf.crs, superlake_min_area_m2=min_piece_m2)
            try:
                sd_gdf, sd_thr, ring_gdf, sd_sens = sd_outline(
                    result, lake_gdf, min_area_m2=min_piece_m2,
                    ring_gap_frac=RING_GAP_FRAC, ring_width_frac=RING_WIDTH_FRAC,
                    min_threshold_m=SD_MIN_THRESHOLD_M)
                print(f"SD outline: threshold {sd_thr:.2f} m, {len(sd_gdf)} piece(s)")
            except Exception as e:
                print(f"WARNING: SD outline failed for {lakename}: {e}")
                sd_gdf, sd_thr, ring_gdf, sd_sens = None, None, None, None
            # Option B: outlines from differences between consecutive epochs
            epoch_res = None
            try:
                epoch_res = build_epoch_differences(
                    result["strip_cache"], strip_index, lake_gdf, result["ref_coords"],
                    search_gdf.crs, epoch_days=EPOCH_DAYS,
                    ring_gap_frac=RING_GAP_FRAC, ring_width_frac=RING_WIDTH_FRAC,
                    min_abs_dz_m=EPOCH_MIN_ABS_DZ_M, min_piece_m2=min_piece_m2,
                    cleaning=result["cleaning"])
                ep = epoch_res["epochs_df"].assign(lake_name=lakename)
                iv = epoch_res["intervals_df"].assign(lake_name=lakename)
                for df, name in ((ep, "epochs.csv"), (iv, "epoch_intervals.csv")):
                    path = os.path.join(run_dir, name)
                    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
            except Exception as e:
                print(f"WARNING: epoch differences failed for {lakename}: {e}")
            # Compare contours: manual polygon and SD outlines (ring-based and
            # fixed thresholds) -- time series, volumes, one GeoPackage each
            try:
                contours = {"manual": lake_gdf.to_crs(search_gdf.crs)}
                if sd_gdf is not None and len(sd_gdf):
                    contours[f"SD ring-based ({sd_thr:.2f} m)"] = sd_gdf
                for thr in SD_FIXED_THRESHOLDS_M:
                    g_thr, _, _, _ = sd_outline(result, lake_gdf, min_area_m2=min_piece_m2,
                                                ring_gap_frac=RING_GAP_FRAC,
                                                ring_width_frac=RING_WIDTH_FRAC,
                                                sensitivity_thresholds_m=(),
                                                fixed_threshold_m=thr)
                    if len(g_thr):
                        contours[f"SD >= {thr:g} m"] = g_thr
                    g_out = g_thr.assign(lake_name=lakename, tile_id=tile_id, sd_threshold_m=thr)
                    _append_gpkg_layer(os.path.join(run_dir, f"contours_sd_{thr:g}m.gpkg"),
                                       "sd_outline", g_out)
                if sd_gdf is not None and len(sd_gdf):
                    _append_gpkg_layer(os.path.join(run_dir, "contours_sd_ring.gpkg"), "sd_outline",
                                       sd_gdf.assign(lake_name=lakename, tile_id=tile_id,
                                                     sd_threshold_m=sd_thr))
                ts_by, vol_by, rows = {}, {}, []
                for cname, cg in contours.items():
                    ts_c = contour_timeseries(result["strip_cache"], strip_index, cg, ring_gdf,
                                              result["ref_coords"])
                    ts_by[cname] = ts_c
                    if len(ts_c):
                        rows.append(ts_c.assign(lake_name=lakename, contour=cname))
                    if epoch_res is not None:
                        v_c = contour_volumes(epoch_res, cg, result["ref_coords"])
                        vol_by[cname] = v_c
                        if len(v_c):
                            path = os.path.join(run_dir, "contour_volumes.csv")
                            v_c.assign(lake_name=lakename, contour=cname).to_csv(
                                path, mode="a", header=not os.path.exists(path), index=False)
                if rows:
                    path = os.path.join(run_dir, "contour_timeseries.csv")
                    pd.concat(rows).to_csv(path, mode="a", header=not os.path.exists(path),
                                           index=False)
                plot_contour_comparison(ts_by, vol_by, lake_name=lakename, output_dir=run_dir)
            except Exception as e:
                print(f"WARNING: contour comparison failed for {lakename}: {e}")
            result["strip_cache"] = None  # free the cropped strips
            gc.collect()
            try:
                export_persistence_gpkg(
                    gpkg_path, lakename, tile_id, lake_gdf, search_gdf,
                    result, period_gdf, superlake_gdf,
                    sd_outline_gdf=sd_gdf, ring_gdf=ring_gdf,
                    epoch_outlines_gdf=epoch_res["outlines_gdf"] if epoch_res else None,
                    epoch_superlake_gdf=epoch_res["superlake_gdf"] if epoch_res else None)
            except Exception as e:
                print(f"WARNING: GeoPackage export failed for {lakename}: {e}")
            try:
                ice = lake_ice_thickness(lake_gdf)
            except Exception as e:
                print(f"WARNING: ice thickness failed for {lakename}: {e}")
                ice = {"ice_thickness_m": np.nan, "ice_thickness_centroid_m": np.nan,
                       "bed_error_m": np.nan}
            qc_rows.append({"lake_name": lakename, "tile_id": tile_id, "status": "ok",
                            "ice_thickness_centroid_m": ice["ice_thickness_centroid_m"],
                            "bed_error_m": ice["bed_error_m"],
                            **persistence_qc(result, lake_gdf, search_gdf,
                                             superlake_gdf=superlake_gdf,
                                             sd_outline_gdf=sd_gdf,
                                             sd_threshold_m=sd_thr,
                                             sd_sensitivity_df=sd_sens,
                                             epoch_res=epoch_res)})
            qc_rows[-1] = add_d_over_h(qc_rows[-1], ice["ice_thickness_m"])
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
            print(f"QC table updated: {qc_path}")
            # 4. Diagnostics figures
            plot_persistence_diagnostics(result, lake_gdf, search_gdf, lake_name=lakename,
                                         output_dir=run_dir,
                                         period_gdf=period_gdf, superlake_gdf=superlake_gdf,
                                         sd_outline_gdf=sd_gdf, sd_threshold_m=sd_thr,
                                         ring_gdf=ring_gdf,
                                         epoch_superlake_gdf=(epoch_res["superlake_gdf"]
                                                              if epoch_res else None))
            if epoch_res is not None:
                plot_epoch_differences(epoch_res, lake_gdf, search_gdf, result["ref_coords"],
                                       lake_name=lakename, output_dir=run_dir)
            del result, period_gdf, superlake_gdf, epoch_res
            gc.collect()
        except Exception as e:
            # One lake failing (no coverage, no usable periods, ...) must not
            # stop the run: record why and move on
            reason = f"{type(e).__name__}: {e}"
            print(f"SKIPPED {lake_list_name}: {reason}")
            qc_rows.append({"lake_name": lake_list_name, "status": "skipped",
                            "reason": reason})
            pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
            gc.collect()

    if qc_rows:
        print(pd.DataFrame(qc_rows).round(2).to_string())
