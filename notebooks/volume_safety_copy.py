"""
Stage 1: coarse event detection from existing 5x5-cluster elevation-history files.
Stage 2: refined volume from full-resolution zarr rasters, cropped to lake polygons,
         restricted to the event window found in Stage 1.
"""
# %%
import os
import re
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

# NOTE: elevation values in the history files are already long-term-detrended
# (confirmed: smooth, large-scale detrend, so residual differences from raw
# are expected to be small). Stage 1 event detection below therefore works
# directly on the elevation column -- no extra per-site detrend applied.
# simple_detrend() is kept only for optional use on control/off-lake sites
# when estimating the noise floor, where a residual large-scale trend is
# harmless to remove.


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


def get_tile_zarr_dir(tile_id, subfolder, base_dir=TILE_ZARR_BASE):
    """
    subfolder: REQUIRED, no default -- see list_tile_zarr_subfolders(tile_id)
    to see what's available for this specific tile before choosing. Picking
    silently would be a real methodological choice (which reference/filter
    pipeline the strips were processed with), not just a naming detail.
    """
    path = os.path.join(base_dir, tile_id, subfolder)
    if not os.path.isdir(path):
        available = list_tile_zarr_subfolders(tile_id, base_dir)
        raise FileNotFoundError(
            f"{path} doesn't exist. Available subfolders for tile {tile_id}: "
            f"{available}"
        )
    return path
 

def list_tile_zarr_subfolders(tile_id, base_dir=TILE_ZARR_BASE):
    """
    Lists the processing-variant subfolders actually present for this tile
    (e.g. cs2- vs mosaic-referenced, with/without outlier removal). Use this
    to see your options before choosing one for build_strip_index().
    """
    tile_dir = os.path.join(base_dir, tile_id)
    if not os.path.isdir(tile_dir):
        raise FileNotFoundError(f"No tile directory found: {tile_dir}")
    return sorted(
        entry.name for entry in os.scandir(tile_dir) if entry.is_dir()
    )



 
 
def select_preferred_subfolder(tile_id, base_dir=TILE_ZARR_BASE):
    """
    Applies your stated preferences to the raw subfolder list:
    1. "outliersout" (filters strips with SD > 50 m) is ALWAYS preferred
       over the equivalent non-outliersout variant of the same reference
       pipeline (cs2 or mosaic) -- collapsed automatically, no ambiguity.
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
        if name.startswith("cs2"):
            return "cs2"
        if name.startswith("mosaic"):
            return "mosaic"
        return name  # unrecognised naming -- treat as its own group, don't guess
 
    by_reference = {}
    for name in available:
        by_reference.setdefault(reference_type(name), []).append(name)
 
    collapsed = []
    for ref, names in by_reference.items():
        outliersout_variants = [n for n in names if "outliersout" in n]
        collapsed.append(outliersout_variants[0] if outliersout_variants else names[0])
 
    if len(collapsed) == 1:
        chosen = collapsed[0]
        if "nocoreg" in chosen:
            print(f"WARNING: tile {tile_id}'s only available subfolder "
                  f"({chosen}) has NO coregistration applied -- this is a "
                  f"substantively noisier product than the cs2/mosaic-"
                  f"referenced data used elsewhere. Results for this tile "
                  f"should be treated as lower-confidence unless a "
                  f"coregistered alternative genuinely doesn't exist.")
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


def get_tile_transform(tile_id, index_path=TILE_INDEX_PATH,
                        resolution=TILE_RESOLUTION, shape=TILE_SHAPE):
    """
    Returns (transform, crs) as an affine.Affine + CRS string, with the
    index polygon's buffer automatically stripped based on the known array
    shape, rather than a hardcoded 100 m assumption.
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


def build_lake_buffer_ring(lake_gdf, buffer_dist_m, inner_gap_m=0, tile_crs=None):
    """
    Returns a ring (annulus) polygon around the lake: an outer buffer minus
    an inner buffer. inner_gap_m > 0 leaves a gap between the delineated
    boundary and the start of the ring -- worth using something non-zero
    since the polygon boundary itself has some delineation uncertainty and
    you don't want ring pixels bleeding into real lake signal.
    buffer_dist_m: check Fan et al. 2023's Methods for their exact value
    before trusting this default -- not confirmed from search, left as a
    parameter rather than a guessed constant.

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

    outer = lake_gdf.geometry.buffer(buffer_dist_m)
    inner = lake_gdf.geometry.buffer(inner_gap_m) if inner_gap_m > 0 else lake_gdf.geometry
    ring = outer.difference(inner)
    return gpd.GeoDataFrame(geometry=ring, crs=lake_gdf.crs)


def build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id,
                                       buffer_dist_m=500, inner_gap_m=50,
                                       nodata=-9999.0, return_diagnostics=False):
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
    """
    _, tile_crs = get_tile_transform(tile_id)
    if lake_gdf.crs != tile_crs:
        lake_gdf = lake_gdf.to_crs(tile_crs)  # do this ONCE, before buffering
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
                                 buffer_dist_m=500, inner_gap_m=50,
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
                         buffer_dist_m=500, inner_gap_m=50, verbose=True):
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
        print("plotting as magnitude (signed=False) -- using sequential colormap")
        if np.any(valid < 0):
            print("warning: signed=False but negative values found -- "
                  "is this really a magnitude quantity (e.g. dz_map)? "
                  "check you didn't pass diff_map here by mistake.")
        vmax = np.percentile(valid, vmax_percentile)
        if vmax == 0:
            vmax = np.max(valid) or 1.0
        cmap = plt.get_cmap("Greens").copy()  # sequential, starts at 0 -- no blue half wasted
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
        output_path = f"pixel_change_map_{name_slug}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path


def plot_polygon_timeseries(ts_df, window=None, output_path=None, lake_name=None,
                             site_coords_4326=None, use_corrected=True):
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
    if window is not None:
        df = df[(df["date"] >= window[0]) & (df["date"] <= window[1])]
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

    # remove outliers, same logic as plot_elevation_history
    median, std = df[y_col].median(), df[y_col].std()
    keep = (df[y_col] <= median + 2 * std) & (df[y_col] >= median - 2 * std)
    df, yerr = df[keep].reset_index(drop=True), yerr[keep].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.errorbar(df["date"], df[y_col], yerr=yerr,
                fmt="o", markersize=6, color="royalblue",
                ecolor="gray", elinewidth=1.5, capsize=3,
                alpha=1.0, markeredgecolor="white",
                label="Polygon mean ± combined SD" if use_corrected
                      else "Polygon mean ± SD")

    trend_line = False
    if len(df) > 3 and trend_line is True:
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
        years = f"{df['date'].iloc[0].year}-{df['date'].iloc[-1].year}"
        name_slug = lake_name or "site"
        output_path = f"polygon_elevation_history_{name_slug}_{years}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path
 

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
 


def build_persistence_mask(strip_index, search_gdf, tile_id, pixel_area,
                            lake_gdf=None, periods=None, n_periods=6, threshold_m=None,
                            threshold_n_mad=3.0, mode="fraction",
                            min_persistence_frac=0.5, min_consecutive=2,
                            low_pct=5, high_pct=95, min_valid_per_period=2,
                            verbose=True):
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
    ref_da = None
    for period in periods:
        paths = strips_in_window(strip_index, period)
        if len(paths) < min_valid_per_period:
            print(f"period {period}: only {len(paths)} strips, skipping")
            continue
        try:
            result = refined_volume_from_zarr(
                paths, search_gdf, tile_id, pixel_area,
                low_pct=low_pct, high_pct=high_pct, window=period, verbose=False
            )
        except ValueError as e:
            print(f"period {period}: {e}, skipping")
            continue
        dz = result["dz_map"]
        if ref_da is None:
            ref_da = dz
        period_dz.append(dz.values)
 
    if len(period_dz) < 2:
        raise ValueError(
            f"Only {len(period_dz)} usable period(s) -- need at least "
            f"2 for a meaningful persistence check. Widen periods or lower "
            f"min_valid_per_period."
        )
 
    if threshold_m is None:
        # background mask: True where inside search_gdf but OUTSIDE lake_gdf
        x_vals, y_vals = ref_da["x"].values, ref_da["y"].values
        res_x = float(x_vals[1] - x_vals[0])
        res_y = float(y_vals[1] - y_vals[0])
        transform = rasterio.transform.from_origin(
            x_vals[0] - res_x / 2, y_vals[0] - res_y / 2, res_x, -res_y
        )
        lake_raster = rasterio.features.geometry_mask(
            lake_gdf.geometry, out_shape=ref_da.shape, transform=transform,
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
 
    persistence_da = xr.DataArray(persistence_metric, dims=("y", "x"),
                                   coords={"x": ref_da["x"].values, "y": ref_da["y"].values})
 
    x_vals, y_vals = ref_da["x"].values, ref_da["y"].values
    res_x = float(x_vals[1] - x_vals[0])
    res_y = float(y_vals[1] - y_vals[0])  # typically negative (north-up)
    transform = rasterio.transform.from_origin(
        x_vals[0] - res_x / 2, y_vals[0] - res_y / 2, res_x, -res_y
    )
 
    shapes = rasterio.features.shapes(candidate_mask, mask=(candidate_mask == 1),
                                       transform=transform)
    polygons = [shape(geom) for geom, val in shapes if val == 1]
 
    if verbose:
        criterion = (f"consecutive >= {min_consecutive}" if mode == "consecutive"
                     else f"fraction >= {min_persistence_frac:.0%}")
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
            "threshold_m_used": threshold_m}

def manual_window(start_date, end_date):
    """
    Bypass automatic detection entirely -- pass dates you've picked visually
    from the plot below, e.g. manual_window("2015-06-01", "2016-09-01").
    Returns a (t_start, t_end) tuple in the same format detect_event_window*
    functions return, so it drops straight into strips_in_window().
    """
    return pd.Timestamp(start_date), pd.Timestamp(end_date)


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

    for a in range(len(selection))[:]:
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
            continue

        # Buffer-corrected mega-timeseries (replaces the distant-control-site
        # noise floor for confirmed/delineated lakes -- see build_buffer_corrected_
        # timeseries docstring). Same streaming approach, same memory profile.
        # ts_df = build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id)
        ts_df, diag = build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id, return_diagnostics=True)
        print(f"DIAGNOSTICS: {diag[diag.status != 'kept']}")
        print(f"mega-timeseries: {len(ts_df)} usable strips, "
            f"median lake coverage {ts_df['n_valid_lake'].median():.0f} px")

        if len(ts_df) < 3:
            print(f"Not enough usable strips for {lake_gdf.lake_name} to detect an event, skipping")
            continue

        noise_floor = estimate_noise_floor_from_buffer(ts_df)
        print(f"noise floor (from buffer ring, robust): {noise_floor:.3f} m")
    
        # 1. Plot the FULL range first, unrestricted, to visually spot episodes:
        # plot_polygon_timeseries(ts_df, lake_name=str(lakename+'_full_'),
        #                         site_coords_4326=(lon, lat)
        #                         )
        # full_dz = full_range_dz_map(strip_index, lake_gdf, tile_id)
        # plot_pixel_change_map(full_dz["dz_map"], lake_gdf=lake_gdf, lake_name=lakename + '_full_dz_map', signed=False)

        # full_diff = full_range_diff_map(strip_index, lake_gdf, tile_id)
        # plot_pixel_change_map(full_diff["diff_map"], lake_gdf=lake_gdf, lake_name=lakename + '_full_diff_map')

        # 2a. Automatic detection -- try n_bkps if Pelt keeps returning full range
        #     (e.g. you can see roughly 2 episodes -> try n_bkps=4):
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
        result = build_persistence_mask(
            strip_index, search_gdf, tile_id, pixel_area=4.0,
            lake_gdf=lake_gdf,
            periods=None,  # or leave as None for n_periods split
            # threshold_m=noise_floor * 2.5,      # tune this -- see caveats below
            min_persistence_frac=0.5
        )
        print(result["threshold_m_used"])
        plot_pixel_change_map(result["persistence_map"], lake_gdf=lake_gdf,
                             lake_name=f"persistence_check_{lakename}", signed=False)

        # 3. Only then look at the candidate polygon
        result["candidate_gdf"].plot()

        # for w in range(len(refined_windows)-1):
        #     try:
        #         chosen_window = refined_windows[w]  # replace with your chosen/manual window
        #         paths_in_window = strips_in_window(strip_index, chosen_window)
        #         print(f"{len(paths_in_window)} strips found in refined window {chosen_window}")
        #         if len(paths_in_window) < 2:
        #             print(f"Not enough strips in window {chosen_window} for volume calculation, skipping")
        #             # Continue to the next element in the loop instead of breaking out of it
        #         lake_name = str(lakename+'_' + str(chosen_window[0].year) + '-' + str(chosen_window[1].year))

        #         # 3. Re-plot restricted to just the chosen episode, for the actual figure:
        #         plot_polygon_timeseries(ts_df, window=chosen_window, lake_name=lake_name,
        #                                 # site_coords_4326=(meta["lon"], meta["lat"]))
        #                                 site_coords_4326=(lon,lat)
        #         )   
        #         result1 = refined_volume_from_zarr(paths_in_window, lake_gdf, tile_id,
        #                                         pixel_area=4.0, window=chosen_window)  # 2x2 m native res
        #         plot_pixel_change_map(result1["dz_map"], lake_gdf=lake_gdf, lake_name=lake_name+'_dz_map')
        #         print(f'____________________\n Now the change map')
        
        #         # before = manual_window("2015-01-01", "2015-12-31")
        #         # after = manual_window("2024-01-01", "2024-12-31")
        #         before_window = manual_window(
        #             str(chosen_window[0] - pd.Timedelta(days=180)), str(chosen_window[0])
        #         )
        #         after_window = manual_window(
        #             str(chosen_window[1]), str(chosen_window[1] + pd.Timedelta(days=180))
        #         )
        #         result2 = build_before_after_diff_map(strip_index, lake_gdf, tile_id,
        #                                             before_window, after_window)
        #         print(f'Before {before_window} and After {after_window}')
        #         result2 = build_before_after_diff_map(strip_index, lake_gdf, tile_id, before_window, after_window)
        #         plot_pixel_change_map(result2["diff_map"], lake_gdf=lake_gdf, lake_name=lake_name+'_diff_map'    )

        #     except Exception as e:
        #         print(f"Error processing window {chosen_window}: {e}")
        #         continue

