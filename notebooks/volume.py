"""
Stage 1: coarse event detection from existing 5x5-cluster elevation-history files.
Stage 2: refined volume from full-resolution zarr rasters, cropped to lake polygons,
         restricted to the event window found in Stage 1.
"""

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

# Folder naming: processed_<platform>_<YYYYMMDD>_<catalogID>_<catalogID>[_dt].zarr
# (filled_arrays has no "_dt"; archive tile_zarrs does)
STRIP_DATE_PATTERN = re.compile(r"processed_[A-Za-z0-9]+_(\d{8})_")

# filled_arrays/<tile>/<subfolder> has the same layout for every tile. The
# archive copy (.../ArcticDEM/tile_zarrs/<tile>/...) names its subfolders
# differently per tile, so pass base_dir/subfolder explicitly to use it.
TILE_ZARR_BASE = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/temp/filled_arrays"
TILE_ZARR_SUBFOLDER = "zarr_coreg_cs2_v_999_dh_999_vertical_offset_mean_nuthkaab_deramp"


def get_tile_zarr_dir(tile_id, base_dir=TILE_ZARR_BASE, subfolder=TILE_ZARR_SUBFOLDER):
    return os.path.join(base_dir, tile_id, subfolder)


def build_strip_index(tile_id, base_dir=TILE_ZARR_BASE, subfolder=TILE_ZARR_SUBFOLDER):
    """
    Scans a tile's zarr directory and returns {pd.Timestamp: [zarr_paths]}.
    Uses site->tile_id mapping you already maintain elsewhere -- pass the
    tile_id for the site you're processing.
    A date maps to a list because more than one strip can share an acquisition
    date (different catalog IDs); keep all of them, filtering happens later.
    """
    tile_dir = get_tile_zarr_dir(tile_id, base_dir, subfolder)
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


def zarr_path_from_pairname(tile_id, pairname, base_dir=TILE_ZARR_BASE,
                             subfolder=TILE_ZARR_SUBFOLDER):
    """
    Maps a history-file Pairname (e.g. 'WV01_20241007_1020010104A74400_
    1020010104A87A00') directly to its zarr folder, since the naming
    convention matches. Use this for validation -- it guarantees the strip
    actually has valid data at the site (it's literally where that history
    row's elevation value came from).
    """
    tile_dir = get_tile_zarr_dir(tile_id, base_dir, subfolder)
    for name in (f"processed_{pairname}.zarr", f"processed_{pairname}_dt.zarr"):
        path = os.path.join(tile_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No zarr for {pairname} in {tile_dir}")


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
                                       nodata=-9999.0):
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
    """
    _, tile_crs = get_tile_transform(tile_id)
    if lake_gdf.crs != tile_crs:
        lake_gdf = lake_gdf.to_crs(tile_crs)  # do this ONCE, before buffering
    buffer_gdf = build_lake_buffer_ring(lake_gdf, buffer_dist_m, inner_gap_m,
                                         tile_crs=tile_crs)
 
    records = []
    for date, paths_for_date in sorted(strip_index.items()):
        for zp in paths_for_date:
            try:
                lake_vals = crop_zarr_to_polygon(zp, lake_gdf, tile_id, nodata).values
                buf_vals = crop_zarr_to_polygon(zp, buffer_gdf, tile_id, nodata).values
            except Exception as e:
                print(f"skipping {zp}: {e}")
                continue
 
            n_lake = int(np.sum(~np.isnan(lake_vals)))
            n_buf = int(np.sum(~np.isnan(buf_vals)))
            if n_lake == 0 or n_buf == 0:
                continue
 
            lake_mean = float(np.nanmean(lake_vals))
            buf_mean = float(np.nanmean(buf_vals))
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
    if not records:
        raise ValueError("No strips produced valid lake+buffer data")
    return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)

 
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
 
 
def detect_event_window_from_series(dates, values, noise_floor,
                                     interp_freq_days=30, min_size=3):
    """
    Same change-point logic as detect_event_window(), generalised to take a
    plain (dates, values) pair -- use this on build_polygon_mean_timeseries()'s
    output ("date", "mean_elev" columns), which should be far less noisy than
    the cluster-based series and give the detector a much better chance.
    """
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "elev": values})
    return detect_event_window(df, noise_floor, elev_col="elev",
                                interp_freq_days=interp_freq_days, min_size=min_size)
 
 

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
        mean_map = stack.mean(dim="strip", skipna=True).where(has_enough)
        coverage = float(has_enough.compute().mean())
        return mean_map, coverage, len(cropped)
 
    before_paths = strips_in_window(strip_index, before_window)
    after_paths = strips_in_window(strip_index, after_window)
 
    before_mean, before_cov, n_before = _group_mean_map(before_paths, lake_gdf)
    after_mean, after_cov, n_after = _group_mean_map(after_paths, lake_gdf)
    diff_map = (after_mean - before_mean).compute()
 
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
        background_diff = float((buf_after_mean - buf_before_mean).mean(skipna=True))
        diff_map = diff_map - background_diff
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

 
def plot_pixel_change_map(dz_map, lake_gdf=None, output_path=None,
                           lake_name=None, pixel_area=None, show_volume=False,
                           downsample_factor=None, vmax_percentile=98,
                           nodata_color="lightgrey"):
    """
    Zoomed-in map of the per-pixel elevation-change values (dz_map, from
    refined_volume_from_zarr()'s "dz_map" key) with a diverging colourbar
    centred at zero. Optionally overlays the lake polygon boundary for
    context, and can show per-pixel volume (dz x pixel_area) instead of
    raw dz if pixel_area is given and show_volume=True.
 
    downsample_factor: if given (e.g. 4), block-averages the map by that
    factor in both x and y before plotting -- for a rough visual sketch on
    a large polygon, not for anything you'd read exact values off. Uses
    NaN-aware averaging, so a downsampled cell is only NaN if ALL pixels
    inside it were NaN.
 
    vmax_percentile: color scale is set from this percentile of |data|,
    not the raw max -- a single outlier pixel (e.g. a badly-masked edge
    value) can otherwise stretch the scale so far that every real, small
    dz value renders as near-white and the plot looks empty even with
    valid data underneath. This was the likely cause if you saw a blank
    plot despite dz_map having values.
 
    NaN cells (no contributing data, per refined_volume_from_zarr's
    coverage criteria) are rendered in nodata_color rather than left
    transparent -- transparent-on-white can also look like "no data" even
    when it's really "no data AND that's worth seeing", not nothing.
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
        cbar_label = "Volume change per pixel (m$^3$)"
    else:
        cbar_label = "Elevation change (m)"
 
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        print("dz_map is entirely NaN after any downsampling -- nothing to plot")
        return None
    vmax = np.percentile(np.abs(valid), vmax_percentile)
    if vmax == 0:
        vmax = np.max(np.abs(valid)) or 1.0
 
    data_masked = np.ma.masked_invalid(data)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color=nodata_color)
 
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(x, y, data_masked, cmap=cmap, vmin=-vmax, vmax=vmax,
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
 
    lake_3413 = lake_gdf.to_crs("EPSG:3413") if lake_gdf.crs != "EPSG:3413" else lake_gdf
    if lake_3413 is not None and not lake_3413.empty:
            lake_3413.boundary.plot(ax=ax, color="black", linewidth=1.5)
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
        output_path = f"polygons/pixel_change_map_{name_slug}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path



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
        output_path = f"polygons/polygon_elevation_history_{name_slug}_{years}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")
    return output_path
 
#%% 
# ===========================================================================
# Example end-to-end usage
# ===========================================================================
if __name__ == "__main__":
    index_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/mosaic/ArcticDEM_Mosaic_Index_latest_shp/ArcticDEM_Mosaic_Index_v4_1_2m.shp"
    polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_august26.gpkg"
    polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_01sept26.gpkg"
    polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_01sept26_test.gpkg"
    gdf = gpd.read_file(index_path)
    poly_gdf = gpd.read_file(polygon_path)

    print("CRS:", gdf.crs)
    print("columns:", list(gdf.columns))
    print()
    print(gdf.head(3))
    print()

    # Try to find the row matching your tile_id, checking likely column names
    tile_id = "16_39_1_1"
    # meta, df = parse_elevation_history(f"/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/elevation_histories/selected_candidates_28july2026/dubious/elevation_history_-48.710_67.610_none_2011-2024.txt")
    # tile_id = "12_39_1_1"
    # meta, df = parse_elevation_history(f"/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/elevation_histories/selected_candidates_28july2026/dubious/elevation_history_-48.710_67.610_none_2011-2024.txt")
    # # meta, df = parse_elevation_history(f"/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/elevation_histories/selected_candidates_28july2026/elevation_history_Flade_Isblink_(mitten)_-16.580_81.160_nc_2024-2012.txt")
    # tile_id = "16_39_2_2"
    # meta, df = parse_elevation_history(f"/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/elevation_histories/selected_candidates_28july2026/dubious/elevation_history_-48.710_67.610_none_2011-2024.txt")

    site_id = tile_id

    # control_files = [
    #     "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/elevation_histories/elevation_history_outside_flade_-16.631_81.169_nc_2024-2012.txt",
    # ]
    # noise_floor = estimate_noise_floor(control_files)

    # cluster_windows = detect_event_window(df, noise_floor)
    # for w in cluster_windows:
    #     lo, hi = robust_event_bounds(df, w)
    #     print(f"window {w}: cluster-level dz range = {hi - lo if lo else None}")

    # pick the window that corresponds to a real event, then:
    strip_index = build_strip_index(tile_id)

    # RUN THIS FIRST, on a strip known to have valid data at the site
    # (matched via Pairname from the history file), before trusting the
    # crop below -- picks the most recent full-coverage (Valid Pixels==25) row:
    # known_row = df[df["Valid Pixels"] >= 25].iloc[-1]
    # known_path = zarr_path_from_pairname(tile_id, known_row["Pairname"])
    # validate_georeferencing(known_path, tile_id,
    #                         #  site_lon=meta["lon"], site_lat=meta["lat"],
    #                          expected_elev=known_row["Elevation (m)"])
    # only proceed past this point once the printed diff looks reasonable

    lake_gdf = poly_gdf[poly_gdf['subtile'] == tile_id]
    lake_gdf = poly_gdf[poly_gdf.index == 13]

    # Buffer-corrected mega-timeseries (replaces the distant-control-site
    # noise floor for confirmed/delineated lakes -- see build_buffer_corrected_
    # timeseries docstring). Same streaming approach, same memory profile.
    ts_df = build_buffer_corrected_timeseries(strip_index, lake_gdf, tile_id)
    print(f"mega-timeseries: {len(ts_df)} usable strips, "
          f"median lake coverage {ts_df['n_valid_lake'].median():.0f} px")

    noise_floor = estimate_noise_floor_from_buffer(ts_df)
    print(f"noise floor (from buffer ring, robust): {noise_floor:.3f} m")
 
    # 1. Plot the FULL range first, unrestricted, to visually spot episodes:
    lon, lat = (-47.71134,67.77618) # for BIG ONE (16_39_2_2)
    lon, lat = (-47.83279, 67.90646) # for NORTH ONE (16_39_2_2)
    lon, lat = (-48.71,67.61) # for 16_39_1_1)
    lake_name=str(site_id+'_top_')

    plot_polygon_timeseries(ts_df, lake_name=lake_name,
                            #  site_coords_4326=(meta["lon"], meta["lat"])
                             site_coords_4326=(lon, lat)
                             )
    # !!!!!!!!
    # Am I sure it's plotting the full timeseries? The name seems to be reducing the range to 2015-2024, but the data goes back to 2011. Check the code for any filtering that might be happening before plotting.

    # 2a. Automatic detection -- try n_bkps if Pelt keeps returning full range
    #     (e.g. you can see roughly 2 episodes -> try n_bkps=4):
    refined_windows = detect_event_window_from_series(
        ts_df["date"], ts_df["corrected"], noise_floor, n_bkps=4
    )
    print(f"candidate windows (n_bkps=4): {refined_windows}")
 
    # # 2b. OR skip detection and pick windows by eye from the plot above:
    # chosen_window = manual_window("2015-06-01", "2016-09-01")
    # # or, for several episodes at once:
    # episode_windows = manual_windows_from_breakpoints(
    #     "2013-06-01", "2015-03-01", "2019-01-01", "2021-06-01"
    # )
 
    # chosen_window = refined_windows[0]  # replace with your chosen/manual window
    # paths_in_window = strips_in_window(strip_index, chosen_window)
    # print(f"{len(paths_in_window)} strips found in refined window {chosen_window}")

    for a in range(len(refined_windows)-1):
        chosen_window = refined_windows[a]  # replace with your chosen/manual window
        paths_in_window = strips_in_window(strip_index, chosen_window)
        print(f"{len(paths_in_window)} strips found in refined window {chosen_window}")

        # 3. Re-plot restricted to just the chosen episode, for the actual figure:
        plot_polygon_timeseries(ts_df, window=chosen_window, lake_name=str(lake_name+str(chosen_window[0].year)+'-'+str(chosen_window[1].year)),
                                # site_coords_4326=(meta["lon"], meta["lat"]))
                                site_coords_4326=(lon,lat)
        )   
        result = refined_volume_from_zarr(paths_in_window, lake_gdf, tile_id,
                                        pixel_area=4.0, window=chosen_window)  # 2x2 m native res

        plot_pixel_change_map(result["dz_map"], lake_gdf=lake_gdf, lake_name=str(lake_name+str(chosen_window[0].year)+'-'+str(chosen_window[1].year)))
