"""
Figure 2 of the paper draft: the method on one lake, in three panels.

  a  per-pixel standard deviation (SD) of elevation over the record, with the manual outline,
     the background ring and the SD outline
  b  elevation change over one event interval (epoch differences), with the same outlines
  c  the event's volume change against the area of the outline used

The look (fonts, titles, map axes, colours, legend) comes from paper_style.py, shared with
the other paper figures.

Two steps, so the styling and the SD-outline rule can be changed without recomputing:
  1. compute (about 2 min for Candidate #8): crops the stack and builds the SD map and the epoch
     differences with the lake_volume functions; saves what the figure needs to
     OUTPUT_DIR/volume/fig2_cache/<lake>.npz + .json. The ring (--gap, --width) and the
     interval are fixed here.
  2. plot: reads that cache, makes the SD outline (--smooth_m, --rule, --pct, --n_mad,
     --threshold) and writes OUTPUT_DIR/volume/fig2_<YYYYMMDD_HHMM>/fig2_<lake>.png/.pdf.

Usage (from notebooks/):
    python fig2_methods.py                           # compute if there is no cache, then plot
    python fig2_methods.py --recompute               # e.g. after the tile's stack was rebuilt
    python fig2_methods.py --all                     # every lake (from caches), with provenance.csv
    python fig2_methods.py --all --recompute         # the same, rebuilding every cache
    python fig2_methods.py --curve                   # panel c as a continuous curve
    python fig2_methods.py --lake "Candidate #5" --subfolder nocoreg_outliersout_50m
    python fig2_methods.py --rule fixed --threshold 3
    python fig2_methods.py --rule mad --n_mad 3 --smooth_m 0     # lake_volume's own rule
    python fig2_methods.py --lake "Upernavik W" --interval 3
    python fig2_methods.py --gap 0.5 --width 1.0     # ring: starts gap x r outside, width x r wide

The numbers are provisional until the lake's tile has a stack from the current
coregistration scheme (see notes/HUB.md, section 5); the stack used is printed and saved
next to the figure.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely import wkt  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import LUNA_ROOT  # noqa: E402
import lake_volume as lv  # noqa: E402
import paper_style as ps  # noqa: E402

# Same polygon file, rows and names as 03_volume_batch.py
POLYGON_PATH = os.path.join(LUNA_ROOT, "moralpom", "globe", "data", "ArcticDEM",
                            "candidates_polygons_02sept26.gpkg")
LAKE_ROWS = [1, 9, 10, 11, 13, 19, 20, 21, 23, 24, 26, 27, 28, 29, 30, 31]
LAKE_NAMES = ["Flade Isblink", "Sermeq SW", "Sermeq Centre", "Sermeq NE", "Inuppaat Quuaat 01 (Edge)",
              "Upernavik W", "Upernavik E", "Usulluup Sermia (Big)", "Inuppaat Quuaat 02 (Usu)",
              "Candidate #5", "Candidate #8", "Candidate #7", "Candidate #6", "Sermeq (Jade)",
              "Candidate #9", "Candidate #4"]
TILE_REFERENCE = {"11_38_2_2": "mosaic", "12_38_1_2": "mosaic", "12_38_2_2": "mosaic"}
DEFAULT_REFERENCE = "altim"

SUPERLAKE_MIN_FRAC = 0.10
EPOCH_DAYS = 30
SD_FIXED_THRESHOLDS_M = (2.0, 3.0, 5.0)
CACHE_DIR = os.path.join(lv.VOLUME_OUTPUT_DIR, "fig2_cache")


def slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def load_lake(lake_name):
    gdf = gpd.read_file(POLYGON_PATH)
    sel = gdf[gdf.index.isin(LAKE_ROWS)].copy()
    sel["lake_name"] = LAKE_NAMES
    lake = sel[sel["lake_name"] == lake_name]
    if lake.empty:
        raise SystemExit(f"Unknown lake {lake_name!r}. Known: {', '.join(LAKE_NAMES)}")
    return lake.to_crs("EPSG:3413"), lake["subtile"].iloc[0][:9]


def cache_base(lake_name, subfolder=None):
    """Cache path stem; a stack chosen by hand gets its own cache."""
    return os.path.join(CACHE_DIR, slug(lake_name) + (f"__{slug(subfolder)}" if subfolder else ""))


def compute(lake_name, gap, width, interval=None, subfolder=None):
    """Runs the volume steps for one lake and saves what the figure needs."""
    lake_gdf, tile_id = load_lake(lake_name)
    chosen_by_hand = subfolder is not None
    if not chosen_by_hand:
        subfolder = lv.select_preferred_subfolder(
            tile_id, reference=TILE_REFERENCE.get(tile_id, DEFAULT_REFERENCE))
    print(f"{lake_name}: tile {tile_id}, stack {subfolder}")
    strip_index = lv.build_strip_index(tile_id, subfolder=subfolder)

    radius = float(np.sqrt(lake_gdf.geometry.area.iloc[0] / np.pi))
    # The crop must hold the whole ring (1.5 r is the batch default)
    search_gdf = gpd.GeoDataFrame(
        geometry=lake_gdf.geometry.buffer(radius * max(1.5, gap + width + 0.1)), crs=lake_gdf.crs)
    periods = lv.sliding_windows_by_count(strip_index, strips_per_window=15, step_strips=7)
    result = lv.build_persistence_mask(
        strip_index, search_gdf, tile_id, pixel_area=4.0, lake_gdf=lake_gdf, periods=periods,
        min_persistence_frac=0.5, keep_strip_cache=True)
    min_piece = SUPERLAKE_MIN_FRAC * float(lake_gdf.geometry.area.iloc[0])
    crs = search_gdf.crs

    manual_gdf = lake_gdf.to_crs(crs)
    # cached unmasked; non-ice ground is removed when the figure is drawn (plot)
    ring_gdf = lv.build_background_ring(lake_gdf, gap, width, tile_crs=crs, exclude_nonice=False)
    epoch_res = lv.build_epoch_differences(
        result["strip_cache"], strip_index, lake_gdf, result["ref_coords"], crs,
        epoch_days=EPOCH_DAYS, ring_gap_frac=gap, ring_width_frac=width,
        min_piece_m2=min_piece, cleaning=result["cleaning"])
    if not epoch_res["dz_maps"]:
        raise SystemExit(f"{lake_name}: fewer than two usable epochs, no elevation change to show")
    manual_vol = lv.contour_volumes(epoch_res, manual_gdf, result["ref_coords"])
    if interval is None:
        # the largest volume change (either sign) inside the manual outline, among the
        # intervals that cover at least half of it
        covered = manual_vol[manual_vol["coverage"] >= 0.5]
        interval = int((covered if len(covered) else manual_vol)["volume_m3"].abs().idxmax())
    iv = manual_vol.iloc[interval]
    epochs = epoch_res["epochs_df"]

    x, y = result["ref_coords"]
    sd = np.where(result["n_valid_map"] >= 3, result["sd_map"], np.nan)
    dates = sorted(strip_index)
    meta = {
        "lake_name": lake_name, "tile_id": tile_id,
        "polygon_file": POLYGON_PATH,
        "reference_requested": ("chosen by hand" if chosen_by_hand
                                else TILE_REFERENCE.get(tile_id, DEFAULT_REFERENCE)),
        "coregistered": not subfolder.startswith("nocoreg"),
        "stack": subfolder, "stack_dir": lv.get_tile_zarr_dir(tile_id, subfolder),
        "n_strips_in_stack": int(sum(len(v) for v in strip_index.values())),
        "stack_first_date": f"{dates[0]:%Y-%m-%d}", "stack_last_date": f"{dates[-1]:%Y-%m-%d}",
        "n_strips_with_data": int(result["n_strips_with_data"]),
        "n_strips_in_epochs": int(epochs["n_strips"].sum()), "n_epochs": int(len(epochs)),
        "ring_gap_frac": gap, "ring_width_frac": width, "radius_m": radius,
        "interval_index": interval, "n_intervals": int(len(manual_vol)),
        "interval_from": str(iv["from_epoch_end"]), "interval_to": str(iv["to_epoch_start"]),
        "interval_coverage": float(iv["coverage"]),
        "manual_wkt": manual_gdf.geometry.union_all().wkt,
        "ring_wkt": ring_gdf.geometry.union_all().wkt,
        "computed": datetime.now().isoformat(timespec="minutes"),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = cache_base(lake_name, subfolder if chosen_by_hand else None)
    np.savez_compressed(base + ".npz", sd=sd.astype("float32"),
                        dz=epoch_res["dz_maps"][interval].astype("float32"),
                        x=np.asarray(x), y=np.asarray(y))
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"cache saved: {base}.npz / .json")
    return base


def smooth_nan(a, sigma_m, pixel_m):
    """Gaussian smoothing that ignores gaps (sigma in metres)."""
    if not sigma_m:
        return a
    from scipy.ndimage import gaussian_filter
    s = sigma_m / pixel_m
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0.0), s)
    den = gaussian_filter(ok.astype("float32"), s)
    return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)


def outline_at(sd_s, threshold, ref, min_piece_m2, crs):
    """Outline of sd_s >= threshold, cleaned like the lake_volume outlines."""
    mask = lv._clean_mask(np.nan_to_num(sd_s, nan=-np.inf) >= threshold, 3, 1)
    polys, _ = lv._mask_to_polygons(mask, ref, min_piece_m2)
    return gpd.GeoDataFrame(geometry=polys, crs=crs)


def contour_stats(name, gdf, dz, ref, role="other", label=None):
    """Area, mean elevation change and volume change inside an outline."""
    inside = dz[lv._ring_mask(gdf, ref, dz.shape)]
    inside = inside[np.isfinite(inside)]
    area = float(gdf.geometry.union_all().area)
    mean_dz = float(inside.mean()) if inside.size else np.nan
    return {"name": name, "label": label or name, "role": role, "geom": gdf.geometry.union_all(),
            "area_km2": area / 1e6,
            "mean_dz_m": mean_dz, "volume_Mm3": mean_dz * area / 1e6}


def volume_curve(sd_s, dz, ref, inner_gdf, min_piece_m2, pixel, thresholds):
    """
    Area and volume change inside "SD >= threshold" for each threshold, kept to the region
    inside the background ring (inner_gdf). Pieces smaller than min_piece_m2 are dropped,
    as for the outlines. Returns rows of (threshold, area_km2, volume_Mm3).
    """
    from scipy.ndimage import label
    inner = lv._ring_mask(inner_gdf, ref, dz.shape)
    filled = np.nan_to_num(sd_s, nan=-np.inf)
    out = []
    for t in thresholds:
        mask = lv._clean_mask(filled >= t, 3, 1).astype(bool) & inner
        labels, n = label(mask)
        if n:
            sizes = np.bincount(labels.ravel())[1:] * pixel ** 2
            mask = np.isin(labels, np.flatnonzero(sizes >= min_piece_m2) + 1)
        area = float(mask.sum()) * pixel ** 2
        vals = dz[mask]
        vals = vals[np.isfinite(vals)]
        out.append((float(t), area / 1e6, float(vals.mean()) * area / 1e6 if vals.size else np.nan))
    return out


def plot(base, out_dir=None, smooth_m=20.0, rule="pct", n_mad=3.0, pct=99.0, threshold=None,
         legend_loc="upper left", curve=False):
    """
    Draws the figure from a cache written by compute().

    The SD outline is made here, so its rule can be changed without recomputing: the SD map is
    smoothed (Gaussian, smooth_m metres) and thresholded at
      rule "pct"   : the pct-th percentile of the smoothed SD inside the background ring;
      rule "mad"   : ring median + n_mad x robust SD (1.4826 x MAD) (lake_volume's rule is this
                     with n_mad = 3 and no smoothing);
      rule "fixed" : threshold (m).
    The ring itself only depends on the manual outline.
    """
    with open(base + ".json") as f:
        meta = json.load(f)
    data = np.load(base + ".npz")
    sd, dz, x, y = data["sd"], data["dz"], data["x"], data["y"]
    ref = (x, y)
    crs = "EPSG:3413"
    pixel = float(abs(x[1] - x[0]))
    hx, hy = pixel / 2, abs(y[1] - y[0]) / 2
    extent = (x.min() - hx, x.max() + hx, y.min() - hy, y.max() + hy)
    manual_gdf = gpd.GeoDataFrame(geometry=[wkt.loads(meta["manual_wkt"])], crs=crs)
    # Non-ice ground (GrIMP) out of the ring, also for caches made before the masking
    ring_geom, ring_masked = lv.mask_nonice(wkt.loads(meta["ring_wkt"]), label=meta.get("lake_name", ""))
    ring_gdf = gpd.GeoDataFrame(geometry=[ring_geom], crs=crs)
    manual, ring = manual_gdf.geometry.iloc[0], ring_gdf.geometry.iloc[0]
    min_piece = SUPERLAKE_MIN_FRAC * manual.area

    in_ring = lv._ring_mask(ring_gdf, ref, sd.shape)
    sd_s = smooth_nan(sd, smooth_m, pixel)
    ring_vals = sd_s[in_ring & np.isfinite(sd_s)]
    med = float(np.median(ring_vals))
    rsd = float(1.4826 * np.median(np.abs(ring_vals - med)))
    if rule == "pct":
        thr, rule_txt = float(np.percentile(ring_vals, pct)), f"ring {pct:g}th percentile"
    elif rule == "mad":
        thr, rule_txt = med + n_mad * rsd, f"ring median + {n_mad:g} robust SD"
    else:
        thr, rule_txt = float(threshold), "fixed"
    chosen = outline_at(sd_s, thr, ref, min_piece, crs)

    # Outlines compared in panels b and c: manual, the chosen one, lake_volume's threshold
    # (ring median + 3 robust SD of the unsmoothed map) and fixed thresholds, all drawn on
    # the smoothed map. An outline that spreads over a quarter of the background ring is
    # background, not lake: it is left out and listed in the values file.
    raw_vals = sd[in_ring & np.isfinite(sd)]
    raw_med = float(np.median(raw_vals))
    raw_thr = raw_med + 3 * 1.4826 * float(np.median(np.abs(raw_vals - raw_med)))
    rows = [contour_stats("Manual", manual_gdf, dz, ref, "manual")]
    skipped = []
    if len(chosen):
        rows.append(contour_stats(f"SD ≥ {thr:.2f} m ({rule_txt})", chosen, dz, ref, "chosen",
                                  label=f"SD ≥ {thr:.2f} m"))
    others = [(raw_thr, f"SD ≥ {raw_thr:.2f} m (ring median + 3 robust SD of the unsmoothed map)",
               f"SD ≥ {raw_thr:.2f} m")]
    others += [(t, f"SD ≥ {t:g} m", f"SD ≥ {t:g} m") for t in SD_FIXED_THRESHOLDS_M]
    for t, name, label in others:
        if abs(t - thr) <= 0.25:
            continue  # same outline as the chosen one, to within the line width
        g = outline_at(sd_s, t, ref, min_piece, crs)
        if not len(g):
            skipped.append({"name": name, "reason": "no pixels above the threshold"})
        elif g.geometry.union_all().intersection(ring).area > 0.25 * ring.area:
            skipped.append({"name": name, "reason": "spreads over the background ring"})
        else:
            rows.append(contour_stats(name, g, dz, ref, label=label))
    sd_geom = rows[1]["geom"] if len(chosen) else None
    # One colour per SD outline, shared by panels b and c: lowest threshold darkest
    sd_rows = sorted((r for r in rows if r["role"] != "manual"), key=lambda r: -r["area_km2"])
    for i, r in enumerate(sd_rows):
        r["color"] = ps.SD_RAMP[min(i, len(ps.SD_RAMP) - 1)]
    rows[0]["color"] = ps.C_MANUAL

    ps.apply()
    map_ratio = (extent[3] - extent[2]) / (extent[1] - extent[0])
    c_width = 0.72  # panel c is narrower than the maps
    # Height follows the map's shape (maps are about 1.93 in wide in this layout)
    fig_height = float(np.clip(1.38 + 1.93 * map_ratio, 2.4, 7.5))
    fig = plt.figure(figsize=(ps.FIG_WIDTH_IN, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, c_width], height_ratios=[1, ps.CBAR_HEIGHT_RATIO])
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    sd_max = float(np.ceil(np.nanpercentile(sd, 99.9)))
    im_sd = ax_a.imshow(np.ma.masked_invalid(sd), extent=extent, origin="upper", cmap=ps.CMAP_SD,
                        vmin=0, vmax=sd_max, interpolation="nearest")
    dz_lim = float(np.ceil(np.nanpercentile(np.abs(dz), 99.5) / 5) * 5)
    im_dz = ax_b.imshow(np.ma.masked_invalid(dz), extent=extent, origin="upper", cmap=ps.CMAP_DZ,
                        vmin=-dz_lim, vmax=dz_lim, interpolation="nearest")
    for col, (ax, im, label, ext) in enumerate(((ax_a, im_sd, "SD of elevation (m)", "max"),
                                                (ax_b, im_dz, "dh (m)", "both"))):
        ps.draw_outline(ax, ring, ps.C_RING, linewidth=1.1, linestyle=ps.RING_DASH)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_facecolor(ps.C_NODATA)
        # The two maps share an extent: Y labels on the left of a, latitude on the right of b
        ps.geo_axes(ax, extent, label_left=ax is ax_a, label_right=ax is ax_b)
        ps.map_colorbar(fig, im, fig.add_subplot(gs[1, col]), label, extend=ext)
    # a: the ring and the chosen SD outline; b: the manual outline and every SD outline of c
    if sd_geom is not None:
        ps.draw_outline(ax_a, sd_geom, rows[1]["color"])
    for r in sd_rows:
        ps.draw_outline(ax_b, r["geom"], r["color"], linewidth=0.9, casing=False)
    ps.draw_outline(ax_b, manual, ps.C_MANUAL)

    handles = [ps.outline_handle(ps.C_MANUAL, "Manual outline"),
               ps.outline_handle(ps.C_RING, "Background ring", ps.RING_DASH),
               ps.ramp_handle([r["color"] for r in sd_rows], "SD outlines")]
    ps.corner_legend(ax_b, handles, loc=legend_loc)

    # c: volume change of the event against the outline's area. Labels alternate between the
    # right and the left of the points, the two sides the curve leaves free
    points = [dict(r) for r in rows]
    extra_v, extra_a = [], []
    if curve:
        # Continuous version: every SD threshold from the lake's core down to the ring's
        # median, inside the region the ring encloses; it ends at that whole region
        inner_gdf = gpd.GeoDataFrame(
            geometry=[manual.buffer(meta["ring_gap_frac"] * meta["radius_m"])], crs=crs)
        in_inner = lv._ring_mask(inner_gdf, ref, sd.shape)
        t_hi = float(np.nanpercentile(sd_s[in_inner], 99.5))
        sweep = volume_curve(sd_s, dz, ref, inner_gdf, min_piece, pixel,
                             np.geomspace(t_hi, max(med, 0.05), 60))
        whole = contour_stats("Whole region inside the ring", inner_gdf, dz, ref)
        sweep = [c for c in sweep if c[1] > 0 and np.isfinite(c[2])]
        ax_c.plot([c[1] for c in sweep] + [whole["area_km2"]], [c[2] for c in sweep] + [whole["volume_Mm3"]],
                  color=ps.C_OTHER, linewidth=1.0, zorder=2)
        ax_c.axhline(whole["volume_Mm3"], color=ps.INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax_c.plot(whole["area_km2"], whole["volume_Mm3"], "o", markersize=5.5, markerfacecolor="white",
                  markeredgecolor=ps.INK, markeredgewidth=0.8, zorder=4)
        # the SD outlines' points are put on the curve (same region, same thresholds)
        for pt in points:
            if pt["role"] != "manual":
                t = float(pt["label"].split("≥")[1].split("m")[0])
                _, pt["area_km2"], pt["volume_Mm3"] = volume_curve(
                    sd_s, dz, ref, inner_gdf, min_piece, pixel, [t])[0]
        extra_v, extra_a = [whole["volume_Mm3"]] + [c[2] for c in sweep], [whole["area_km2"], sweep[0][1]]
        values_curve = {"whole_region_area_km2": whole["area_km2"],
                        "whole_region_volume_Mm3": whole["volume_Mm3"],
                        "curve": [{"sd_threshold_m": c[0], "area_km2": c[1], "volume_Mm3": c[2]}
                                  for c in sweep]}
    order = sorted(points, key=lambda r: r["area_km2"])
    if not curve:
        ax_c.plot([r["area_km2"] for r in order], [r["volume_Mm3"] for r in order],
                  color=ps.C_OTHER, linewidth=0.8, zorder=2)
    for r in order:
        ax_c.plot(r["area_km2"], r["volume_Mm3"], "o", markersize=5.5, color=r["color"],
                  markeredgecolor=ps.INK, markeredgewidth=0.5, zorder=5 if r["role"] == "manual" else 4)
    if curve:
        # Named in a small legend in the corner the curve leaves free; the unnamed dots are
        # the other SD outlines of panel b, in the same colours (larger area = lower threshold)
        from matplotlib.lines import Line2D

        def dot(color, label, face=None):
            return Line2D([], [], linestyle="none", marker="o", markersize=4.5, label=label,
                          markerfacecolor=face or color, markeredgecolor=ps.INK, markeredgewidth=0.5)
        named = [dot(r["color"], r["label"]) for r in rows if r["role"] in ("manual", "chosen")]
        named.append(dot(ps.INK, "Whole region\ninside the ring", face="white"))
        ps.corner_legend(ax_c, named, loc="upper right" if whole["volume_Mm3"] < 0 else "lower right")
    else:
        for i, r in enumerate(order):
            right = i % 2 == 0
            ax_c.annotate(r["label"], (r["area_km2"], r["volume_Mm3"]),
                          xytext=(4.5, 0) if right else (-4.5, 0), textcoords="offset points",
                          ha="left" if right else "right", va="center",
                          fontsize=ps.SMALL_PT - 0.5, color=ps.INK)
    vols = [r["volume_Mm3"] for r in points] + extra_v
    areas = [r["area_km2"] for r in points] + extra_a
    pad_v = 0.08 * (max(vols) - min(vols) or 1.0)
    pad_a = 0.21 * (max(areas) - min(areas) or 1.0)  # room for the labels beside the end points
    ax_c.set_ylim(min(vols) - pad_v, max(vols) + pad_v)
    ax_c.set_xlim(0 if curve else min(areas) - pad_a, max(areas) + (0.5 if curve else 1) * pad_a)
    ax_c.set_xlabel("Outline area (km²)")
    ax_c.set_ylabel("Volume change (10⁶ m³)")
    ax_c.grid(color=ps.C_GRID, linewidth=0.5)
    ax_c.set_axisbelow(True)
    ax_c.set_box_aspect(map_ratio / c_width)  # same height as the maps
    for side in ("top", "right"):
        ax_c.spines[side].set_visible(False)

    def month(date):
        return datetime.strptime(date[:10], "%Y-%m-%d").strftime("%b %Y")
    ps.panel_title(ax_a, "a", "Variability of elevation", pad=16)
    ps.panel_title(ax_b, "b", f"dh ({month(meta['interval_from'])} to {month(meta['interval_to'])})",
                   pad=16)
    ps.panel_title(ax_c, "c", "Volume change vs outline size", pad=16)
    ps.figure_title(fig, meta["lake_name"]
                    + ("" if meta.get("coregistered", True) else " (uncoregistered)"))

    out_dir = out_dir or os.path.join(lv.VOLUME_OUTPUT_DIR, f"fig2_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fig2_{slug(meta['lake_name'])}"
                        + ("" if meta.get("coregistered", True) else "_uncoregistered")
                        + ("_curve" if curve else ""))
    ps.save(fig, stem)
    plt.close(fig)
    values = {k: v for k, v in meta.items() if not k.endswith("_wkt") and k != "contours"}
    values.update(ring_nonice_masked_frac=round(ring_masked, 3),
                  lake_nonice_frac=round(lv.nonice_fraction(manual), 3),
                  sd_smoothing_m=smooth_m, sd_rule=rule_txt, sd_threshold_m=thr,
                  ring_median_m=med, ring_robust_sd_m=rsd, outlines_left_out=skipped,
                  outlines=[{k: (v.wkt if k == "geom" else v) for k, v in r.items()} for r in rows])
    if curve:
        values.update(values_curve)
    with open(stem + "_values.json", "w") as f:
        json.dump(values, f, indent=1)
    print(f"figure saved: {stem}.png / .pdf  (values: {stem}_values.json)")
    print(f"  stack: {meta['stack']}  ({meta['n_strips_with_data']} strips with data)")
    for item in skipped:
        print(f"  left out: {item['name']} ({item['reason']})")
    print(f"  SD smoothed {smooth_m:g} m; ring median {med:.2f} m, robust SD {rsd:.2f} m; "
          f"threshold {thr:.2f} m ({rule_txt}); interval {meta['interval_from']} -> {meta['interval_to']}")
    for r in rows:
        print(f"  {r['name']:52s} {r['area_km2']:5.2f} km2  {r['mean_dz_m']:7.2f} m  "
              f"{r['volume_Mm3']:7.1f} Mm3")
    return stem + ".png"


def run_all(extra_args, recompute=False):
    """
    Every lake of the polygon file (from its cache unless recompute), each in its own process (a big tile can then fail or
    run out of memory without stopping the others), fewest strips first. Figures, the log
    and provenance.csv go to one folder, OUTPUT_DIR/volume/fig2_all_<YYYYMMDD_HHMM>/.
    """
    import subprocess

    import pandas as pd
    out_dir = os.path.join(lv.VOLUME_OUTPUT_DIR, f"fig2_all_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out_dir, exist_ok=True)

    def n_strips(name):
        try:
            _, tile_id = load_lake(name)
            return len(lv.build_strip_index(
                tile_id, reference=TILE_REFERENCE.get(tile_id, DEFAULT_REFERENCE)))
        except Exception:
            return 0
    order = sorted(LAKE_NAMES, key=n_strips)
    records = []
    for name in order:
        print(f"=== {name}  ({datetime.now():%H:%M})", flush=True)
        log = os.path.join(out_dir, f"log_{slug(name)}.txt")
        with open(log, "w") as f:
            code = subprocess.call([sys.executable, "-u", os.path.abspath(__file__), "--lake", name,
                                    "--out_dir", out_dir, *(["--recompute"] if recompute else []),
                                    *extra_args],
                                   stdout=f, stderr=subprocess.STDOUT)
        values_path = os.path.join(out_dir, f"fig2_{slug(name)}_values.json")
        if code == 0 and os.path.exists(values_path):
            with open(values_path) as f:
                v = json.load(f)
            rec = {k: v.get(k) for k in (
                "lake_name", "tile_id", "reference_requested", "stack", "n_strips_in_stack",
                "stack_first_date", "stack_last_date", "n_strips_with_data", "n_strips_in_epochs",
                "n_epochs", "interval_from", "interval_to", "interval_coverage", "sd_threshold_m",
                "stack_dir", "polygon_file", "computed")}
            manual = next(o for o in v["outlines"] if o["role"] == "manual")
            rec.update(status="ok", manual_area_km2=manual["area_km2"], manual_dh_m=manual["mean_dz_m"],
                       manual_volume_Mm3=manual["volume_Mm3"],
                       outlines_left_out="; ".join(o["name"] for o in v["outlines_left_out"]))
        else:
            with open(log) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            rec = {"lake_name": name, "status": "failed", "reason": lines[-1][:300] if lines else ""}
        records.append(rec)
        pd.DataFrame(records).to_csv(os.path.join(out_dir, "provenance.csv"), index=False)
        print(f"    {rec['status']}", flush=True)
    print(f"done: {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lake", default="Candidate #8")
    p.add_argument("--all", action="store_true",
                   help="Every lake of the polygon file, into one fig2_all_<time> folder")
    p.add_argument("--recompute", action="store_true", help="Rebuild the cache even if it exists")
    p.add_argument("--interval", type=int, default=None,
                   help="Epoch interval to show (default: largest loss in the manual outline)")
    p.add_argument("--gap", type=float, default=0.5, help="Ring starts gap x r outside the outline")
    p.add_argument("--width", type=float, default=1.0, help="Ring is width x r wide")
    p.add_argument("--smooth_m", type=float, default=20.0,
                   help="Gaussian smoothing of the SD map before the outline (m; 0 = none)")
    p.add_argument("--rule", choices=["pct", "mad", "fixed"], default="pct",
                   help="SD threshold: ring percentile, ring median + n_mad robust SD, or fixed")
    p.add_argument("--pct", type=float, default=99.0)
    p.add_argument("--n_mad", type=float, default=3.0)
    p.add_argument("--threshold", type=float, default=None, help="SD threshold (m) for --rule fixed")
    p.add_argument("--legend_loc", default="upper left", help="Corner of panel b for the legend")
    p.add_argument("--curve", action="store_true",
                   help="Panel c as the continuous volume-vs-area curve out to the ring's inner edge")
    p.add_argument("--subfolder", default=None,
                   help="Stack folder to use instead of the preferred one, e.g. nocoreg_outliersout_50m")
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()
    if a.rule == "fixed" and a.threshold is None:
        p.error("--rule fixed needs --threshold")
    if a.all:
        passed = [arg for arg in sys.argv[1:] if arg not in ("--all", "--recompute")]
        run_all(passed, recompute=a.recompute)
        sys.exit(0)
    base = cache_base(a.lake, a.subfolder)
    custom = a.interval is not None or (a.gap, a.width) != (0.5, 1.0)
    if a.recompute or custom or not os.path.exists(base + ".npz"):
        base = compute(a.lake, a.gap, a.width, a.interval, a.subfolder)
    plot(base, a.out_dir, smooth_m=a.smooth_m, rule=a.rule, n_mad=a.n_mad, pct=a.pct,
         threshold=a.threshold, legend_loc=a.legend_loc, curve=a.curve)
