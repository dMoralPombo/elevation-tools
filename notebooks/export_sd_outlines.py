"""
Export the semi-automatic SD outlines of every lake to a GeoPackage for QGIS, alongside the manual ones.

The outline is the one Fig 2 draws (fig2_methods.plot): the lake's SD map, smoothed with a Gaussian of
--smooth_m metres, thresholded at the --pct-th percentile of the smoothed SD inside the background ring,
cleaned like the lake_volume outlines, pieces smaller than 10 % of the manual area dropped. The ring
still comes from the manual outline (it starts 0.5 r outside it, 1 r wide).

It reads the Fig 2 caches (volume/fig2_cache/<lake>.npz/.json; make them with
`fig2_methods.py --lake <name> [--recompute]`), so it takes seconds. A lake with only a hand-picked
stack cache (e.g. "__nocoreg_outliersout_50m") uses that one; the stack used is in the attributes.

Layers written (EPSG:4326 like the manual file; areas and distances computed in EPSG:3413):
  sd_outlines  one row per lake: subtile (as in the manual file), lake_name, the outline (MultiPolygon),
               threshold, areas, overlap with the manual outline, stack and a "flag" for outlines that
               differ a lot from the manual one
  manual       the manual outlines of the same lakes, with lake_name
  rings        the background rings used for the threshold

Usage (from notebooks/):
    python export_sd_outlines.py                      # pct 99, smooth 20 m
    python export_sd_outlines.py --pct 99.5 --smooth_m 20 --out <file.gpkg>
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime

import geopandas as gpd
import numpy as np
from shapely import wkt
from shapely.geometry import MultiPolygon

sys.path.append(os.path.abspath(".."))
import fig2_methods as f2  # noqa: E402
import lake_volume as lv  # noqa: E402

CRS = "EPSG:3413"


def cache_for(lake):
    """Default cache of the lake, else a hand-picked-stack cache; None if there is none."""
    base = f2.cache_base(lake)
    if os.path.exists(base + ".npz"):
        return base
    alt = sorted(glob.glob(base + "__*.npz"))
    return alt[0][:-4] if alt else None


def stack_kind(stack):
    if not stack:
        return "?"
    if stack.startswith("nocoreg"):
        return "uncoregistered"
    if "cs2fb20m" in stack:
        return "new scheme (is2 + cs2)"
    if "nuthkaab" in stack:
        return "old chain (provisional)"
    return "other"


def outline(base, pct, smooth_m, label=""):
    with open(base + ".json") as f:
        meta = json.load(f)
    data = np.load(base + ".npz")
    sd, x, y = data["sd"], data["x"], data["y"]
    ref = (x, y)
    pixel = float(abs(x[1] - x[0]))
    manual = wkt.loads(meta["manual_wkt"])
    # Non-ice ground (GrIMP) out of the ring, also for caches made before the masking
    ring, ring_masked = lv.mask_nonice(wkt.loads(meta["ring_wkt"]), label=label)
    ring_gdf = gpd.GeoDataFrame(geometry=[ring], crs=CRS)
    sd_s = f2.smooth_nan(sd, smooth_m, pixel)
    in_ring = lv._ring_mask(ring_gdf, ref, sd.shape)
    vals = sd_s[in_ring & np.isfinite(sd_s)]
    thr = float(np.percentile(vals, pct))
    g = f2.outline_at(sd_s, thr, ref, f2.SUPERLAKE_MIN_FRAC * manual.area, CRS)
    geom = g.geometry.union_all() if len(g) else None
    return meta, manual, ring, ring_masked, thr, geom, len(g)


def compare(manual, geom):
    if geom is None or geom.is_empty:
        return {"area_km2": 0.0, "area_ratio": 0.0, "iou": 0.0, "manual_covered": 0.0, "outline_in_manual": 0.0,
                "pieces_off_manual": 0, "centroid_shift_m": None}
    inter = geom.intersection(manual).area
    pieces = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    return {"area_km2": round(geom.area / 1e6, 3),
            "area_ratio": round(geom.area / manual.area, 2),
            "iou": round(inter / geom.union(manual).area, 2),
            "manual_covered": round(inter / manual.area, 2),
            "outline_in_manual": round(inter / geom.area, 2),
            "pieces_off_manual": sum(1 for p in pieces if not p.intersects(manual)),
            "centroid_shift_m": round(geom.centroid.distance(manual.centroid))}


def flag(row):
    reasons = []
    if row["area_km2"] == 0:
        reasons.append("no outline")
        if row["ring_nonice_masked"] > lv.NONICE_FLAG_FRAC:
            reasons.append(f"{row['ring_nonice_masked']:.0%} of the ring non-ice (masked)")
        return "; ".join(reasons)
    if row["iou"] < 0.5:
        reasons.append(f"IoU {row['iou']:.2f}")
    if not 0.5 <= row["area_ratio"] <= 2:
        reasons.append(f"area x{row['area_ratio']:.2f}")
    if row["pieces_off_manual"]:
        reasons.append(f"{row['pieces_off_manual']} piece(s) away from the manual outline")
    if row["ring_nonice_masked"] > lv.NONICE_FLAG_FRAC:
        reasons.append(f"{row['ring_nonice_masked']:.0%} of the ring non-ice (masked)")
    if row["lake_nonice"] > lv.NONICE_FLAG_FRAC:
        reasons.append(f"{row['lake_nonice']:.0%} of the lake non-ice (GrIMP)")
    if row["stack_kind"] != "new scheme (is2 + cs2)":
        reasons.append(row["stack_kind"])
    return "; ".join(reasons)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pct", type=float, default=99.0, help="Ring percentile of the smoothed SD")
    p.add_argument("--smooth_m", type=float, default=20.0, help="Gaussian smoothing of the SD map (m)")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(os.path.dirname(f2.POLYGON_PATH),
                                f"candidates_polygons_sd_p{a.pct:g}_s{a.smooth_m:g}m_{datetime.now():%Y%m%d_%H%M}.gpkg")

    src = gpd.read_file(f2.POLYGON_PATH)
    sel = src[src.index.isin(f2.LAKE_ROWS)].copy()
    sel["lake_name"] = f2.LAKE_NAMES
    rows, manual_rows, ring_rows = [], [], []
    for subtile, lake in sel[["subtile", "lake_name"]].itertuples(index=False):
        base = cache_for(lake)
        if base is None:
            print(f"  {lake}: no Fig 2 cache (run fig2_methods.py --lake '{lake}'), skipped")
            continue
        try:
            meta, manual, ring, ring_masked, thr, geom, n = outline(base, a.pct, a.smooth_m, label=lake)
        except Exception as e:  # e.g. a cache truncated by a killed run
            print(f"  {lake}: cache unreadable ({type(e).__name__}: {e}); rebuild it with fig2_methods.py "
                  f"--lake '{lake}' --recompute, skipped")
            continue
        row = {"subtile": subtile, "lake_name": lake, "tile_id": meta["tile_id"],
               "rule": f"ring p{a.pct:g} of SD smoothed {a.smooth_m:g} m", "threshold_m": round(thr, 2),
               "n_pieces": n, "manual_area_km2": round(manual.area / 1e6, 3), **compare(manual, geom),
               "ring_nonice_masked": round(ring_masked, 3),
               "lake_nonice": round(lv.nonice_fraction(geom if geom is not None and not geom.is_empty
                                                       else manual), 3),
               "stack": meta.get("stack"), "stack_kind": stack_kind(meta.get("stack")),
               "n_strips_in_stack": meta.get("n_strips_in_stack"), "sd_computed": meta.get("computed")}
        row["flag"] = flag(row)
        if geom is not None and not geom.is_empty:
            row["geometry"] = geom if isinstance(geom, MultiPolygon) else MultiPolygon([geom])
        else:
            row["geometry"] = None
        rows.append(row)
        manual_rows.append({"subtile": subtile, "lake_name": lake, "geometry": manual})
        ring_rows.append({"lake_name": lake, "geometry": ring})
        print(f"  {lake}: SD >= {thr:.2f} m, {row['area_km2']} km2 vs manual {row['manual_area_km2']} "
              f"(x{row['area_ratio']}, IoU {row['iou']}) {('<- ' + row['flag']) if row['flag'] else ''}")

    sd = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS).to_crs("EPSG:4326")
    # GeoPackage = SQLite, whose locking fails on the luna (CIFS) mount: write locally, then copy
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, os.path.basename(out))
        sd.to_file(local, layer="sd_outlines", driver="GPKG")
        gpd.GeoDataFrame(manual_rows, crs=CRS).to_crs("EPSG:4326").to_file(local, layer="manual", driver="GPKG")
        gpd.GeoDataFrame(ring_rows, crs=CRS).to_crs("EPSG:4326").to_file(local, layer="rings", driver="GPKG")
        shutil.copyfile(local, out)
    sd.drop(columns="geometry").to_csv(out[:-5] + ".csv", index=False)
    print(f"Saved {out} (layers sd_outlines, manual, rings) and {os.path.basename(out[:-5])}.csv")


if __name__ == "__main__":
    main()
