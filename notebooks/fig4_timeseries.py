"""
Figure 4 of the paper draft: lake time series with each lake's fill and drain intervals shaded,
per lake and per group of neighbouring lakes, to show which events happen at the same time.

Per lake (one panel):
  - black points and line: each strip's lake-mean minus background-ring mean over the SD
    outline (timeseries_sd_<lake>.csv from 03_volume_batch.py, no outlier removal), relative to
    the lake's median. The line joins the strips covering at least half of the lake (the ones
    the events use); partial strips are hollow points, unjoined, and the dh axis is scaled to
    the full strips, so a partial strip off that scale is a hollow arrow on the edge;
  - shading: the intervals between the two observations that bracket each event
    (lake_intervals.csv, event column; lake_volume.lake_events: lake-mean dh beyond 3 x the
    ring's robust sigma and at least 2 m, with at least half the lake observed).
    Drains dull red, fills hatched dull blue (red = lowering, as in the dh maps).

Groups: single-linkage on polygon centroids, a lake joining a group when it is within
--group_km (default 15 km) of any lake in it. With 15 km (Nov 2026): Candidates #7/#8/#9,
Sermeq SW/Centre/NE, Upernavik W/E; every other lake is at least 27 km from its nearest neighbour.

Outputs, in OUTPUT_DIR/volume/fig4_<YYYYMMDD_HHMM>/ (PNG 300 dpi + PDF, paper_style.py):
  fig4_lake_<lake>       one figure per lake
  fig4_group_<n>         one figure per group of two or more lakes: one panel per lake, stacked,
                         sharing the time and dh axes
  fig4_events.csv        the shaded intervals
  fig4_provenance.csv    run folder, stack, lake used, flags and group per lake

Reads only the CSVs of 03_volume_batch.py run folders (OUTPUT_DIR/volume/persistence_*); for a
lake in several folders the newest one with an SD-outline series wins.

Usage (from notebooks/):
    python fig4_timeseries.py                         # newest runs
    python fig4_timeseries.py --runs persistence_20261117_1022
    python fig4_timeseries.py --group_km 30           # looser groups
    python fig4_timeseries.py --final                 # without the "provisional" note
"""
import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
import paper_style as ps  # noqa: E402
from lake_volume import EVENT_MIN_COVERAGE, VOLUME_OUTPUT_DIR  # noqa: E402
from fig2_methods import LAKE_NAMES, LAKE_ROWS, POLYGON_PATH, slug  # noqa: E402

C_DRAIN = "#b2182b"   # red end of RdBu (ps.CMAP_DZ), shown pale
C_FILL = "#2166ac"    # blue end of RdBu, hatched
SHADE_ALPHA = 0.22
PIXEL_AREA_M2 = 4.0
PANEL_H_IN = 1.35


def run_dirs(names=None):
    """Run folders, oldest first (later ones win)."""
    if names:
        return [n if os.path.isabs(n) else os.path.join(VOLUME_OUTPUT_DIR, n) for n in names]
    return sorted(glob.glob(os.path.join(VOLUME_OUTPUT_DIR, "persistence_*")))


def load(dirs):
    """{lake: (series, intervals, qc row, run dir)} from the newest run with an SD series."""
    out = {}
    for d in dirs:
        iv_path, qc_path = os.path.join(d, "lake_intervals.csv"), os.path.join(d, "persistence_qc.csv")
        if not os.path.exists(iv_path):
            continue  # a run from before the SD-outline switch
        ivs = pd.read_csv(iv_path, parse_dates=["from_epoch_start", "from_epoch_end", "to_epoch_start"])
        qc = pd.read_csv(qc_path) if os.path.exists(qc_path) else pd.DataFrame(columns=["lake_name"])
        for path in glob.glob(os.path.join(d, "timeseries_sd_*.csv")):
            lake = os.path.basename(path)[len("timeseries_sd_"):-4]
            ts = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            iv = ivs[ivs["lake_name"] == lake]
            area = iv["contour_area_m2"].iloc[0] if len(iv) else np.nan
            n_px = area / PIXEL_AREA_M2 if np.isfinite(area) else ts["n_valid_lake"].max()
            ts["coverage"] = ts["n_valid_lake"] / n_px
            ts["anomaly"] = ts["corrected"] - ts["corrected"].median()
            q = qc[qc["lake_name"] == lake]
            out[lake] = (ts, iv, q.iloc[-1] if len(q) else pd.Series(dtype=object), d)
    return out


def lake_groups(lakes, max_km):
    """Single-linkage groups of the lakes' polygon centroids (EPSG:3413), in LAKE_NAMES order."""
    src = gpd.read_file(POLYGON_PATH)
    sel = src[src.index.isin(LAKE_ROWS)].copy()
    sel["lake_name"] = LAKE_NAMES
    sel = sel[sel["lake_name"].isin(lakes)].to_crs("EPSG:3413")
    cent = dict(zip(sel["lake_name"], sel.geometry.centroid))
    names = [n for n in LAKE_NAMES if n in cent]
    parent = {n: n for n in names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if cent[a].distance(cent[b]) <= max_km * 1e3:
                parent[root(b)] = root(a)
    groups = {}
    for n in names:
        groups.setdefault(root(n), []).append(n)
    return list(groups.values())


def y_range(series_list):
    """dh limits from the strips covering at least half the lake (partial strips can be far off)."""
    vals = np.concatenate([ts.loc[ts["coverage"] >= EVENT_MIN_COVERAGE, "anomaly"].values
                           for ts in series_list] or [np.array([0.0])])
    lo, hi = (float(np.min(vals)), float(np.max(vals))) if vals.size else (-1.0, 1.0)
    pad = 0.12 * max(hi - lo, 2.0)
    return lo - pad, hi + pad


def draw_lake(ax, ts, iv, ylim):
    """
    Event shading, then the series: a black line through the strips covering at least half
    the lake (the ones the events use), and hollow points for partial strips. Partial strips
    outside ylim sit on the edge as hollow arrows.
    Returns the events drawn.
    """
    ev = iv[iv["event"].fillna("").isin(["drain", "fill"])]
    for _, e in ev.iterrows():
        if e["event"] == "drain":
            ax.axvspan(e["from_epoch_end"], e["to_epoch_start"], color=C_DRAIN, alpha=SHADE_ALPHA,
                       lw=0, zorder=0)
        else:
            ax.axvspan(e["from_epoch_end"], e["to_epoch_start"], facecolor="none", edgecolor=C_FILL,
                       hatch="///", alpha=0.45, lw=0, zorder=0)
    full = ts["coverage"] >= EVENT_MIN_COVERAGE
    lo, hi = ylim
    ax.plot(ts.loc[full, "date"], ts.loc[full, "anomaly"], "-o", color=ps.INK, lw=0.8, ms=3.2,
            mec="white", mew=0.4, zorder=3)
    part = ts[~full]
    inside = part["anomaly"].between(lo, hi)
    ax.plot(part.loc[inside, "date"], part.loc[inside, "anomaly"], "o", mfc="white", mec=ps.INK,
            ms=3.2, mew=0.7, zorder=3)
    for sel, y, m in ((part["anomaly"] > hi, hi, "^"), (part["anomaly"] < lo, lo, "v")):
        ax.plot(part.loc[sel, "date"], np.full(int(sel.sum()), y), m, mfc="white", mec=ps.INK,
                ms=3.6, mew=0.7, zorder=4, clip_on=False)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color=ps.C_GRID, lw=0.6, zorder=1)
    ax.grid(axis="x", color=ps.C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    return ev


def legend_handles():
    return [Line2D([], [], color=ps.INK, marker="o", ms=3.2, lw=0.8, mec="white", mew=0.4,
                   label="strip (≥ 50 % of lake)"),
            Line2D([], [], color="none", marker="o", ms=3.2, mfc="white", mec=ps.INK, mew=0.7,
                   label="partial strip (arrow: off scale)"),
            Patch(facecolor=C_DRAIN, alpha=SHADE_ALPHA + 0.1, label="drain interval"),
            Patch(facecolor="none", edgecolor=C_FILL, hatch="///", label="fill interval")]


def bottom_legend(fig):
    """Legend in one row under the panels, so it never covers data."""
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4,
               frameon=False, fontsize=ps.SMALL_PT, handlelength=2.0, columnspacing=1.5)


def year_axis(ax, t0, t1):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(t0 - pd.Timedelta(days=90), t1 + pd.Timedelta(days=90))


def note_text(final):
    note = ("dh: lake-mean minus background-ring mean over the SD outline, relative to the lake's "
            "median.")
    if not final:
        note += "\nPROVISIONAL: some lakes are still on old-chain stacks (see fig4_provenance.csv)."
    return note


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", help="run folder names under volume/ (default: all, newest wins)")
    p.add_argument("--group_km", type=float, default=15.0, help="grouping distance between centroids")
    p.add_argument("--final", action="store_true", help="leave out the provisional note")
    a = p.parse_args()

    data = load(run_dirs(a.runs))
    if not data:
        raise SystemExit("No run folder has SD-outline series (timeseries_sd_*.csv); run 03_volume_batch.py")
    groups = lake_groups(list(data), a.group_km)
    print("groups:", groups)
    t0 = min(ts["date"].min() for ts, *_ in data.values())
    t1 = max(ts["date"].max() for ts, *_ in data.values())

    ps.apply()
    out_dir = os.path.join(VOLUME_OUTPUT_DIR, f"fig4_{datetime.now():%Y%m%d_%H%M}")
    os.makedirs(out_dir, exist_ok=True)
    event_rows, prov_rows = [], []

    # One figure per lake
    for lake, (ts, iv, qc, d) in data.items():
        fig, ax = plt.subplots(figsize=(ps.FIG_WIDTH_IN, PANEL_H_IN + 0.5))
        ev = draw_lake(ax, ts, iv, y_range([ts]))
        ax.set_ylabel("dh (m)")
        year_axis(ax, t0, t1)
        ps.figure_title(fig, lake)
        bottom_legend(fig)
        fig.text(0.01, -0.22, note_text(a.final), fontsize=ps.SMALL_PT - 1, color=ps.C_OTHER, va="top")
        ps.save(fig, os.path.join(out_dir, f"fig4_lake_{slug(lake)}"))
        plt.close(fig)
        group = next(i for i, g in enumerate(groups) if lake in g)
        for _, e in ev.iterrows():
            event_rows.append({"group": group, "lake_name": lake, "event": e["event"],
                               "from": e["from_epoch_end"].date(), "to": e["to_epoch_start"].date(),
                               "mean_dh_m": round(e["mean_dz_m"], 2),
                               "volume_Mm3": round(e["volume_m3"] / 1e6, 2),
                               "coverage": round(e["coverage"], 2)})
        prov_rows.append({"lake_name": lake, "group": group, "group_lakes": "; ".join(groups[group]),
                          "run_dir": os.path.basename(d), "stack": qc.get("stack"),
                          "lake_used": qc.get("lake_used"), "flags": qc.get("flags"),
                          "n_strips": len(ts), "n_events": len(ev)})

    # One figure per group of two or more lakes: stacked panels, shared time and dh axes
    for gi, lakes in enumerate(groups):
        if len(lakes) < 2:
            continue
        fig, axes = plt.subplots(len(lakes), 1, sharex=True, sharey=True,
                                 figsize=(ps.FIG_WIDTH_IN, PANEL_H_IN * len(lakes) + 0.5),
                                 gridspec_kw={"hspace": 0.35})
        ylim = y_range([data[lake][0] for lake in lakes])
        for k, (ax, lake) in enumerate(zip(axes, lakes)):
            ts, iv, _, _ = data[lake]
            draw_lake(ax, ts, iv, ylim)
            ps.panel_title(ax, chr(ord("a") + k), lake, pad=3)
        year_axis(axes[-1], t0, t1)
        bottom_legend(fig)
        fig.supylabel("dh (m)", fontsize=ps.LABEL_PT, x=0.04)
        ps.figure_title(fig, f"{', '.join(lakes)} (within {a.group_km:g} km)")
        fig.text(0.01, -0.06, note_text(a.final), fontsize=ps.SMALL_PT - 1, color=ps.C_OTHER, va="top")
        ps.save(fig, os.path.join(out_dir, f"fig4_group_{gi}"))
        plt.close(fig)

    pd.DataFrame(event_rows).to_csv(os.path.join(out_dir, "fig4_events.csv"), index=False)
    pd.DataFrame(prov_rows).to_csv(os.path.join(out_dir, "fig4_provenance.csv"), index=False)
    print(f"saved to {out_dir}: {len(data)} lake figures, "
          f"{sum(len(g) > 1 for g in groups)} group figures, fig4_events.csv, fig4_provenance.csv")


if __name__ == "__main__":
    main()
