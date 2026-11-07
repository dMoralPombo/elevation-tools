"""
All lakes, as one long run: for every lake polygon in the candidates file,
the time series, outlines, epoch differences, contour comparison, ice
thickness and QC table, written to OUTPUT_DIR/volume/persistence_<YYYYMMDD_HHMM>/.

The functions live in lake_volume.py (repo root); notebooks/03_volume.ipynb
runs the same steps for one lake. Run this as a systemd user service (see
CLAUDE.md); VOLUME_LAKES="Candidate #8;Flade Isblink" runs a subset.
"""
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(".."))
from config import LUNA_ROOT, get_mosaic_index_path  # noqa: E402
from lake_volume import *  # noqa: E402,F401,F403
from lake_volume import _append_gpkg_layer  # noqa: E402

# ===========================================================================
# Example end-to-end usage
# ===========================================================================
if __name__ == "__main__":
    # polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_august26.gpkg"
    # polygon_path = "/home/moralpom/luna/CPOM/moralpom/globe/data/ArcticDEM/candidates_polygons_01sept26.gpkg"
    polygon_path = os.path.join(LUNA_ROOT, "moralpom", "globe", "data", "ArcticDEM",
                                "candidates_polygons_02sept26.gpkg")
    index_path = get_mosaic_index_path()
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
    # Coregistration reference per tile, where a tile has both cs2 and mosaic
    # stacks (the user's rule: mosaic where there's enough bedrock nearby)
    TILE_REFERENCE = {"11_38_2_2": "mosaic", "12_38_1_2": "mosaic", "12_38_2_2": "mosaic"}
    DEFAULT_REFERENCE = "cs2"
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
                return sum(map(len, build_strip_index(
                    subtile[:9], reference=TILE_REFERENCE.get(subtile[:9], DEFAULT_REFERENCE)).values()))
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
                subfolder = select_preferred_subfolder(
                    tile_id, reference=TILE_REFERENCE.get(tile_id, DEFAULT_REFERENCE))
                strip_index = build_strip_index(tile_id, subfolder=subfolder)
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
                strips_df, gaps_df = coverage_gap_report(lake_gdf, tile_id, strip_index, diag,
                                                         subfolder=subfolder)
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
                            "stack": subfolder,
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
