# Elevation from DEMs tools

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tools for extracting elevation time series, transect profiles and DEM differences from the [ArcticDEM](https://www.pgc.umn.edu/data/arcticdem/) strip DEMs (SETSM s2s041, 2 m). They're built for glaciology in Greenland: lakes, outlet glaciers, surface change. Everything runs from Jupyter notebooks.

## What you can do

- **Elevation history** (`01_history_multimode.ipynb`): the elevation of one or more points through time, from every available strip.
- **Transect profiles** (`02_transect_multimode.ipynb`): elevation profiles along a line for every strip, overlaid and differenced.
- **Lake elevation and volume change** (`03_volume.ipynb`, *work in progress*): for a lake polygon, the lake surface relative to its surroundings through time, ice thickness and d/H, and the drain/fill events with their volumes. It reads the per-tile zarr stacks made by the GLOBE pipeline.
- **DEM differences** (`04_diffdems.ipynb`, `04_diffstrips.ipynb`, *experimental*): difference maps between two strips.

The tools search the PGC catalogue (STAC API) for every strip over your point or line and time range. They then read the strips from the CPOM archive on luna, either raw or coregistered (see below).

## Quick start (CPOM2)

You need an account on the CPOM2 virtual machine, with the luna share mounted at `~/luna`. The tools search the PGC catalogue online, so the VM also needs internet access.

```bash
# 1. Get the code
git clone https://github.com/dMoralPombo/elevation-tools.git
cd elevation-tools

# 2. Create the environment (once)
conda env create -f environment.yml
conda activate elevation-tools
#   or, in an existing environment: pip install -r requirements.txt

# 3. Check the paths
python -c "import config"
```

Step 3 prints the paths the tools will use, with a ✓ or a warning for each:

| Path | Default | Override with |
|---|---|---|
| luna CPOM folder | `~/luna/CPOM` | `LUNA_ROOT` |
| Strip archive | `<LUNA_ROOT>/archive/SATS/OPTICAL/ArcticDEM/strips/s2s041/2m/` | `ARCTICDEM_ARCHIVE_DIR` |
| Mosaic (v4.1, 2 m) | `<LUNA_ROOT>/archive/SATS/OPTICAL/ArcticDEM/mosaic/v4.1/2m/` | `ARCTICDEM_MOSAIC_DIR` |
| Mosaic index | `<LUNA_ROOT>/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM_Mosaic_Index_latest_shp/` | `ARCTICDEM_MOSAIC_INDEX_DIR` |
| Your outputs | `<LUNA_ROOT>/<your username>/elevation_tools_outputs/` | `ARCTICDEM_OUTPUT_DIR` |

If luna is mounted somewhere else, set `LUNA_ROOT` before starting Jupyter. All the defaults follow it:

```bash
export LUNA_ROOT=/path/to/luna/CPOM
```

The output folder is created the first time you import the tools.

Then open a notebook in `notebooks/`, in Jupyter or in the VSCode Interactive window, and run the cells in order.

### Lake volume notebook (03)

`03_volume.ipynb` works on one lake at a time, from a polygon file (one polygon per lake). Its default file holds the current candidate lakes. The steps:
1. **Stack:** it lists the tile's zarr stacks and picks one. Rebuilt stacks (`_rs` in the folder name) come first; set `REFERENCE` to `'cs2'` or `'mosaic'` when a tile has both.
2. **Time series:** lake-mean elevation minus a background ring, for every strip.
3. **Ice thickness:** BedMachine thickness and the lake's d/H.
4. **Outline and events** (optional, a few minutes): an SD-based outline, and the drain/fill events between observations with their volumes.

Outputs go to `OUTPUT_DIR/volume/notebook_<lake>_<date>/`. Results from stacks not yet rebuilt (no `_rs` in the name) are provisional.

## How the notebooks work

Every notebook has the same layout:

1. **Setup:** imports the modules from the repository root. Re-running this cell picks up code changes without restarting the kernel.
2. **Parameters:** everything you normally change is here.
3. **Select the location**, with one of the input methods:
   - **Interactive map:** drag the marker(s) and click **Confirm**, then **run the next cell**. The map only passes your choice on after its cell has finished.
   - **Manual:** type `lon,lat` (or `lon1,lat1,lon2,lat2` for a transect).
   - **File:** a GeoPackage or GeoParquet with many points or transects, run as a batch.
   - **Auto-orthogonal** (02 only): two crossing transects centred on a point.
4. **Run**, then **view the results.**

Cells after "View results" (temporal analysis, lake detection, optional batches) are experiments and may change.

### Main parameters

| Parameter | Meaning |
|---|---|
| `TIME_RANGE` | `"YYYY-MM-DD/YYYY-MM-DD"`. ArcticDEM strips span 2009 to the present. |
| `COREG_MODE` | `'none'`, `'altim'` or `'mosaic'` (next section). |
| `WINDOW_SIZE`, `WINDOW_TYPE` (01) | Pixels averaged around each point: `3` = 3×3. `'square'` = full window, `'cross'` = centre row and column. The spread across the window is reported as the uncertainty. |
| `MAX_DEMS` (02) | Cap on the number of strips; `'none'` mode is slower, so it's capped lower. |
| `MAX_CLOUD_COVER` | Drop strips whose catalogue cloud fraction (0–1) is above this. |
| `LAKE_NAME` | Optional label for plot titles. |

## Coregistration modes

Each strip in the archive exists raw, and some also exist coregistered (aligned to a reference surface). `COREG_MODE` chooses which one is read:

| Mode | File read | Notes |
|---|---|---|
| `none` | the raw strip `.tar.gz`, read in memory | Every strip is available, but each can be offset by metres. Cloud, water and edge pixels are removed using the strip's bitmask. Slowest mode. |
| `altim` | the strip coregistered to CryoSat-2 altimetry | Works inland. Over gentle, sparsely sampled ice the horizontal fit can fail, so check large outliers. |
| `mosaic` | the strip coregistered to the ArcticDEM mosaic | Uses control points on **exposed bedrock** only, so it's reliable near the ice margin and unavailable far inland. |

A strip that hasn't been coregistered in the chosen mode is skipped. The exact coregistered files used are set by `COREG_PARAMS` in `config.py`.

## Outputs

Everything is written under your `OUTPUT_DIR`:

```
elevation_histories/    elevation_history_<lon>_<lat>_<mode>_<years>.png / .txt
transects/              one profile plot per strip
transects_combined/     combined profiles (.txt) and plots per transect
```

File names carry the coordinates, the coregistration mode and the years of `TIME_RANGE`. The `.txt` files hold the data behind each plot.

## Tips

- **Start small:** a short `TIME_RANGE` or a low `MAX_DEMS` first, then widen.
- **Catalogue search is online.** If PGC's STAC API is slow or down, the search step fails; just re-run it.
- **Coordinates** are `lon, lat` in WGS84. Everything is processed in EPSG:3413 (NSIDC Polar Stereographic North), so it's Greenland/Arctic only.
- **Never unpack or edit files in the archive.** The tools read the tarballs in memory.

## Repository layout

```
config.py              paths, catalogue and coregistration settings
elevation_utils.py     catalogue search, strip lookup and reading, window sampling, transects
analysis.py            elevation-history and transect workflows, plots
interactive_maps.py    the ipyleaflet point/transect pickers
lake_volume.py         lake time series, outlines, epoch differences and volumes from the zarr stacks
lake_detection.py      experimental lake-boundary detection from transects
diff_analysis*.py      DEM differencing (zarr-based and GeoTIFF-based)
notebooks/             the notebooks (old/ holds retired versions);
                       03_volume_batch.py runs the volume steps for every lake
```

## Limitations

- Greenland/Arctic only (EPSG:3413), and only the `arcticdem-strips-s2s041-2m` collection.
- The paths assume the CPOM luna layout. Elsewhere, set the environment variables above.
- Reading many strips is slow and memory-hungry; `'none'` mode most of all.

## Citation

If you use these tools, please cite the data:

- Porter, C., et al. (2023). ArcticDEM – Strips, Version 4.1. Harvard Dataverse. https://doi.org/10.7910/DVN/OHHUKH

## Contact

Diego Moral Pombo, Lancaster University: d.moralpombo@lancaster.ac.uk (GitHub: @dMoralPombo)

## Acknowledgements

- ArcticDEM data provided by the Polar Geospatial Center (PGC); catalogue via https://stac.pgc.umn.edu/
- Coregistered strips from the DEMSquad / GLOBE processing pipeline (thanks, DEMSquad!)

## License

MIT
