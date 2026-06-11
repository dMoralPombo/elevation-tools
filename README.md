# Elevation from DEMs Tools

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Tools for extracting and analyzing elevation profiles and time series from the [ArcticDEM](https://www.pgc.umn.edu/data/arcticdem/) strip dataset. Designed for glaciology >

## Features

- **Elevation History**: Extract elevation time series at a point across hundreds of DEMs
- **Transect Profiles** (⚠️ under development): Extract and compare elevation profiles along a transect
- **Interactive Maps**: Select coordinates using draggable markers on satellite imagery
- **Window-based Sampling**: Robust elevation extraction using configurable pixel windows
- **Multiple Coregistration Modes**: Support for raw, altimetry-coregistered, and mosaic-coregistered DEMs
- **Automatic STAC Queries**: Search the ArcticDEM catalog by location and time range

# Installation

## Clone the repository
git clone https://github.com/dMoralPombo/elevation-tools.git
cd elevation_tools

## Install dependencies
pip install -r requirements.txt

# Configuration

Before using, edit `config.py` to set your data paths:

### Edit config.py
nano config_local.py  # Edit the paths
ARCHIVE_DIR = "/your/archive/path/"
MOSAIC_DIR = "/your/mosaic/path/"

## Data Requirements
You need access to:

ArcticDEM Strip Archive: The s2s041/2m/ directory containing SETSM DEM files (.tif or .tar.gz)

Mosaic Index Shapefile (optional): ArcticDEM_Mosaic_Index_v4_1_2m.shp for mosaic tile lookup

The tools access the PGC STAC API to discover available DEMs, so an internet connection is required for the catalog search 
(at least by now until we use a Lancaster/CPOM-based catalogue).

# Usage

## Jupyter Notebook (Recommended)
The primary interface is through Jupyter notebooks:

jupyter notebook

Then open one of the example notebooks:

✅ Elevation History Analysis (notebooks/01_elevation_history.ipynb)
This is the stable and fully functional notebook. Use it to:

1. Select a point on an interactive satellite map

2. Search the PGC-API ArcticDEM catalog for all available DEMs at that location

3. Extract elevations using a configurable window for uncertainties (3x3, 5x5, cross-pattern, etc.)

4. Plot elevation time series with error bars and trend lines

5. Export data as CSV and figures

## Key parameters:

WINDOW_SIZE: Number of pixels for elevation averaging (3 = 3x3 window)

WINDOW_TYPE: 'square' for full window, 'cross' for plus-sign pattern

COREG_MODE: 'none' (raw DEMs), 'altim' (CryoSat-2 coregistered), 'mosaic' (mosaic-coregistered)

TIME_RANGE: Date range in format "YYYY-MM-DD/YYYY-MM-DD" (total PGC range: 2009-2025)

## Output Structure

outputs/

├── elevation_histories/

│   ├── elevation_history_-54.200_75.000_nc_2010-2020.png   # Plot

│   └── elevation_history_-54.200_75.000_nc_2010-2020.txt   # Data

├── transects/

│   └── profile_SETSM_20150701_...png                         # Individual profiles

└── transects_combined/

    ├── combined_profiles_-54.200_75.000_...txt               # Combined data
    
    └── profile_-54.200_75.000_...png                         # Combined plot

## Coregistration Modes

Mode	Description	File Pattern	Use Case
none	Raw compressed DEMs	*_dem.tif.gz or *.tar.gz	Original SETSM elevations
altim	CryoSat-2 coregistered	*cs2*coregistered.tif	Corrected to altimetry
mosaic	Mosaic-coregistered	*mosaic*coregistered.tif	Corrected to ArcticDEM mosaic

Coregistration parameters are configured in config.py under COREG_PARAMS.

## Window Types

The elevation extraction uses a window of pixels around the target coordinates:

-Square (window_type='square'): Full N×N window averaging

-Cross (window_type='cross'): Plus-sign pattern (center row + center column)

Window size examples:

-window_size=1: Single pixel (no averaging)

-window_size=3: 3×3 window (9 pixels) or cross of 5 pixels

-window_size=5: 5×5 window (25 pixels) or cross of 9 pixels

Larger windows provide more robust elevation estimates with uncertainty (standard deviation).

## Limitations

-Coordinate System: Currently hardcoded for Greenland (EPSG:3413). For other Arctic regions, modify the CRS in config.py.

-DEM Source: Only supports the arcticdem-strips-s2s041-2m collection from PGC.

-File Format: Assumes SETSM v4.1 naming conventions (SETSM_s2s041_<pairname>_2m_lsf_seg1_dem.tif).

-Internet Required: STAC catalog queries need internet access.

-Memory: Reading multiple DEMs can be memory-intensive for large transects.

# Contributing

This tool is under active development. If you encounter issues or have suggestions:

1. Check the GitHub Issues

2. Create a new issue with:

-Description of the problem

-Steps to reproduce

-Your environment (Python version, operating system)

-Any error messages

# License

MIT License - see LICENSE file for details.

# Citation

If you use this tool in your research, please cite:

-Porter, C., et al. (2023). ArcticDEM. Harvard Dataverse. https://doi.org/10.7910/DVN/OHHUKH

# Contact

-Author: Diego Moral Pombo

-Email: d.moralpombo@lancaster.ac.uk

-GitHub: @dMoralPombo

# Acknowledgments

-ArcticDEM data provided by the Polar Geospatial Center (PGC)

-STAC API access via https://stac.pgc.umn.edu/

-Built with inspiration from the GLOBE processing pipeline (thanks DEMSquad!)
