"""
Configuration for Elevation From DEMs
==================================
Edit the paths below to match your system setup.
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Main archive directory for ArcticDEM strip files
ARCHIVE_DIR = os.getenv(
    'ARCTICDEM_ARCHIVE_DIR',
    "~/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/strips/s2s041/2m/"
)

# Mosaic directory
MOSAIC_DIR = os.getenv(
    'ARCTICDEM_MOSAIC_DIR',
    "~/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/mosaic/v4.1/2m/"
)

# Mosaic index shapefile directory
MOSAIC_INDEX_DIR = os.getenv(
    'ARCTICDEM_MOSAIC_INDEX_DIR',
    "~/luna/CPOM/archive/SATS/OPTICAL/ArcticDEM/ArcticDEM_Mosaic_Index_latest_shp/"
)

# Output directory for results
OUTPUT_DIR = os.getenv(
    'ARCTICDEM_OUTPUT_DIR',
    "~/luna/CPOM/{your_username}/{whatever_folder}/"
)

# ============================================================================
# API CONFIGURATION
# ============================================================================

STAC_API_URL = "https://stac.pgc.umn.edu/api/v1/"
COLLECTION_ID = "arcticdem-strips-s2s041-2m"

# ============================================================================
# COORDINATE SYSTEMS
# ============================================================================

DEFAULT_CRS = "EPSG:3413"      # Arctic Polar Stereographic
GEOGRAPHIC_CRS = "EPSG:4326"   # WGS84

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

DEFAULT_NUM_SAMPLES = 100      # Points along transect
DEFAULT_WINDOW_SIZE = 3        # Window size for elevation extraction
DEFAULT_WINDOW_TYPE = 'square' # 'square' or 'cross'
MAX_CLOUD_COVER = 0.2          # Maximum cloud area percentage (0-1)

# ============================================================================
# DEFAULT REGIONS
# ============================================================================

DEFAULT_BBOX = (-54.3, 74.9, -54.1, 75.1)  # (West, South, East, North)
DEFAULT_TIME_RANGE = "2010-01-01/2026-12-31"

# ============================================================================
# COREGISTRATION PARAMETERS
# ============================================================================

COREG_PARAMS = {
    'altim': {
        'reference_data': 'cs2',
        'coreg_choice': 'vertical_offset_mean nuthkaab deramp',
        'filter_vel': '999',
        'filter_dhdt': '999',
    },
    'mosaic': {
        'reference_data': 'mosaic',
        'coreg_choice': 'vertical_offset_mean nuthkaab deramp',
        'filter_vel': '0',
        'filter_dhdt': '0',
    },
    'none': {}
}

# ============================================================================
# VALIDATION AND SETUP
# ============================================================================

def validate_config():
    """Check if configured paths exist and warn if not."""
    paths_to_check = {
        'ARCHIVE_DIR': ARCHIVE_DIR,
        'MOSAIC_DIR': MOSAIC_DIR,
        'MOSAIC_INDEX_DIR': MOSAIC_INDEX_DIR,
    }
    
    print("\n" + "="*60)
    print("Elevation from DEMs Configuration")
    print("="*60)
    
    for name, path in paths_to_check.items():
        if 'REPLACE_WITH' in path:
            print(f"⚠️  WARNING: {name} not configured!")
            print(f"   Please edit config.py or set the {name} environment variable")
        elif not os.path.exists(path):
            print(f"⚠️  WARNING: {name} does not exist: {path}")
        else:
            print(f"✓ {name}: {path}")
    
    print(f"✓ OUTPUT_DIR: {OUTPUT_DIR}")
    print("="*60 + "\n")

def create_output_directories():
    """Create output subdirectories if they don't exist."""
    subdirs = ['transects', 'transects_combined', 'elevation_histories']
    for subdir in subdirs:
        Path(OUTPUT_DIR, subdir).mkdir(parents=True, exist_ok=True)

def get_output_path(subdirectory, filename):
    """Generate a full output path and create directories if needed."""
    full_path = Path(OUTPUT_DIR) / subdirectory / filename
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return str(full_path)

def get_mosaic_index_path():
    """Get the full path to the mosaic index shapefile."""
    return os.path.join(MOSAIC_INDEX_DIR, "ArcticDEM_Mosaic_Index_v4_1_2m.shp")

# Run validation on import
create_output_directories()
validate_config()
