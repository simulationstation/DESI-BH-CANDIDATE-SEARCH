"""
Data I/O module for kSZ analysis pipeline.

Provides classes and functions for loading DESI galaxy catalogs,
CMB temperature maps, and managing data caching.
"""

from .desi_catalogs import (
    DESIGalaxyCatalog,
    load_desi_catalog,
    load_desi_randoms,
)
from .cmb_maps import (
    CMBTemperatureMap,
    PlanckMap,
    ACTMap,
    load_cmb_map,
)
from .download import (
    download_desi_lss,
    download_act_maps,
    download_planck_maps,
    DATA_MANIFEST,
)
from .cache import (
    CacheManager,
    cache_to_hdf5,
    load_from_hdf5,
)

__all__ = [
    # DESI catalogs
    "DESIGalaxyCatalog",
    "load_desi_catalog",
    "load_desi_randoms",
    # CMB maps
    "CMBTemperatureMap",
    "PlanckMap",
    "ACTMap",
    "load_cmb_map",
    # Download
    "download_desi_lss",
    "download_act_maps",
    "download_planck_maps",
    "DATA_MANIFEST",
    # Cache
    "CacheManager",
    "cache_to_hdf5",
    "load_from_hdf5",
]
