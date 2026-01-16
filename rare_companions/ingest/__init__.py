"""
Data ingestion module for rare_companions pipeline.
"""

from .desi import load_desi_rv_epochs, DESIRVLoader
from .lamost import load_lamost_epochs, LAMOSTLoader
from .unified import RVTimeSeries, UnifiedRVLoader, merge_rv_sources
