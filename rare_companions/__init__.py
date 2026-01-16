"""
rare_companions - Multi-experiment search framework for rare binaries and compact-object systems

This package provides a scalable pipeline for identifying candidate compact companions
(neutron stars, black holes, white dwarfs) and other rare binary systems using public
survey data from DESI, Gaia, LAMOST, WISE, GALEX, TESS, ZTF, and X-ray/radio catalogs.

All results are CANDIDATES requiring follow-up confirmation. No dynamical masses are
measured in this triage pipeline.
"""

__version__ = "1.0.0"
__author__ = "DESI BH Candidate Search Team"

from .config import Config, load_config
