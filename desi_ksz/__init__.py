"""
DESI DR1 Pairwise kSZ + kSZ Tomography Analysis Pipeline.

This package implements a publication-grade analysis pipeline for measuring
the kinetic Sunyaev-Zel'dovich (kSZ) effect using DESI DR1 spectroscopic
galaxy catalogs cross-correlated with CMB temperature maps.

Main Components
---------------
io : Data input/output
    - desi_catalogs: DESI LSS catalog loading (BGS, LRG, ELG)
    - cmb_maps: CMB temperature map handling (ACT DR6, Planck PR4)
    - download: Data download utilities

selection : Catalog selection and weighting
    - quality_cuts: Galaxy quality cuts
    - redshift_bins: Tomographic redshift binning
    - weights: Systematic weight computation

maps : CMB map processing
    - healpix_ops: HEALPix operations
    - filtering: Optimal filtering (matched, Wiener)
    - masking: Point source, cluster, Galactic masks

estimators : kSZ signal estimation
    - aperture_photometry: Aperture photometry stacking
    - pairwise_momentum: Pairwise kSZ momentum estimator
    - pair_counting: Efficient pair counting with KDTree
    - theory_template: Linear theory predictions

covariance : Uncertainty estimation
    - jackknife: Spatial jackknife resampling
    - hartlap: Hartlap correction for precision matrix

systematics : Systematic error tests
    - null_tests: Comprehensive null test suite
    - tsz_leakage: tSZ contamination diagnostics

inference : Parameter inference
    - likelihood: Gaussian likelihood
    - mcmc: MCMC sampling with emcee

Example Usage
-------------
>>> from desi_ksz.io import DESIGalaxyCatalog, PlanckMap
>>> from desi_ksz.estimators import PairwiseMomentumEstimator
>>>
>>> # Load data
>>> catalog = DESIGalaxyCatalog('LRG', data_dir='data/ksz/catalogs/')
>>> cmb_map = PlanckMap('commander', data_dir='data/ksz/maps/')
>>>
>>> # Compute pairwise momentum
>>> estimator = PairwiseMomentumEstimator(separation_bins=np.linspace(5, 150, 15))
>>> result = estimator.compute(catalog, cmb_map)

References
----------
- Ferreira, P. G., Juszkiewicz, R., Feldman, H. A., Davis, M., & Jaffe, A. H.
  1999, ApJ, 515, L1 (Original pairwise momentum estimator)
- Hand, N., et al. 2012, PRL, 109, 041101 (First kSZ detection)
- Schaan, E., et al. 2021, PRD, 103, 063513 (ACT kSZ analysis)
"""

__version__ = "0.1.0"
__author__ = "DESI kSZ Analysis Team"

from . import config
from . import io
from . import selection
from . import maps
from . import estimators
from . import covariance
from . import inference
from . import systematics
from . import sims
from . import plots
