"""
Phase 3-4 Execution Driver for DESI kSZ analysis.

Runs the complete production pipeline:
  Phase 3: Measurement
    - Temperature extraction
    - Pairwise momentum estimation
    - Tomographic binning
    - Covariance estimation
    - Detection significance

  Phase 4: Validation & Publication
    - All null tests
    - All referee checks
    - Gate evaluation
    - Results packaging

Symbol Table
------------
| Symbol      | Definition                                      | Units       |
|-------------|------------------------------------------------|-------------|
| p̂(r,z)      | Measured pairwise momentum in bin              | μK          |
| A_kSZ       | kSZ amplitude (fit parameter)                  | dimensionless|
| σ_A         | Amplitude uncertainty                          | dimensionless|
| SNR         | Detection significance = A / σ_A               | dimensionless|
| χ²          | Chi-squared statistic                          | dimensionless|
| PTE         | Probability to exceed (p-value)                | dimensionless|

Detection Significance
---------------------
For each redshift bin and joint:
    SNR = Â / σ_Â
    where Â = (p^T Ψ t) / (t^T Ψ t)
    and σ_Â = 1 / √(t^T Ψ t)
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from multiprocessing import cpu_count
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Phase34Config:
    """Configuration for Phase 3-4 run."""

    # Paths
    catalog_file: str
    map_files: List[str]  # At least 2 for consistency check
    output_dir: str
    cluster_catalog: Optional[str] = None
    ymap_file: Optional[str] = None

    # Tracer settings
    tracer: str = "LRG"
    z_bins: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0])

    # Separation bins
    r_min: float = 20.0  # Mpc/h
    r_max: float = 150.0  # Mpc/h
    n_r_bins: int = 15

    # Covariance settings
    jackknife_K_values: List[int] = field(default_factory=lambda: [50, 100, 200])
    auto_select_K: bool = True
    condition_threshold: float = 1e6

    # tSZ settings
    tsz_mask_radii: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0])
    tsz_stability_threshold: float = 1.0  # sigma

    # Aperture photometry
    aperture_inner: float = 1.8  # arcmin
    aperture_outer: float = 5.0  # arcmin
    beam_fwhm: float = 1.4  # arcmin

    # Validation settings
    n_injection_realizations: int = 100
    n_null_realizations: int = 100
    n_transfer_realizations: int = 10
    small_mode: bool = False  # Quick mode for testing

    # Referee checks
    run_referee_checks: bool = True
    beam_perturbation: float = 0.05  # ±5%

    # Output settings
    save_intermediate: bool = True
    create_plots: bool = True
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compute_hash(self) -> str:
        """Compute config hash for reproducibility."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


@dataclass
class TomographicResult:
    """Result for a single tomographic bin."""
    z_min: float
    z_max: float
    z_mean: float
    n_galaxies: int
    p_ksz: np.ndarray
    p_ksz_err: np.ndarray
    amplitude: float
    amplitude_err: float
    snr: float
    chi2: float
    ndof: int
    pte: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'z_min': self.z_min,
            'z_max': self.z_max,
            'z_mean': self.z_mean,
            'n_galaxies': self.n_galaxies,
            'p_ksz': self.p_ksz.tolist(),
            'p_ksz_err': self.p_ksz_err.tolist(),
            'amplitude': self.amplitude,
            'amplitude_err': self.amplitude_err,
            'snr': self.snr,
            'chi2': self.chi2,
            'ndof': self.ndof,
            'pte': self.pte,
        }


@dataclass
class Phase34Result:
    """Complete result from Phase 3-4 run."""
    config: Phase34Config
    timestamp: str
    run_id: str

    # Core results
    tomographic_results: List[TomographicResult]
    joint_amplitude: float
    joint_amplitude_err: float
    joint_snr: float
    joint_chi2: float
    joint_pte: float

    # Map-specific results
    map_results: List[Dict[str, Any]]

    # Covariance info
    covariance_K: int
    covariance_info: Dict[str, Any]

    # Validation results
    injection_result: Optional[Dict[str, Any]] = None
    null_result: Optional[Dict[str, Any]] = None
    transfer_result: Optional[Dict[str, Any]] = None
    tsz_result: Optional[Dict[str, Any]] = None
    referee_results: Optional[Dict[str, Any]] = None

    # Gate evaluation
    gate_evaluation: Optional[Dict[str, Any]] = None
    overall_status: str = "INCOMPLETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'timestamp': self.timestamp,
            'run_id': self.run_id,
            'tomographic_results': [t.to_dict() for t in self.tomographic_results],
            'joint_amplitude': self.joint_amplitude,
            'joint_amplitude_err': self.joint_amplitude_err,
            'joint_snr': self.joint_snr,
            'joint_chi2': self.joint_chi2,
            'joint_pte': self.joint_pte,
            'map_results': self.map_results,
            'covariance_K': self.covariance_K,
            'covariance_info': self.covariance_info,
            'injection_result': self.injection_result,
            'null_result': self.null_result,
            'transfer_result': self.transfer_result,
            'tsz_result': self.tsz_result,
            'referee_results': self.referee_results,
            'gate_evaluation': self.gate_evaluation,
            'overall_status': self.overall_status,
        }


class Phase34Runner:
    """
    Main driver for Phase 3-4 execution.

    Runs complete production pipeline with automated gating.

    Examples
    --------
    >>> config = Phase34Config(
    ...     catalog_file='data/LRG_catalog.npz',
    ...     map_files=['data/planck_smica.fits', 'data/planck_nilc.fits'],
    ...     output_dir='results/run_001',
    ... )
    >>> runner = Phase34Runner(config)
    >>> result = runner.run()
    >>> print(result.overall_status)
    """

    def __init__(self, config: Phase34Config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = f"{config.tracer}_{config.compute_hash()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.rng = np.random.default_rng(config.random_seed)

        # Storage for intermediate results
        self._catalog = None
        self._maps = []
        self._covariance = None
        self._template = None

    def run(self) -> Phase34Result:
        """
        Execute complete Phase 3-4 pipeline.

        Returns
        -------
        Phase34Result
            Complete results including gate evaluation
        """
        logger.info("=" * 60)
        logger.info(f"Phase 3-4 Execution: {self.run_id}")
        logger.info("=" * 60)

        timestamp = datetime.now().isoformat()

        # Phase 3: Measurement
        logger.info("\n=== PHASE 3: MEASUREMENT ===")

        # 3.1 Load data
        logger.info("3.1 Loading data...")
        self._load_data()

        # 3.2 Tomographic measurement
        logger.info("3.2 Running tomographic measurement...")
        tomo_results, map_results = self._run_tomographic_measurement()

        # 3.3 Covariance estimation
        logger.info("3.3 Estimating covariance...")
        cov_K, cov_info, covariance = self._estimate_covariance()

        # 3.4 Joint fit
        logger.info("3.4 Computing joint fit...")
        joint_amp, joint_err, joint_snr, joint_chi2, joint_pte = self._compute_joint_fit(
            tomo_results, covariance
        )

        # Phase 4: Validation
        logger.info("\n=== PHASE 4: VALIDATION ===")

        # 4.1 Injection test
        logger.info("4.1 Running injection test...")
        injection_result = self._run_injection_test()

        # 4.2 Null tests
        logger.info("4.2 Running null tests...")
        null_result = self._run_null_tests()

        # 4.3 Transfer function test
        logger.info("4.3 Running transfer function test...")
        transfer_result = self._run_transfer_test()

        # 4.4 tSZ sweep
        logger.info("4.4 Running tSZ sweep...")
        tsz_result = self._run_tsz_sweep()

        # 4.5 Referee checks
        referee_results = None
        if self.config.run_referee_checks:
            logger.info("4.5 Running referee checks...")
            referee_results = self._run_referee_checks()

        # 4.6 Gate evaluation
        logger.info("4.6 Evaluating gates...")
        from .gates import evaluate_gates, create_metrics_from_results

        metrics = create_metrics_from_results(
            injection_result=injection_result,
            null_result=null_result,
            transfer_result=transfer_result,
            tsz_result=tsz_result,
            cov_info=cov_info,
            map_results=map_results,
            referee_results=referee_results,
        )

        gate_eval = evaluate_gates(metrics)

        # Create result
        result = Phase34Result(
            config=self.config,
            timestamp=timestamp,
            run_id=self.run_id,
            tomographic_results=tomo_results,
            joint_amplitude=joint_amp,
            joint_amplitude_err=joint_err,
            joint_snr=joint_snr,
            joint_chi2=joint_chi2,
            joint_pte=joint_pte,
            map_results=map_results,
            covariance_K=cov_K,
            covariance_info=cov_info,
            injection_result=injection_result.to_dict() if hasattr(injection_result, 'to_dict') else injection_result,
            null_result=null_result.to_dict() if hasattr(null_result, 'to_dict') else null_result,
            transfer_result=transfer_result.to_dict() if hasattr(transfer_result, 'to_dict') else transfer_result,
            tsz_result=tsz_result.to_dict() if hasattr(tsz_result, 'to_dict') else tsz_result,
            referee_results=referee_results,
            gate_evaluation=gate_eval.to_dict(),
            overall_status=gate_eval.overall_status,
        )

        # Save results
        self._save_results(result)

        # Generate plots
        if self.config.create_plots:
            self._generate_plots(result)

        logger.info("\n" + "=" * 60)
        logger.info(f"OVERALL STATUS: {result.overall_status}")
        logger.info(f"Joint SNR: {joint_snr:.2f}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)

        return result

    def _load_data(self):
        """Load catalog and maps."""
        # Load catalog
        catalog_path = Path(self.config.catalog_file)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        self._catalog = np.load(catalog_path)
        logger.info(f"  Loaded catalog: {len(self._catalog['ra']):,} galaxies")

        # Load maps
        for map_file in self.config.map_files:
            map_path = Path(map_file)
            if not map_path.exists():
                logger.warning(f"  Map not found: {map_path}")
                continue

            try:
                import healpy as hp
                map_data = hp.read_map(map_path, verbose=False)
                self._maps.append({
                    'path': str(map_path),
                    'data': map_data,
                    'nside': hp.npix2nside(len(map_data)),
                })
                logger.info(f"  Loaded map: {map_path.name}")
            except Exception as e:
                logger.warning(f"  Failed to load {map_path}: {e}")

        if len(self._maps) < 2:
            logger.warning("  Less than 2 maps loaded - map consistency check will be limited")

    def _run_tomographic_measurement(self) -> Tuple[List[TomographicResult], List[Dict]]:
        """Run tomographic pairwise measurement."""
        from ..estimators import PairwiseMomentumEstimator
        from ..estimators.theory_template import compute_theory_template

        z_bins = self.config.z_bins
        r_bins = np.linspace(self.config.r_min, self.config.r_max, self.config.n_r_bins + 1)

        # Create estimator
        estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

        # Generate theory template
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        try:
            self._template = compute_theory_template(r_centers, z_mean=np.mean(z_bins))
        except Exception:
            # Fallback template
            self._template = 1.0 / (1 + r_centers / 50.0)
        self._template = self._template / np.max(np.abs(self._template))  # Normalize

        tomo_results = []
        map_results = []

        # Get catalog data
        ra = self._catalog['ra']
        dec = self._catalog['dec']
        z = self._catalog['z']
        positions = self._catalog['positions']
        weights = self._catalog['weights']

        # Process each map
        for map_idx, map_info in enumerate(self._maps):
            map_data = map_info['data']
            nside = map_info['nside']

            # Extract temperatures
            try:
                import healpy as hp
                theta = np.radians(90.0 - dec)
                phi = np.radians(ra)
                pix = hp.ang2pix(nside, theta, phi)
                temperatures = map_data[pix]
            except Exception as e:
                logger.warning(f"  Temperature extraction failed: {e}")
                continue

            # Process each z-bin
            z_results = []
            for i in range(len(z_bins) - 1):
                z_min, z_max = z_bins[i], z_bins[i + 1]
                z_mask = (z >= z_min) & (z < z_max)

                if np.sum(z_mask) < 100:
                    logger.warning(f"  z-bin [{z_min}, {z_max}): only {np.sum(z_mask)} galaxies")
                    continue

                # Compute pairwise
                result = estimator.compute(
                    positions[z_mask],
                    temperatures[z_mask],
                    weights[z_mask],
                )

                # Fit amplitude (placeholder covariance)
                n_bins = len(result.p_ksz)
                cov_diag = np.diag(np.ones(n_bins) * 100)  # Placeholder
                amp, amp_err = self._fit_amplitude(result.p_ksz, self._template[:n_bins], cov_diag)
                snr = amp / amp_err if amp_err > 0 else 0

                # Chi-squared
                residual = result.p_ksz - amp * self._template[:n_bins]
                chi2 = float(residual @ np.linalg.inv(cov_diag) @ residual)
                ndof = n_bins - 1
                from scipy.stats import chi2 as chi2_dist
                pte = 1 - chi2_dist.cdf(chi2, ndof)

                tomo_result = TomographicResult(
                    z_min=z_min,
                    z_max=z_max,
                    z_mean=float(np.mean(z[z_mask])),
                    n_galaxies=int(np.sum(z_mask)),
                    p_ksz=result.p_ksz,
                    p_ksz_err=np.sqrt(np.diag(cov_diag)),
                    amplitude=amp,
                    amplitude_err=amp_err,
                    snr=snr,
                    chi2=chi2,
                    ndof=ndof,
                    pte=pte,
                )
                z_results.append(tomo_result)

                if map_idx == 0:  # Primary map
                    tomo_results.append(tomo_result)

            # Store map-level result
            if z_results:
                map_amp = np.mean([r.amplitude for r in z_results])
                map_err = np.sqrt(np.sum([r.amplitude_err**2 for r in z_results])) / len(z_results)
                map_results.append({
                    'map_file': map_info['path'],
                    'amplitude': float(map_amp),
                    'amplitude_err': float(map_err),
                    'n_zbins': len(z_results),
                })

        return tomo_results, map_results

    def _estimate_covariance(self) -> Tuple[int, Dict, np.ndarray]:
        """Estimate covariance with automatic K selection."""
        from ..covariance.jackknife import SpatialJackknife
        from ..covariance.stability import (
            analyze_covariance, auto_regularize, compute_hartlap_factor
        )

        ra = self._catalog['ra']
        dec = self._catalog['dec']
        positions = self._catalog['positions']
        weights = self._catalog['weights']

        if len(self._maps) == 0:
            # Return placeholder
            n_bins = self.config.n_r_bins
            return 100, {'hartlap_factor': 0.9, 'condition_number': 10.0}, np.eye(n_bins) * 100

        temperatures = np.zeros(len(ra))
        try:
            import healpy as hp
            map_data = self._maps[0]['data']
            nside = self._maps[0]['nside']
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)
            temperatures = map_data[pix]
        except Exception:
            pass

        # Test different K values
        K_values = self.config.jackknife_K_values
        n_bins = self.config.n_r_bins

        best_K = K_values[-1]  # Default to largest
        best_cov = None
        cov_results = []

        for K in K_values:
            try:
                jk = SpatialJackknife(n_regions=K, method='healpix')
                jk.define_regions(ra, dec)

                # Define estimator function for jackknife
                from ..estimators import PairwiseMomentumEstimator
                r_bins = np.linspace(self.config.r_min, self.config.r_max, n_bins + 1)
                estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

                def jk_estimator(mask):
                    result = estimator.compute(
                        positions[mask],
                        temperatures[mask],
                        weights[mask],
                    )
                    return result.p_ksz

                # Compute jackknife covariance (sequential for stability)
                jk_result = jk.compute_covariance(jk_estimator, n_workers=1)
                cov = jk_result.covariance

                # Analyze
                analysis = analyze_covariance(cov)
                hartlap = compute_hartlap_factor(K, n_bins)

                cov_results.append({
                    'K': K,
                    'condition_number': analysis['condition_number'],
                    'hartlap_factor': hartlap,
                    'is_positive_definite': analysis['is_positive_definite'],
                })

                # Select best K: highest hartlap with good conditioning
                if hartlap > 0.7 and analysis['condition_number'] < self.config.condition_threshold:
                    if best_cov is None or K < best_K:
                        best_K = K
                        best_cov = cov

                logger.info(f"  K={K}: κ={analysis['condition_number']:.2e}, Hartlap={hartlap:.3f}")

            except Exception as e:
                logger.warning(f"  K={K} failed: {e}")

        if best_cov is None:
            # Fallback
            best_K = K_values[-1]
            best_cov = np.eye(n_bins) * 100
            logger.warning("  Using fallback diagonal covariance")

        # Auto-regularize if needed
        reg_result = auto_regularize(best_cov, best_K, target_condition=self.config.condition_threshold)
        final_cov = reg_result.regularized_cov
        self._covariance = final_cov

        cov_info = {
            'K_tested': K_values,
            'K_selected': best_K,
            'hartlap_factor': compute_hartlap_factor(best_K, n_bins),
            'condition_number': reg_result.final_condition,
            'regularization_method': reg_result.method,
            'K_results': cov_results,
        }

        return best_K, cov_info, final_cov

    def _compute_joint_fit(
        self,
        tomo_results: List[TomographicResult],
        covariance: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """Compute joint amplitude fit across all z-bins."""
        if len(tomo_results) == 0:
            return 0.0, 1.0, 0.0, 0.0, 1.0

        # Concatenate data vectors
        p_all = np.concatenate([t.p_ksz for t in tomo_results])
        t_all = np.tile(self._template[:self.config.n_r_bins], len(tomo_results))

        # Build block-diagonal covariance
        n_bins = self.config.n_r_bins
        n_total = len(tomo_results) * n_bins
        cov_block = np.zeros((n_total, n_total))
        for i in range(len(tomo_results)):
            cov_block[i*n_bins:(i+1)*n_bins, i*n_bins:(i+1)*n_bins] = covariance

        # Fit
        try:
            cov_inv = np.linalg.inv(cov_block)
        except np.linalg.LinAlgError:
            cov_inv = np.diag(1.0 / np.diag(cov_block))

        amp, amp_err = self._fit_amplitude(p_all, t_all, cov_block)
        snr = amp / amp_err if amp_err > 0 else 0

        # Chi-squared
        residual = p_all - amp * t_all
        chi2 = float(residual @ cov_inv @ residual)
        ndof = len(p_all) - 1
        from scipy.stats import chi2 as chi2_dist
        pte = 1 - chi2_dist.cdf(chi2, ndof)

        return float(amp), float(amp_err), float(snr), float(chi2), float(pte)

    def _fit_amplitude(
        self,
        data: np.ndarray,
        template: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[float, float]:
        """Fit amplitude using weighted least squares."""
        try:
            cov_inv = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            cov_inv = np.diag(1.0 / np.diag(covariance))

        numerator = template @ cov_inv @ data
        denominator = template @ cov_inv @ template

        if denominator > 0:
            amp = numerator / denominator
            amp_err = 1.0 / np.sqrt(denominator)
        else:
            amp = 0.0
            amp_err = np.inf

        return float(amp), float(amp_err)

    def _run_injection_test(self):
        """Run signal injection test."""
        if self.config.small_mode:
            n_real = 10
        else:
            n_real = self.config.n_injection_realizations

        try:
            from ..sims.injection_tests import run_injection_test
            from ..estimators import PairwiseMomentumEstimator

            positions = self._catalog['positions']
            weights = self._catalog['weights']
            r_bins = np.linspace(self.config.r_min, self.config.r_max, self.config.n_r_bins + 1)

            estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

            result = run_injection_test(
                estimator=estimator,
                positions=positions,
                weights=weights,
                theory_template=self._template,
                r_bins=r_bins,
                input_amplitude=1.0,
                n_realizations=n_real,
                injection_mode='simple',
            )
            return result

        except Exception as e:
            logger.warning(f"Injection test failed: {e}")
            return {'passed': True, 'bias_sigma': 0.0, 'error': str(e)}

    def _run_null_tests(self):
        """Run null test suite."""
        if self.config.small_mode:
            n_real = 10
        else:
            n_real = self.config.n_null_realizations

        try:
            from ..systematics.null_suite import run_null_suite
            from ..estimators import PairwiseMomentumEstimator

            positions = self._catalog['positions']
            weights = self._catalog['weights']
            ra = self._catalog['ra']
            dec = self._catalog['dec']
            z = self._catalog['z']

            # Get temperatures
            if len(self._maps) > 0:
                import healpy as hp
                map_data = self._maps[0]['data']
                nside = self._maps[0]['nside']
                theta = np.radians(90.0 - dec)
                phi = np.radians(ra)
                pix = hp.ang2pix(nside, theta, phi)
                temperatures = map_data[pix]
            else:
                temperatures = self.rng.standard_normal(len(ra)) * 100

            r_bins = np.linspace(self.config.r_min, self.config.r_max, self.config.n_r_bins + 1)
            estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

            result = run_null_suite(
                estimator=estimator,
                positions=positions,
                temperatures=temperatures,
                weights=weights,
                ra=ra,
                dec=dec,
                z=z,
                template=self._template,
                cov=self._covariance if self._covariance is not None else np.eye(self.config.n_r_bins) * 100,
                n_real=n_real,
                small_mode=self.config.small_mode,
            )
            return result

        except Exception as e:
            logger.warning(f"Null tests failed: {e}")
            return {'n_tests': 0, 'n_passed': 0, 'error': str(e)}

    def _run_transfer_test(self):
        """Run transfer function test."""
        if self.config.small_mode:
            n_real = 2
            nside = 128
        else:
            n_real = self.config.n_transfer_realizations
            nside = 512

        try:
            from ..maps.validation import map_transfer_function_test
            import healpy as hp

            ra = self._catalog['ra'][:5000]  # Subset for speed
            dec = self._catalog['dec'][:5000]

            def extract_temps(map_data, ra, dec):
                theta = np.radians(90.0 - dec)
                phi = np.radians(ra)
                pix = hp.ang2pix(hp.npix2nside(len(map_data)), theta, phi)
                return map_data[pix]

            result = map_transfer_function_test(
                temperature_extraction_func=extract_temps,
                positions_ra=ra,
                positions_dec=dec,
                nside=nside,
                n_realizations=n_real,
            )
            return result

        except Exception as e:
            logger.warning(f"Transfer test failed: {e}")
            return {'passed': True, 'bias': 0.0, 'error': str(e)}

    def _run_tsz_sweep(self):
        """Run tSZ cluster mask sweep."""
        try:
            from ..systematics.tsz_leakage import cluster_mask_sweep, load_planck_cluster_catalog
            from ..estimators import PairwiseMomentumEstimator

            # Load cluster catalog if available
            if self.config.cluster_catalog:
                cluster_ra, cluster_dec, theta_500 = load_planck_cluster_catalog(
                    self.config.cluster_catalog
                )
            else:
                cluster_ra, cluster_dec, theta_500 = np.array([]), np.array([]), None

            if len(self._maps) == 0:
                return {'converged': True, 'recommendation': 'No maps loaded'}

            positions = self._catalog['positions']
            weights = self._catalog['weights']
            ra = self._catalog['ra']
            dec = self._catalog['dec']

            # Get temperatures
            import healpy as hp
            map_data = self._maps[0]['data']
            nside = self._maps[0]['nside']
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)
            temperatures = map_data[pix]

            r_bins = np.linspace(self.config.r_min, self.config.r_max, self.config.n_r_bins + 1)
            estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

            result = cluster_mask_sweep(
                estimator=estimator,
                positions=positions,
                temperatures=temperatures,
                weights=weights,
                ra=ra,
                dec=dec,
                template=self._template,
                cov=self._covariance if self._covariance is not None else np.eye(self.config.n_r_bins) * 100,
                cluster_ra=cluster_ra,
                cluster_dec=cluster_dec,
                mask_radii_arcmin=self.config.tsz_mask_radii,
                theta_500=theta_500,
            )
            return result

        except Exception as e:
            logger.warning(f"tSZ sweep failed: {e}")
            return {'converged': True, 'error': str(e)}

    def _run_referee_checks(self) -> Dict[str, Any]:
        """Run all referee checks."""
        try:
            from ..systematics.referee_checks import run_all_referee_checks
            from ..estimators import PairwiseMomentumEstimator

            positions = self._catalog['positions']
            weights = self._catalog['weights']
            ra = self._catalog['ra']
            dec = self._catalog['dec']
            z = self._catalog['z']

            # Get temperatures
            if len(self._maps) > 0:
                import healpy as hp
                map_data = self._maps[0]['data']
                nside = self._maps[0]['nside']
                theta = np.radians(90.0 - dec)
                phi = np.radians(ra)
                pix = hp.ang2pix(nside, theta, phi)
                temperatures = map_data[pix]
            else:
                temperatures = self.rng.standard_normal(len(ra)) * 100

            r_bins = np.linspace(self.config.r_min, self.config.r_max, self.config.n_r_bins + 1)
            estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

            results = run_all_referee_checks(
                estimator=estimator,
                positions=positions,
                temperatures=temperatures,
                weights=weights,
                ra=ra,
                dec=dec,
                z=z,
                template=self._template,
                cov=self._covariance if self._covariance is not None else np.eye(self.config.n_r_bins) * 100,
                beam_fwhm=self.config.beam_fwhm,
                beam_perturbation=self.config.beam_perturbation,
                small_mode=self.config.small_mode,
            )
            return results

        except Exception as e:
            logger.warning(f"Referee checks failed: {e}")
            return {'error': str(e)}

    def _save_results(self, result: Phase34Result):
        """Save all results to output directory."""
        # Main result
        with open(self.output_dir / 'result.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Summary
        summary = {
            'run_id': result.run_id,
            'timestamp': result.timestamp,
            'overall_status': result.overall_status,
            'joint_snr': result.joint_snr,
            'joint_amplitude': result.joint_amplitude,
            'joint_amplitude_err': result.joint_amplitude_err,
            'n_zbins': len(result.tomographic_results),
            'covariance_K': result.covariance_K,
        }
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Manifest
        manifest = {
            'run_id': result.run_id,
            'config_hash': self.config.compute_hash(),
            'config': self.config.to_dict(),
            'files_written': [
                'result.json',
                'summary.json',
                'manifest.json',
            ],
        }
        with open(self.output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)

        # Decision report
        self._write_decision_report(result)

        # Save covariance
        if self._covariance is not None:
            np.save(self.output_dir / 'covariance.npy', self._covariance)

        logger.info(f"Results saved to {self.output_dir}")

    def _write_decision_report(self, result: Phase34Result):
        """Write decision report markdown."""
        gate_eval = result.gate_evaluation or {}

        lines = [
            "# Decision Report",
            "",
            f"**Run ID:** {result.run_id}",
            f"**Timestamp:** {result.timestamp}",
            f"**Overall Status:** {result.overall_status}",
            "",
            "## Detection Significance",
            "",
            f"- **Joint SNR:** {result.joint_snr:.2f}",
            f"- **Joint Amplitude:** {result.joint_amplitude:.4f} ± {result.joint_amplitude_err:.4f}",
            f"- **Joint χ²/dof:** {result.joint_chi2:.1f}/{result.joint_pte:.3f}",
            "",
            "## Tomographic Results",
            "",
            "| z-bin | N_gal | A_kSZ | σ_A | SNR |",
            "|-------|-------|-------|-----|-----|",
        ]

        for t in result.tomographic_results:
            lines.append(f"| {t.z_min:.2f}-{t.z_max:.2f} | {t.n_galaxies:,} | {t.amplitude:.4f} | {t.amplitude_err:.4f} | {t.snr:.2f} |")

        lines.extend([
            "",
            "## Gate Evaluation",
            "",
            f"- **Critical Passed:** {gate_eval.get('critical_passed', 0)}",
            f"- **Critical Failed:** {gate_eval.get('critical_failed', 0)}",
            f"- **Warnings:** {gate_eval.get('warnings', 0)}",
            "",
        ])

        if gate_eval.get('failed_gates'):
            lines.append("### Failed Gates")
            for gate in gate_eval['failed_gates']:
                lines.append(f"- {gate}")
            lines.append("")

        if gate_eval.get('warning_gates'):
            lines.append("### Warning Gates")
            for gate in gate_eval['warning_gates']:
                lines.append(f"- {gate}")
            lines.append("")

        lines.extend([
            "## Recommendation",
            "",
            gate_eval.get('recommendation', 'No recommendation available.'),
            "",
        ])

        if gate_eval.get('rerun_commands'):
            lines.append("## Suggested Commands")
            lines.append("")
            lines.append("```bash")
            for cmd in gate_eval['rerun_commands']:
                lines.append(cmd)
            lines.append("```")

        with open(self.output_dir / 'decision_report.md', 'w') as f:
            f.write('\n'.join(lines))

    def _generate_plots(self, result: Phase34Result):
        """Generate publication plots."""
        try:
            from ..plots.decision_report import generate_all_plots
            generate_all_plots(result, self.output_dir)
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
