"""
Configuration schemas using Pydantic for validation.

These schemas define the structure and validation rules for all
configuration parameters used in the kSZ analysis pipeline.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Literal, Dict, Any, Union
import yaml
import numpy as np
from dataclasses import dataclass, field

# Try pydantic v2, fall back to dataclasses if not available
try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # Fallback

from .defaults import (
    DEFAULT_SEPARATION_BINS,
    DEFAULT_APERTURE_INNER,
    DEFAULT_APERTURE_OUTER,
    DEFAULT_JACKKNIFE_REGIONS,
    COSMOLOGY_PARAMS,
)


if PYDANTIC_AVAILABLE:
    class DESIConfig(BaseModel):
        """Configuration for DESI galaxy catalog loading.

        Attributes
        ----------
        tracer : str
            Galaxy tracer type: BGS_BRIGHT, BGS_FAINT, LRG, ELG_LOP, QSO
        regions : List[str]
            Galactic cap regions to use: ['N'], ['S'], or ['N', 'S']
        data_dir : Path
            Directory containing DESI LSS catalogs
        z_min : float
            Minimum redshift cut
        z_max : float
            Maximum redshift cut
        apply_weights : bool
            Whether to apply systematic weights
        random_fraction : float
            Fraction of random catalog to use (for speed)
        """
        tracer: Literal["BGS_BRIGHT", "BGS_FAINT", "LRG", "ELG_LOP", "QSO"] = "LRG"
        regions: List[Literal["N", "S"]] = ["N", "S"]
        data_dir: Path = Path("data/ksz/catalogs/")
        z_min: float = 0.4
        z_max: float = 0.8
        apply_weights: bool = True
        random_fraction: float = 1.0

        @field_validator("z_min", "z_max")
        @classmethod
        def validate_redshift(cls, v: float) -> float:
            if not 0.0 <= v <= 3.0:
                raise ValueError(f"Redshift must be between 0 and 3, got {v}")
            return v

        @model_validator(mode="after")
        def validate_z_range(self):
            if self.z_min >= self.z_max:
                raise ValueError(f"z_min ({self.z_min}) must be < z_max ({self.z_max})")
            return self


    class CMBConfig(BaseModel):
        """Configuration for CMB map handling.

        Attributes
        ----------
        source : str
            CMB data source: 'act_dr6' or 'planck_pr4'
        frequency : int
            Frequency band in GHz (for ACT)
        component : str
            Component separation method (for Planck)
        data_dir : Path
            Directory containing CMB maps
        apply_mask : bool
            Whether to apply point source / survey mask
        mask_path : Optional[Path]
            Path to custom mask file
        """
        source: Literal["act_dr6", "planck_pr4"] = "planck_pr4"
        frequency: int = 150
        component: Literal["commander", "sevem", "nilc", "smica"] = "commander"
        data_dir: Path = Path("data/ksz/maps/")
        apply_mask: bool = True
        mask_path: Optional[Path] = None

        @field_validator("frequency")
        @classmethod
        def validate_frequency(cls, v: int) -> int:
            valid_freqs = [90, 100, 143, 150, 217, 220, 353]
            if v not in valid_freqs:
                raise ValueError(f"Frequency must be one of {valid_freqs}, got {v}")
            return v


    class EstimatorConfig(BaseModel):
        """Configuration for kSZ estimators.

        Attributes
        ----------
        separation_bins : List[float]
            Comoving separation bin edges in Mpc/h
        aperture_inner : float
            Inner aperture radius in arcmin
        aperture_outer : float
            Outer aperture radius in arcmin
        filter_type : str
            Filter type: 'compensated', 'tophat', 'matched'
        use_physical_aperture : bool
            If True, convert physical aperture (Mpc) to angular at each z
        physical_aperture : float
            Physical aperture radius in Mpc/h (if use_physical_aperture=True)
        pair_counting_method : str
            Pair counting backend: 'kdtree', 'balltree', 'corrfunc'
        max_separation : float
            Maximum pair separation to consider in Mpc/h
        n_jobs : int
            Number of parallel jobs for pair counting (-1 = all cores)
        """
        separation_bins: List[float] = Field(
            default_factory=lambda: DEFAULT_SEPARATION_BINS.tolist()
        )
        aperture_inner: float = DEFAULT_APERTURE_INNER
        aperture_outer: float = DEFAULT_APERTURE_OUTER
        filter_type: Literal["compensated", "tophat", "matched", "gaussian"] = "compensated"
        use_physical_aperture: bool = False
        physical_aperture: float = 0.5  # Mpc/h
        pair_counting_method: Literal["kdtree", "balltree", "corrfunc"] = "kdtree"
        max_separation: float = 200.0  # Mpc/h
        n_jobs: int = -1

        @field_validator("aperture_inner", "aperture_outer")
        @classmethod
        def validate_aperture(cls, v: float) -> float:
            if not 0.1 <= v <= 30.0:
                raise ValueError(f"Aperture must be between 0.1 and 30 arcmin, got {v}")
            return v


    class CovarianceConfig(BaseModel):
        """Configuration for covariance estimation.

        Attributes
        ----------
        method : str
            Covariance estimation method: 'jackknife', 'bootstrap', 'analytic'
        n_regions : int
            Number of jackknife/bootstrap regions
        region_method : str
            Method to define regions: 'healpix', 'kmeans'
        apply_hartlap : bool
            Whether to apply Hartlap correction to precision matrix
        regularization : str
            Covariance regularization: 'none', 'shrinkage', 'eigenvalue_floor'
        shrinkage_target : float
            Shrinkage parameter (if regularization='shrinkage')
        eigenvalue_floor : float
            Minimum eigenvalue (if regularization='eigenvalue_floor')
        """
        method: Literal["jackknife", "bootstrap", "analytic", "mocks"] = "jackknife"
        n_regions: int = DEFAULT_JACKKNIFE_REGIONS
        region_method: Literal["healpix", "kmeans"] = "healpix"
        apply_hartlap: bool = True
        regularization: Literal["none", "shrinkage", "eigenvalue_floor"] = "none"
        shrinkage_target: float = 0.1
        eigenvalue_floor: float = 1e-10

        @field_validator("n_regions")
        @classmethod
        def validate_n_regions(cls, v: int) -> int:
            if not 20 <= v <= 500:
                raise ValueError(f"n_regions must be between 20 and 500, got {v}")
            return v


    class InferenceConfig(BaseModel):
        """Configuration for parameter inference.

        Attributes
        ----------
        method : str
            Inference method: 'analytic', 'mcmc', 'nested'
        n_walkers : int
            Number of MCMC walkers
        n_samples : int
            Number of MCMC samples
        n_burnin : int
            Number of burn-in samples to discard
        parameters : List[str]
            Parameters to infer: ['A_ksz'], ['A_ksz', 'f_sigma8'], etc.
        prior_type : str
            Prior type: 'flat', 'gaussian'
        prior_bounds : Dict[str, Tuple[float, float]]
            Prior bounds for each parameter
        """
        method: Literal["analytic", "mcmc", "nested"] = "mcmc"
        n_walkers: int = 32
        n_samples: int = 10000
        n_burnin: int = 2000
        parameters: List[str] = ["A_ksz"]
        prior_type: Literal["flat", "gaussian"] = "flat"
        prior_bounds: Dict[str, Tuple[float, float]] = Field(
            default_factory=lambda: {"A_ksz": (-5.0, 5.0), "f_sigma8": (0.0, 2.0)}
        )

        @field_validator("n_walkers")
        @classmethod
        def validate_walkers(cls, v: int) -> int:
            if v < 4:
                raise ValueError(f"n_walkers must be >= 4, got {v}")
            return v


    class CosmologyConfig(BaseModel):
        """Fiducial cosmology parameters.

        Attributes
        ----------
        h : float
            Hubble parameter H_0 / (100 km/s/Mpc)
        Omega_m : float
            Total matter density parameter
        Omega_b : float
            Baryon density parameter
        sigma_8 : float
            Matter fluctuation amplitude at 8 Mpc/h
        n_s : float
            Scalar spectral index
        """
        h: float = COSMOLOGY_PARAMS["h"]
        Omega_m: float = COSMOLOGY_PARAMS["Omega_m"]
        Omega_b: float = COSMOLOGY_PARAMS["Omega_b"]
        sigma_8: float = COSMOLOGY_PARAMS["sigma_8"]
        n_s: float = COSMOLOGY_PARAMS["n_s"]


    class PipelineConfig(BaseModel):
        """Master configuration for the full kSZ pipeline.

        Attributes
        ----------
        name : str
            Run name for output organization
        desi : DESIConfig
            DESI catalog configuration
        cmb : CMBConfig
            CMB map configuration
        estimator : EstimatorConfig
            Estimator configuration
        covariance : CovarianceConfig
            Covariance estimation configuration
        inference : InferenceConfig
            Parameter inference configuration
        cosmology : CosmologyConfig
            Fiducial cosmology
        output_dir : Path
            Base output directory
        random_seed : int
            Random seed for reproducibility
        verbose : bool
            Enable verbose output
        """
        name: str = "ksz_run"
        desi: DESIConfig = Field(default_factory=DESIConfig)
        cmb: CMBConfig = Field(default_factory=CMBConfig)
        estimator: EstimatorConfig = Field(default_factory=EstimatorConfig)
        covariance: CovarianceConfig = Field(default_factory=CovarianceConfig)
        inference: InferenceConfig = Field(default_factory=InferenceConfig)
        cosmology: CosmologyConfig = Field(default_factory=CosmologyConfig)
        output_dir: Path = Path("data/ksz/")
        random_seed: int = 42
        verbose: bool = True

else:
    # Fallback dataclass implementations when pydantic is not available
    @dataclass
    class DESIConfig:
        tracer: str = "LRG"
        regions: List[str] = field(default_factory=lambda: ["N", "S"])
        data_dir: Path = Path("data/ksz/catalogs/")
        z_min: float = 0.4
        z_max: float = 0.8
        apply_weights: bool = True
        random_fraction: float = 1.0

    @dataclass
    class CMBConfig:
        source: str = "planck_pr4"
        frequency: int = 150
        component: str = "commander"
        data_dir: Path = Path("data/ksz/maps/")
        apply_mask: bool = True
        mask_path: Optional[Path] = None

    @dataclass
    class EstimatorConfig:
        separation_bins: List[float] = field(
            default_factory=lambda: DEFAULT_SEPARATION_BINS.tolist()
        )
        aperture_inner: float = DEFAULT_APERTURE_INNER
        aperture_outer: float = DEFAULT_APERTURE_OUTER
        filter_type: str = "compensated"
        use_physical_aperture: bool = False
        physical_aperture: float = 0.5
        pair_counting_method: str = "kdtree"
        max_separation: float = 200.0
        n_jobs: int = -1

    @dataclass
    class CovarianceConfig:
        method: str = "jackknife"
        n_regions: int = DEFAULT_JACKKNIFE_REGIONS
        region_method: str = "healpix"
        apply_hartlap: bool = True
        regularization: str = "none"
        shrinkage_target: float = 0.1
        eigenvalue_floor: float = 1e-10

    @dataclass
    class InferenceConfig:
        method: str = "mcmc"
        n_walkers: int = 32
        n_samples: int = 10000
        n_burnin: int = 2000
        parameters: List[str] = field(default_factory=lambda: ["A_ksz"])
        prior_type: str = "flat"
        prior_bounds: Dict[str, Tuple[float, float]] = field(
            default_factory=lambda: {"A_ksz": (-5.0, 5.0), "f_sigma8": (0.0, 2.0)}
        )

    @dataclass
    class CosmologyConfig:
        h: float = COSMOLOGY_PARAMS["h"]
        Omega_m: float = COSMOLOGY_PARAMS["Omega_m"]
        Omega_b: float = COSMOLOGY_PARAMS["Omega_b"]
        sigma_8: float = COSMOLOGY_PARAMS["sigma_8"]
        n_s: float = COSMOLOGY_PARAMS["n_s"]

    @dataclass
    class PipelineConfig:
        name: str = "ksz_run"
        desi: DESIConfig = field(default_factory=DESIConfig)
        cmb: CMBConfig = field(default_factory=CMBConfig)
        estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
        covariance: CovarianceConfig = field(default_factory=CovarianceConfig)
        inference: InferenceConfig = field(default_factory=InferenceConfig)
        cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
        output_dir: Path = Path("data/ksz/")
        random_seed: int = 42
        verbose: bool = True


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load pipeline configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file

    Returns
    -------
    PipelineConfig
        Validated pipeline configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if PYDANTIC_AVAILABLE:
        return PipelineConfig(**config_dict)
    else:
        # Manual construction for dataclass version
        desi = DESIConfig(**config_dict.get("desi", {}))
        cmb = CMBConfig(**config_dict.get("cmb", {}))
        estimator = EstimatorConfig(**config_dict.get("estimator", {}))
        covariance = CovarianceConfig(**config_dict.get("covariance", {}))
        inference = InferenceConfig(**config_dict.get("inference", {}))
        cosmology = CosmologyConfig(**config_dict.get("cosmology", {}))

        return PipelineConfig(
            name=config_dict.get("name", "ksz_run"),
            desi=desi,
            cmb=cmb,
            estimator=estimator,
            covariance=covariance,
            inference=inference,
            cosmology=cosmology,
            output_dir=Path(config_dict.get("output_dir", "data/ksz/")),
            random_seed=config_dict.get("random_seed", 42),
            verbose=config_dict.get("verbose", True),
        )


def save_config(config: PipelineConfig, output_path: Union[str, Path]) -> None:
    """Save pipeline configuration to YAML file.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration to save
    output_path : str or Path
        Output path for YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if PYDANTIC_AVAILABLE:
        config_dict = config.model_dump()
    else:
        # Convert dataclass to dict
        import dataclasses
        config_dict = dataclasses.asdict(config)

    # Convert Path objects to strings for YAML
    def convert_paths(d):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                convert_paths(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
        return d

    config_dict = convert_paths(config_dict)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
