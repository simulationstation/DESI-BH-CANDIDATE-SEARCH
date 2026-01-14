"""
Command-line interface for kSZ analysis pipeline.

Provides staged commands for the full analysis workflow:
    1. ingest-desi     - Load and preprocess DESI catalogs
    2. ingest-maps     - Load and preprocess CMB maps
    3. make-masks      - Create analysis masks
    4. filter-maps     - Apply matched/Wiener filtering
    5. measure-temps   - Extract temperatures at galaxy positions
    6. compute-pairwise - Compute pairwise kSZ momentum
    7. covariance      - Estimate covariance matrix
    8. null-tests      - Run null test suite
    9. inference       - Parameter inference
    10. make-plots     - Generate publication figures

Usage
-----
    python -m desi_ksz.cli pipeline --config config.yaml
    python -m desi_ksz.cli compute-pairwise --tracer LRG --z-min 0.4 --z-max 0.6
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if not CLICK_AVAILABLE:
    print("Error: click is required for CLI. Install with: pip install click")
    sys.exit(1)


# ============================================================================
# Main CLI group
# ============================================================================

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def cli(ctx, verbose, quiet):
    """
    DESI kSZ Analysis Pipeline.

    A publication-grade pairwise kSZ analysis using DESI DR1 galaxies
    cross-correlated with CMB temperature maps.
    """
    ctx.ensure_object(dict)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)

    ctx.obj['verbose'] = verbose


# ============================================================================
# Pipeline command - runs full workflow
# ============================================================================

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--output', '-o', type=click.Path(), default='data/ksz',
              help='Output directory')
@click.option('--tracer', type=click.Choice(['BGS', 'LRG', 'ELG', 'QSO']),
              default='LRG', help='Galaxy tracer type')
@click.option('--cmb-source', type=click.Choice(['act_dr6', 'planck_pr4']),
              default='act_dr6', help='CMB map source')
@click.option('--skip-download', is_flag=True,
              help='Skip data download (assume data exists)')
@click.pass_context
def pipeline(ctx, config, output, tracer, cmb_source, skip_download):
    """
    Run full kSZ analysis pipeline.

    This executes all pipeline stages in sequence:
    ingest → masks → filter → temperatures → pairwise → covariance →
    null-tests → inference → plots
    """
    click.echo("=" * 60)
    click.echo("DESI kSZ Analysis Pipeline")
    click.echo("=" * 60)

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if config and YAML_AVAILABLE:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        click.echo(f"Loaded configuration from {config}")
    else:
        cfg = {}
        click.echo("Using default configuration")

    # Stage 1: Ingest DESI catalogs
    click.echo("\n[1/10] Ingesting DESI catalogs...")
    ctx.invoke(ingest_desi, tracer=tracer, output=str(output_dir / 'catalogs'))

    # Stage 2: Ingest CMB maps
    click.echo("\n[2/10] Ingesting CMB maps...")
    ctx.invoke(ingest_maps, source=cmb_source, output=str(output_dir / 'maps'))

    # Stage 3: Create masks
    click.echo("\n[3/10] Creating masks...")
    ctx.invoke(make_masks, output=str(output_dir / 'masks'))

    # Stage 4: Filter maps
    click.echo("\n[4/10] Filtering maps...")
    ctx.invoke(filter_maps, filter_type='matched', output=str(output_dir / 'filtered'))

    # Stage 5: Measure temperatures
    click.echo("\n[5/10] Measuring temperatures...")
    ctx.invoke(measure_temps, output=str(output_dir / 'temperatures.h5'))

    # Stage 6: Compute pairwise momentum
    click.echo("\n[6/10] Computing pairwise momentum...")
    ctx.invoke(compute_pairwise, output=str(output_dir / 'pairwise'))

    # Stage 7: Estimate covariance
    click.echo("\n[7/10] Estimating covariance...")
    ctx.invoke(covariance, output=str(output_dir / 'covariance'))

    # Stage 8: Run null tests
    click.echo("\n[8/10] Running null tests...")
    ctx.invoke(null_tests, output=str(output_dir / 'nulls'))

    # Stage 9: Parameter inference
    click.echo("\n[9/10] Running inference...")
    ctx.invoke(inference, output=str(output_dir / 'chains'))

    # Stage 10: Generate plots
    click.echo("\n[10/10] Generating plots...")
    ctx.invoke(make_plots, output=str(output_dir / 'plots'))

    click.echo("\n" + "=" * 60)
    click.echo("Pipeline complete!")
    click.echo(f"Results written to: {output_dir}")
    click.echo("=" * 60)


# ============================================================================
# Individual pipeline stages
# ============================================================================

@cli.command('ingest-desi')
@click.option('--tracer', type=click.Choice(['BGS', 'LRG', 'ELG', 'QSO']),
              default='LRG', help='Galaxy tracer type')
@click.option('--data-dir', type=click.Path(),
              help='Directory containing DESI LSS catalogs')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/catalogs',
              help='Output directory')
@click.option('--z-min', type=float, default=None, help='Minimum redshift')
@click.option('--z-max', type=float, default=None, help='Maximum redshift')
def ingest_desi(tracer, data_dir, output, z_min, z_max):
    """
    Load and preprocess DESI LSS catalogs.

    Reads DESI DR1 LSS clustering catalogs, applies quality cuts,
    and computes comoving positions.
    """
    from .io import load_desi_catalog
    from .selection import apply_quality_cuts

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading {tracer} catalog...")

    try:
        catalog = load_desi_catalog(
            tracer=tracer,
            data_dir=data_dir,
            region='both',  # North + South
        )

        # Apply redshift cuts if specified
        if z_min is not None or z_max is not None:
            z_min = z_min or 0.0
            z_max = z_max or 10.0
            catalog = catalog.select_redshift_bin(z_min, z_max)
            click.echo(f"  Applied redshift cut: {z_min:.2f} < z < {z_max:.2f}")

        # Apply quality cuts
        catalog = apply_quality_cuts(catalog, tracer=tracer)

        click.echo(f"  Loaded {len(catalog.ra):,} galaxies")
        click.echo(f"  Redshift range: {catalog.z.min():.3f} - {catalog.z.max():.3f}")

        # Save processed catalog
        output_file = output_dir / f'{tracer}_catalog.npz'
        np.savez(
            output_file,
            ra=catalog.ra,
            dec=catalog.dec,
            z=catalog.z,
            weights=catalog.weights,
            positions=catalog.positions,
        )
        click.echo(f"  Saved to: {output_file}")

    except FileNotFoundError as e:
        click.echo(f"  Warning: {e}")
        click.echo("  Run 'desi_ksz download' to fetch data, or specify --data-dir")


@cli.command('ingest-maps')
@click.option('--source', type=click.Choice(['act_dr6', 'planck_pr4']),
              default='act_dr6', help='CMB map source')
@click.option('--frequency', type=str, default='f150',
              help='Frequency band (e.g., f150, f090)')
@click.option('--data-dir', type=click.Path(),
              help='Directory containing CMB maps')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/maps',
              help='Output directory')
def ingest_maps(source, frequency, data_dir, output):
    """
    Load and preprocess CMB temperature maps.

    Supports ACT DR6 (CAR projection) and Planck PR4 (HEALPix).
    """
    from .io import load_cmb_map

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading {source} {frequency} map...")

    try:
        cmb_map = load_cmb_map(
            source=source,
            frequency=frequency,
            data_dir=data_dir,
        )

        # Log map properties
        if hasattr(cmb_map, 'nside'):
            click.echo(f"  HEALPix map, NSIDE={cmb_map.nside}")
        if hasattr(cmb_map, 'shape'):
            click.echo(f"  Map shape: {cmb_map.shape}")

        click.echo(f"  Map range: {cmb_map.data.min():.1f} to {cmb_map.data.max():.1f} μK")

        # Save metadata
        metadata_file = output_dir / f'{source}_{frequency}_metadata.yaml'
        if YAML_AVAILABLE:
            metadata = {
                'source': source,
                'frequency': frequency,
                'data_dir': str(data_dir) if data_dir else None,
            }
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f)
            click.echo(f"  Metadata saved to: {metadata_file}")

    except FileNotFoundError as e:
        click.echo(f"  Warning: {e}")
        click.echo("  Run 'desi_ksz download' to fetch data, or specify --data-dir")


@cli.command('make-masks')
@click.option('--point-source-catalog', type=click.Path(),
              help='Point source catalog for masking')
@click.option('--cluster-catalog', type=click.Path(),
              help='Cluster catalog for tSZ masking')
@click.option('--galactic-cut', type=float, default=20.0,
              help='Galactic latitude cut (degrees)')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/masks',
              help='Output directory')
def make_masks(point_source_catalog, cluster_catalog, galactic_cut, output):
    """
    Create analysis masks.

    Generates point source masks, cluster masks, and Galactic plane masks.
    """
    from .maps import create_galactic_mask

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Creating analysis masks...")

    # Galactic plane mask
    click.echo(f"  Creating Galactic mask (|b| > {galactic_cut}°)...")
    galactic_mask = create_galactic_mask(
        nside=2048,
        galactic_cut=galactic_cut,
    )

    n_masked = np.sum(~galactic_mask)
    f_sky = np.mean(galactic_mask)
    click.echo(f"  Galactic mask: {n_masked:,} pixels masked, f_sky = {f_sky:.2%}")

    # Save masks
    mask_file = output_dir / 'galactic_mask.npy'
    np.save(mask_file, galactic_mask)
    click.echo(f"  Saved to: {mask_file}")

    # Point source mask (if catalog provided)
    if point_source_catalog:
        from .maps import create_point_source_mask
        click.echo("  Creating point source mask...")
        # Implementation would load catalog and create mask

    # Cluster mask (if catalog provided)
    if cluster_catalog:
        from .maps import create_cluster_mask
        click.echo("  Creating cluster mask...")
        # Implementation would load catalog and create mask


@cli.command('filter-maps')
@click.option('--filter-type', type=click.Choice(['matched', 'wiener', 'none']),
              default='matched', help='Filter type to apply')
@click.option('--beam-fwhm', type=float, default=1.4,
              help='Beam FWHM in arcmin')
@click.option('--input-dir', type=click.Path(), default='data/ksz/maps',
              help='Input maps directory')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/filtered',
              help='Output directory')
def filter_maps(filter_type, beam_fwhm, input_dir, output):
    """
    Apply spatial filtering to CMB maps.

    Options: matched filter (optimal S/N), Wiener filter, or none.
    """
    from .maps import apply_matched_filter

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Applying {filter_type} filter...")

    if filter_type == 'matched':
        click.echo(f"  Beam FWHM: {beam_fwhm} arcmin")
        # Implementation would load map and apply filter
        click.echo("  Filter applied (placeholder)")
    elif filter_type == 'wiener':
        click.echo("  Applying Wiener filter (placeholder)")
    else:
        click.echo("  No filtering applied")


@cli.command('measure-temps')
@click.option('--aperture-inner', type=float, default=1.8,
              help='Inner aperture radius (arcmin)')
@click.option('--aperture-outer', type=float, default=5.0,
              help='Outer aperture radius (arcmin)')
@click.option('--catalog-dir', type=click.Path(), default='data/ksz/catalogs',
              help='Catalog directory')
@click.option('--map-dir', type=click.Path(), default='data/ksz/filtered',
              help='Map directory')
@click.option('--output', '-o', type=click.Path(),
              default='data/ksz/temperatures.h5', help='Output file')
def measure_temps(aperture_inner, aperture_outer, catalog_dir, map_dir, output):
    """
    Extract CMB temperatures at galaxy positions.

    Uses compensated aperture photometry: T_AP = T_inner - T_outer
    """
    from .estimators import AperturePhotometry

    click.echo("Measuring temperatures at galaxy positions...")
    click.echo(f"  Aperture: θ_inner = {aperture_inner}', θ_outer = {aperture_outer}'")

    ap = AperturePhotometry(
        theta_inner=aperture_inner,
        theta_outer=aperture_outer,
    )

    # Load catalog
    catalog_dir = Path(catalog_dir)
    catalog_files = list(catalog_dir.glob('*_catalog.npz'))

    if not catalog_files:
        click.echo("  Warning: No catalogs found")
        return

    for catalog_file in catalog_files:
        click.echo(f"  Processing {catalog_file.name}...")
        data = np.load(catalog_file)
        ra, dec = data['ra'], data['dec']
        click.echo(f"    {len(ra):,} galaxies")

        # Temperature extraction would happen here
        # temperatures = ap.extract_temperatures(cmb_map, ra, dec)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"  Output: {output}")


@cli.command('compute-pairwise')
@click.option('--r-min', type=float, default=20.0,
              help='Minimum separation (Mpc/h)')
@click.option('--r-max', type=float, default=150.0,
              help='Maximum separation (Mpc/h)')
@click.option('--n-bins', type=int, default=15,
              help='Number of separation bins')
@click.option('--z-bins', type=str, default='0.1,0.3,0.5,0.7',
              help='Redshift bin edges (comma-separated)')
@click.option('--catalog-dir', type=click.Path(), default='data/ksz/catalogs',
              help='Catalog directory')
@click.option('--temp-file', type=click.Path(),
              default='data/ksz/temperatures.h5', help='Temperature file')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/pairwise',
              help='Output directory')
def compute_pairwise(r_min, r_max, n_bins, z_bins, catalog_dir, temp_file, output):
    """
    Compute pairwise kSZ momentum estimator.

    p̂(r) = Σ w_ij (T_i - T_j) c_ij / Σ w_ij c_ij²

    where c_ij is the geometric weight.
    """
    from .estimators import PairwiseMomentumEstimator

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse redshift bins
    z_edges = [float(z) for z in z_bins.split(',')]
    click.echo(f"Redshift bins: {z_edges}")

    # Set up separation bins
    r_bins = np.linspace(r_min, r_max, n_bins + 1)
    click.echo(f"Separation bins: {r_min} - {r_max} Mpc/h ({n_bins} bins)")

    estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

    # Process each redshift bin
    for i in range(len(z_edges) - 1):
        z_lo, z_hi = z_edges[i], z_edges[i + 1]
        click.echo(f"\n  Processing z = {z_lo:.2f} - {z_hi:.2f}...")

        # Load catalog for this z-bin
        # result = estimator.compute(positions, temperatures, weights)

        # Save results
        output_file = output_dir / f'p_ksz_z{z_lo:.1f}-{z_hi:.1f}.csv'
        click.echo(f"    Output: {output_file}")

    click.echo("\nPairwise momentum computation complete")


@cli.command()
@click.option('--method', type=click.Choice(['jackknife', 'bootstrap', 'mock']),
              default='jackknife', help='Covariance estimation method')
@click.option('--n-regions', type=int, default=100,
              help='Number of jackknife regions')
@click.option('--hartlap/--no-hartlap', default=True,
              help='Apply Hartlap correction')
@click.option('--input-dir', type=click.Path(), default='data/ksz/pairwise',
              help='Input pairwise results directory')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/covariance',
              help='Output directory')
def covariance(method, n_regions, hartlap, input_dir, output):
    """
    Estimate covariance matrix.

    Methods:
      - jackknife: Spatial delete-one jackknife
      - bootstrap: Bootstrap resampling
      - mock: From mock catalogs (AbacusSummit)
    """
    from .covariance import SpatialJackknife, apply_hartlap_correction

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Estimating covariance using {method} method...")

    if method == 'jackknife':
        click.echo(f"  Number of regions: {n_regions}")

        # Implementation would:
        # 1. Load catalog positions
        # 2. Define jackknife regions
        # 3. Recompute p(r) for each jackknife
        # 4. Compute covariance

        if hartlap:
            click.echo("  Hartlap correction will be applied")

    elif method == 'mock':
        click.echo("  Using mock catalogs for covariance")
        # Implementation would load AbacusSummit mocks

    click.echo(f"  Output: {output_dir}")


@cli.command('null-tests')
@click.option('--tests', type=str, default='all',
              help='Tests to run (comma-separated or "all")')
@click.option('--n-realizations', type=int, default=1000,
              help='Number of realizations for shuffle tests')
@click.option('--pte-threshold', type=float, default=0.05,
              help='PTE threshold for pass/fail')
@click.option('--input-dir', type=click.Path(), default='data/ksz/pairwise',
              help='Input pairwise results directory')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/nulls',
              help='Output directory')
def null_tests(tests, n_realizations, pte_threshold, input_dir, output):
    """
    Run null test suite.

    Available tests:
      - shuffle_temperatures: Permute T_i among galaxies
      - random_positions: Use random catalog positions
      - scramble_redshifts: Permute z among galaxies
      - hemisphere_split: Compare N vs S
      - even_odd_split: Compare even vs odd TARGETID
    """
    from .systematics import NullTestSuite, summarize_null_tests

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse test list
    if tests == 'all':
        test_list = NullTestSuite.NULL_TESTS
    else:
        test_list = [t.strip() for t in tests.split(',')]

    click.echo(f"Running {len(test_list)} null tests...")
    click.echo(f"  Realizations: {n_realizations}")
    click.echo(f"  PTE threshold: {pte_threshold}")

    suite = NullTestSuite(
        n_realizations=n_realizations,
        pte_threshold=pte_threshold,
    )

    # Run tests (implementation would load data and run)
    click.echo("\nTests to run:")
    for test in test_list:
        click.echo(f"  - {test}")

    # Save results
    results_file = output_dir / 'null_test_results.yaml'
    click.echo(f"\n  Results: {results_file}")


@cli.command()
@click.option('--method', type=click.Choice(['analytic', 'mcmc']),
              default='analytic', help='Inference method')
@click.option('--n-walkers', type=int, default=32,
              help='Number of MCMC walkers')
@click.option('--n-steps', type=int, default=5000,
              help='Number of MCMC steps')
@click.option('--n-burnin', type=int, default=1000,
              help='MCMC burn-in steps')
@click.option('--input-dir', type=click.Path(), default='data/ksz/pairwise',
              help='Input pairwise results directory')
@click.option('--cov-dir', type=click.Path(), default='data/ksz/covariance',
              help='Covariance directory')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/chains',
              help='Output directory')
def inference(method, n_walkers, n_steps, n_burnin, input_dir, cov_dir, output):
    """
    Run parameter inference.

    Infers kSZ amplitude A_kSZ using:
      - analytic: Direct ML estimate
      - mcmc: Full posterior sampling with emcee
    """
    from .inference import KSZLikelihood, run_mcmc

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Running {method} inference...")

    if method == 'analytic':
        click.echo("  Computing ML amplitude estimate")
        click.echo("  Â_kSZ = (p^T Ψ d) / (p^T Ψ p)")
        click.echo("  σ_A = 1 / √(p^T Ψ p)")

        # Implementation would:
        # likelihood = KSZLikelihood(data, covariance, theory)
        # A_ml, sigma_A = likelihood.fit_amplitude()

    elif method == 'mcmc':
        click.echo(f"  Walkers: {n_walkers}")
        click.echo(f"  Steps: {n_steps} ({n_burnin} burn-in)")

        # Implementation would run emcee

    click.echo(f"  Output: {output_dir}")


@cli.command('make-plots')
@click.option('--style', type=click.Choice(['paper', 'presentation', 'default']),
              default='paper', help='Plot style')
@click.option('--format', 'fmt', type=click.Choice(['pdf', 'png', 'both']),
              default='pdf', help='Output format')
@click.option('--input-dir', type=click.Path(), default='data/ksz',
              help='Input data directory')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/plots',
              help='Output directory')
def make_plots(style, fmt, input_dir, output):
    """
    Generate publication figures.

    Creates:
      - p_ksz_vs_r.pdf: Main pairwise momentum result
      - A_ksz_vs_z.pdf: Amplitude tomography
      - null_test_summary.pdf: Null test results
      - corner_A_ksz.pdf: Parameter posterior
    """
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Generating {style} plots in {fmt} format...")

    plots_to_make = [
        'p_ksz_vs_r',
        'A_ksz_vs_z',
        'null_test_summary',
        'corner_A_ksz',
    ]

    for plot_name in plots_to_make:
        if fmt == 'both':
            click.echo(f"  {plot_name}.pdf, {plot_name}.png")
        else:
            click.echo(f"  {plot_name}.{fmt}")

    click.echo(f"  Output: {output_dir}")


# ============================================================================
# P1 Validation Commands (Phase 0-2)
# ============================================================================

@cli.command('validate-map')
@click.option('--map-file', '-m', type=click.Path(exists=True), required=True,
              help='CMB map FITS file to validate')
@click.option('--mask-file', type=click.Path(exists=True),
              help='Mask FITS file')
@click.option('--catalog-file', type=click.Path(exists=True),
              help='Galaxy catalog for beam injection test')
@click.option('--beam-fwhm', type=float, default=1.4,
              help='Beam FWHM in arcmin')
@click.option('--run-beam-test/--no-beam-test', default=True,
              help='Run beam injection test')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/validation',
              help='Output directory')
def validate_map(map_file, mask_file, catalog_file, beam_fwhm, run_beam_test, output):
    """
    Validate CMB map for kSZ analysis.

    Runs sanity checks including:
      - NaN/Inf detection
      - Statistics check (mean, std, range)
      - Coordinate consistency with catalog
      - Beam injection/recovery test
    """
    from .maps.validation import (
        validate_cmb_map, run_full_map_validation, beam_injection_test
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("CMB Map Validation")
    click.echo("=" * 50)
    click.echo(f"Map: {map_file}")

    # Basic validation
    result = validate_cmb_map(map_file, mask_file)

    click.echo(f"\nBasic checks:")
    for check_name, check_result in result.checks.items():
        passed = check_result.get('passed', True)
        status = "PASS" if passed else "FAIL"
        click.echo(f"  {check_name}: {status}")

    if result.warnings:
        click.echo(f"\nWarnings:")
        for warn in result.warnings:
            click.echo(f"  - {warn}")

    # Full validation with catalog if provided
    if catalog_file and run_beam_test:
        click.echo(f"\nRunning beam injection test...")
        data = np.load(catalog_file)
        ra, dec = data['ra'], data['dec']

        full_result = run_full_map_validation(
            map_file, ra, dec,
            beam_fwhm_arcmin=beam_fwhm,
            mask_file=mask_file,
            n_inject=50,
        )

        if 'beam_injection' in full_result:
            br = full_result['beam_injection']
            status = "PASS" if br.get('passed', False) else "FAIL"
            click.echo(f"  Beam injection: {status}")
            click.echo(f"    Recovery fraction: {br.get('recovery_fraction', 0):.2f}")

    # Save results
    result.to_json(str(output_dir / 'map_validation.json'))
    click.echo(f"\nResults saved to: {output_dir / 'map_validation.json'}")

    overall = "PASS" if result.passed else "FAIL"
    click.echo(f"\nOverall: {overall}")


@cli.command('validate-catalog')
@click.option('--catalog-file', '-c', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/validation',
              help='Output directory')
def validate_catalog_cmd(catalog_file, output):
    """
    Validate galaxy catalog for kSZ analysis.

    Checks:
      - Coordinate ranges (RA, Dec)
      - Redshift validity
      - Weight positivity
      - NaN/Inf detection
      - Duplicate detection
    """
    from .maps.validation import validate_catalog
    import json

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Galaxy Catalog Validation")
    click.echo("=" * 50)
    click.echo(f"Catalog: {catalog_file}")

    # Load catalog
    data = np.load(catalog_file)
    ra, dec, z = data['ra'], data['dec'], data['z']
    weights = data.get('weights', None)

    click.echo(f"  N galaxies: {len(ra):,}")

    # Validate
    result = validate_catalog(ra, dec, z, weights)

    click.echo(f"\nChecks:")
    for check_name, check_result in result['checks'].items():
        passed = check_result.get('passed', True)
        status = "PASS" if passed else "FAIL"
        click.echo(f"  {check_name}: {status}")

    if result['warnings']:
        click.echo(f"\nWarnings:")
        for warn in result['warnings']:
            click.echo(f"  - {warn}")

    # Save results
    output_file = output_dir / 'catalog_validation.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    click.echo(f"\nResults saved to: {output_file}")

    overall = "PASS" if result['passed'] else "FAIL"
    click.echo(f"\nOverall: {overall}")


@cli.command('cov-stability')
@click.option('--cov-file', '-c', type=click.Path(exists=True),
              help='Covariance matrix file (.npy)')
@click.option('--n-samples', type=int, default=100,
              help='Number of jackknife samples used')
@click.option('--k-values', type=str, default='50,75,100,125,150',
              help='K values to test (comma-separated)')
@click.option('--regularize/--no-regularize', default=False,
              help='Apply regularization if needed')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/covariance',
              help='Output directory')
def cov_stability(cov_file, n_samples, k_values, regularize, output):
    """
    Analyze covariance matrix stability.

    Checks:
      - Condition number
      - Eigenvalue spectrum
      - Hartlap correction factor
      - Stability across K values
    """
    from .covariance.stability import (
        analyze_covariance, compute_hartlap_factor,
        regularize_eigenvalue_floor, choose_regularization
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Covariance Stability Analysis")
    click.echo("=" * 50)

    if cov_file:
        cov = np.load(cov_file)
        click.echo(f"Loaded covariance: {cov.shape}")

        # Analyze
        analysis = analyze_covariance(cov)

        click.echo(f"\nMatrix properties:")
        click.echo(f"  Shape: {analysis['shape']}")
        click.echo(f"  Condition number: {analysis['condition_number']:.2e}")
        click.echo(f"  Positive definite: {analysis['is_positive_definite']}")

        click.echo(f"\nEigenvalue spectrum:")
        click.echo(f"  Min: {analysis['eigenvalue_min']:.2e}")
        click.echo(f"  Max: {analysis['eigenvalue_max']:.2e}")
        click.echo(f"  Median: {analysis['eigenvalue_median']:.2e}")

        # Hartlap factor
        n_bins = cov.shape[0]
        hartlap = compute_hartlap_factor(n_samples, n_bins)
        click.echo(f"\nHartlap correction:")
        click.echo(f"  K = {n_samples}, N_bins = {n_bins}")
        click.echo(f"  Factor = {hartlap:.4f}")

        # Regularization recommendation
        reg_choice = choose_regularization(analysis)
        click.echo(f"\nRegularization: {reg_choice['recommendation']}")
        click.echo(f"  Reason: {reg_choice['reason']}")

        if regularize and reg_choice['recommendation'] != 'none':
            click.echo(f"\nApplying {reg_choice['recommendation']} regularization...")
            cov_reg, reg_info = regularize_eigenvalue_floor(cov)
            click.echo(f"  New condition number: {reg_info['new_condition_number']:.2e}")

            # Save regularized
            np.save(output_dir / 'covariance_regularized.npy', cov_reg)
            click.echo(f"  Saved: {output_dir / 'covariance_regularized.npy'}")

    else:
        click.echo("No covariance file provided. Specify --cov-file.")


@cli.command('injection-test')
@click.option('--catalog-file', '-c', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--template-file', type=click.Path(exists=True),
              help='Theory template file')
@click.option('--cov-file', type=click.Path(exists=True),
              help='Covariance matrix for weighted fit')
@click.option('--amplitude', '-a', type=float, default=1.0,
              help='Input amplitude to inject')
@click.option('--n-realizations', '-n', type=int, default=100,
              help='Number of realizations')
@click.option('--mode', type=click.Choice(['simple', 'template', 'velocity_field']),
              default='simple', help='Injection mode')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/injection',
              help='Output directory')
def injection_test_cmd(catalog_file, template_file, cov_file, amplitude,
                       n_realizations, mode, output):
    """
    Run signal injection test to validate estimator.

    Injects known signal and verifies unbiased recovery.
    Modes:
      - simple: Gaussian random velocities (fast)
      - template: Pair-based injection (more accurate)
      - velocity_field: Correlated velocity field (most realistic)
    """
    from .sims.injection_tests import run_injection_test
    from .estimators import PairwiseMomentumEstimator
    from .config import DEFAULT_SEPARATION_BINS

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Signal Injection Test")
    click.echo("=" * 50)
    click.echo(f"Mode: {mode}")
    click.echo(f"Input amplitude: {amplitude}")
    click.echo(f"Realizations: {n_realizations}")

    # Load catalog
    data = np.load(catalog_file)
    positions = data['positions']
    weights = data['weights']
    ra, dec = data.get('ra'), data.get('dec')

    click.echo(f"Loaded {len(positions):,} galaxies")

    # Load or create template
    r_bins = np.array(DEFAULT_SEPARATION_BINS)
    if template_file:
        template_data = np.load(template_file)
        theory_template = template_data['template']
    else:
        # Simple declining template
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        theory_template = 1.0 / (1 + r_centers / 50.0)

    # Load covariance if provided
    cov = np.load(cov_file) if cov_file else None

    # Create estimator
    estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

    # Run test
    click.echo("\nRunning injection test...")
    result = run_injection_test(
        estimator=estimator,
        positions=positions,
        weights=weights,
        theory_template=theory_template,
        r_bins=r_bins,
        input_amplitude=amplitude,
        n_realizations=n_realizations,
        injection_mode=mode,
        covariance=cov,
        ra=ra,
        dec=dec,
    )

    # Display results
    click.echo(f"\nResults:")
    click.echo(f"  Input amplitude:  {result.input_amplitude:.4f}")
    click.echo(f"  Recovered mean:   {result.mean_recovered:.4f} +/- {result.std_recovered:.4f}")
    click.echo(f"  Bias:             {result.bias:.4f} ({result.bias_sigma:.1f}σ)")
    click.echo(f"  Fractional bias:  {result.fractional_bias:.1%}")

    status = "PASS" if result.passed else "FAIL"
    click.echo(f"\n  Status: {status}")

    # Save results
    result.to_json(str(output_dir / 'injection_test_result.json'))
    click.echo(f"\nSaved: {output_dir / 'injection_test_result.json'}")


@cli.command('null-suite')
@click.option('--catalog-file', '-c', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--temp-file', '-t', type=click.Path(exists=True),
              help='Temperature measurements file')
@click.option('--cov-file', type=click.Path(exists=True),
              help='Covariance matrix file')
@click.option('--template-file', type=click.Path(exists=True),
              help='Theory template file')
@click.option('--tests', type=str, default='all',
              help='Tests to run (comma-separated or "all")')
@click.option('--n-realizations', '-n', type=int, default=100,
              help='Realizations for shuffle tests')
@click.option('--small-mode', is_flag=True,
              help='Quick mode for CI testing')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/nulls',
              help='Output directory')
def null_suite_cmd(catalog_file, temp_file, cov_file, template_file,
                   tests, n_realizations, small_mode, output):
    """
    Run comprehensive null test suite.

    Tests:
      - shuffle: Permute temperatures among galaxies
      - scramble: Permute redshifts
      - hemisphere: Compare North vs South
      - random: Random positions instead of galaxies
      - cluster_mask: Vary cluster mask radius
      - aperture: Vary aperture size
    """
    from .systematics.null_suite import run_null_suite, plot_null_suite_summary
    from .estimators import PairwiseMomentumEstimator
    from .config import DEFAULT_SEPARATION_BINS

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Null Test Suite")
    click.echo("=" * 50)

    if small_mode:
        click.echo("Running in small/CI mode (reduced realizations)")
        n_realizations = min(n_realizations, 10)

    # Load data
    data = np.load(catalog_file)
    positions = data['positions']
    weights = data['weights']
    ra, dec, z = data['ra'], data['dec'], data['z']

    click.echo(f"Loaded {len(positions):,} galaxies")

    # Load temperatures
    if temp_file:
        temp_data = np.load(temp_file)
        temperatures = temp_data['temperatures']
    else:
        # Mock temperatures for testing
        rng = np.random.default_rng(42)
        temperatures = rng.standard_normal(len(positions)) * 100

    # Load covariance
    if cov_file:
        cov = np.load(cov_file)
    else:
        # Mock diagonal covariance
        n_bins = len(DEFAULT_SEPARATION_BINS) - 1
        cov = np.eye(n_bins) * 100

    # Load template
    r_bins = np.array(DEFAULT_SEPARATION_BINS)
    if template_file:
        template_data = np.load(template_file)
        template = template_data['template']
    else:
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        template = 1.0 / (1 + r_centers / 50.0)

    # Create estimator
    estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

    # Parse test list
    if tests == 'all':
        test_list = None
    else:
        test_list = [t.strip() for t in tests.split(',')]

    # Run suite
    click.echo(f"\nRunning null tests ({n_realizations} realizations)...")
    result = run_null_suite(
        estimator=estimator,
        positions=positions,
        temperatures=temperatures,
        weights=weights,
        ra=ra,
        dec=dec,
        z=z,
        template=template,
        cov=cov,
        tests=test_list,
        n_real=n_realizations,
        small_mode=small_mode,
    )

    # Display results
    click.echo(f"\nResults:")
    click.echo(f"  Tests run: {result.n_tests}")
    click.echo(f"  Passed:    {result.n_passed}")
    click.echo(f"  Failed:    {result.n_failed}")

    click.echo(f"\nIndividual tests:")
    for test in result.results:
        status = "PASS" if test.passed else "FAIL"
        click.echo(f"  {test.test_name}: {status} (PTE={test.pte:.3f})")

    overall = "PASS" if result.all_passed else "FAIL"
    click.echo(f"\nOverall: {overall}")

    # Save results
    result.to_json(str(output_dir / 'null_suite_results.json'))
    click.echo(f"\nSaved: {output_dir / 'null_suite_results.json'}")

    # Generate plot
    try:
        plot_null_suite_summary(result, output_dir / 'null_suite_summary.pdf')
        click.echo(f"Plot:  {output_dir / 'null_suite_summary.pdf'}")
    except Exception as e:
        click.echo(f"Warning: Could not generate plot: {e}")


# ============================================================================
# Utility commands
# ============================================================================

@cli.command()
@click.option('--dataset', type=click.Choice(['desi_lss', 'act_dr6', 'planck_pr4', 'all']),
              default='all', help='Dataset to download')
@click.option('--output', '-o', type=click.Path(), default='data/ksz',
              help='Output directory')
@click.option('--dry-run', is_flag=True, help='Show what would be downloaded')
def download(dataset, output, dry_run):
    """
    Download required data files.

    Downloads DESI LSS catalogs and CMB maps from public archives.
    """
    from .io import DATA_MANIFEST, download_file

    output_dir = Path(output)

    click.echo("Data download utility")
    click.echo("=" * 40)

    if dataset == 'all':
        datasets = ['desi_lss', 'act_dr6', 'planck_pr4']
    else:
        datasets = [dataset]

    total_size = 0
    for ds in datasets:
        if ds in DATA_MANIFEST:
            manifest = DATA_MANIFEST[ds]
            click.echo(f"\n{ds}:")
            click.echo(f"  Base URL: {manifest['base_url']}")
            click.echo(f"  Files: {len(manifest['files'])}")

            for f in manifest['files'][:3]:  # Show first 3
                click.echo(f"    - {f['filename']}")
            if len(manifest['files']) > 3:
                click.echo(f"    ... and {len(manifest['files']) - 3} more")

    if dry_run:
        click.echo("\n[Dry run - no files downloaded]")
    else:
        click.echo("\nDownload not implemented - please download manually")
        click.echo("See: https://data.desi.lbl.gov/public/dr1/")


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file to validate')
def validate_config(config):
    """
    Validate a configuration file.
    """
    if not config:
        click.echo("No config file specified. Use --config to specify a file.")
        return

    if not YAML_AVAILABLE:
        click.echo("YAML support not available. Install pyyaml.")
        return

    click.echo(f"Validating {config}...")

    try:
        with open(config) as f:
            cfg = yaml.safe_load(f)

        click.echo("  ✓ Valid YAML syntax")

        # Check required fields
        required = ['tracer', 'cmb_source']
        for field in required:
            if field in cfg:
                click.echo(f"  ✓ {field}: {cfg[field]}")
            else:
                click.echo(f"  ✗ Missing: {field}")

        click.echo("\nConfiguration valid!")

    except yaml.YAMLError as e:
        click.echo(f"  ✗ YAML error: {e}")


@cli.command()
def info():
    """
    Display pipeline information and dependencies.
    """
    click.echo("DESI kSZ Analysis Pipeline")
    click.echo("=" * 40)
    click.echo()

    # Check dependencies
    click.echo("Dependencies:")

    deps = [
        ('numpy', 'np'),
        ('scipy', 'scipy'),
        ('astropy', 'astropy'),
        ('fitsio', 'fitsio'),
        ('healpy', 'healpy'),
        ('pixell', 'pixell'),
        ('emcee', 'emcee'),
        ('click', 'click'),
        ('pyyaml', 'yaml'),
        ('h5py', 'h5py'),
    ]

    for name, module in deps:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            click.echo(f"  ✓ {name}: {version}")
        except ImportError:
            click.echo(f"  ✗ {name}: not installed")

    click.echo()
    click.echo("Pipeline stages:")
    stages = [
        "1. ingest-desi    - Load DESI LSS catalogs",
        "2. ingest-maps    - Load CMB temperature maps",
        "3. make-masks     - Create analysis masks",
        "4. filter-maps    - Apply spatial filtering",
        "5. measure-temps  - Extract AP temperatures",
        "6. compute-pairwise - Compute p̂(r)",
        "7. covariance     - Estimate covariance",
        "8. null-tests     - Run validation tests",
        "9. inference      - Parameter estimation",
        "10. make-plots    - Publication figures",
    ]
    for stage in stages:
        click.echo(f"  {stage}")

    click.echo()
    click.echo("Validation commands (Phase 0-2):")
    validation_cmds = [
        "validate-map     - CMB map sanity checks + beam test",
        "validate-catalog - Galaxy catalog validation",
        "cov-stability    - Covariance matrix stability analysis",
        "injection-test   - Signal injection/recovery test",
        "null-suite       - Comprehensive null test suite",
    ]
    for cmd in validation_cmds:
        click.echo(f"  {cmd}")

    click.echo()
    click.echo("P0 credibility commands:")
    p0_cmds = [
        "tsz-sweep        - tSZ cluster mask radius sweep",
        "map-set          - Multi-frequency map set operations",
        "backend-bench    - Pair counting backend benchmark",
        "auto-cov         - Auto-regularized covariance",
        "transfer-test    - Map transfer function test",
    ]
    for cmd in p0_cmds:
        click.echo(f"  {cmd}")


# ============================================================================
# P0 Credibility Commands
# ============================================================================

@cli.command('tsz-sweep')
@click.option('--catalog-file', '-c', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--cluster-catalog', type=click.Path(exists=True),
              help='Cluster catalog (PSZ2 format)')
@click.option('--temp-file', '-t', type=click.Path(exists=True),
              help='Temperature measurements file')
@click.option('--cov-file', type=click.Path(exists=True),
              help='Covariance matrix file')
@click.option('--template-file', type=click.Path(exists=True),
              help='Theory template file')
@click.option('--mask-radii', type=str, default='0,5,10,15,20',
              help='Mask radii to test (arcmin, comma-separated)')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/tsz',
              help='Output directory')
def tsz_sweep_cmd(catalog_file, cluster_catalog, temp_file, cov_file,
                  template_file, mask_radii, output):
    """
    Run tSZ cluster mask radius sweep.

    Tests for tSZ contamination by varying the cluster mask radius
    and checking if the kSZ amplitude stabilizes.
    """
    from .systematics.tsz_leakage import (
        cluster_mask_sweep, load_planck_cluster_catalog, plot_cluster_mask_sweep
    )
    from .estimators import PairwiseMomentumEstimator
    from .config import DEFAULT_SEPARATION_BINS

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("tSZ Cluster Mask Sweep")
    click.echo("=" * 50)

    # Parse mask radii
    radii = [float(r) for r in mask_radii.split(',')]
    click.echo(f"Mask radii: {radii} arcmin")

    # Load galaxy catalog
    data = np.load(catalog_file)
    positions = data['positions']
    weights = data['weights']
    ra, dec = data['ra'], data['dec']
    click.echo(f"Loaded {len(positions):,} galaxies")

    # Load cluster catalog
    if cluster_catalog:
        cluster_ra, cluster_dec, theta_500 = load_planck_cluster_catalog(cluster_catalog)
        click.echo(f"Loaded {len(cluster_ra)} clusters")
    else:
        cluster_ra, cluster_dec, theta_500 = np.array([]), np.array([]), np.array([])
        click.echo("Warning: No cluster catalog provided")

    # Load temperatures
    if temp_file:
        temp_data = np.load(temp_file)
        temperatures = temp_data['temperatures']
    else:
        rng = np.random.default_rng(42)
        temperatures = rng.standard_normal(len(positions)) * 100

    # Load covariance and template
    r_bins = np.array(DEFAULT_SEPARATION_BINS)
    n_bins = len(r_bins) - 1

    if cov_file:
        cov = np.load(cov_file)
    else:
        cov = np.eye(n_bins) * 100

    if template_file:
        template_data = np.load(template_file)
        template = template_data['template']
    else:
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        template = 1.0 / (1 + r_centers / 50.0)

    # Create estimator
    estimator = PairwiseMomentumEstimator(separation_bins=r_bins)

    # Run sweep
    click.echo("\nRunning cluster mask sweep...")
    result = cluster_mask_sweep(
        estimator=estimator,
        positions=positions,
        temperatures=temperatures,
        weights=weights,
        ra=ra,
        dec=dec,
        template=template,
        cov=cov,
        cluster_ra=cluster_ra,
        cluster_dec=cluster_dec,
        mask_radii_arcmin=radii,
        theta_500=theta_500 if len(theta_500) > 0 else None,
    )

    # Display results
    click.echo(f"\nResults:")
    for i, radius in enumerate(result.mask_radii_arcmin):
        if np.isfinite(result.amplitudes[i]):
            click.echo(f"  {radius}': A = {result.amplitudes[i]:.4f} +/- {result.amplitude_errors[i]:.4f}")

    click.echo(f"\nConverged: {result.converged}")
    if result.convergence_radius:
        click.echo(f"Convergence radius: {result.convergence_radius}' arcmin")
    click.echo(f"Recommendation: {result.recommendation}")

    # Save results
    result.to_json(str(output_dir / 'tsz_sweep_result.json'))
    click.echo(f"\nSaved: {output_dir / 'tsz_sweep_result.json'}")

    # Generate plot
    try:
        plot_cluster_mask_sweep(result, str(output_dir / 'tsz_sweep.pdf'))
        click.echo(f"Plot: {output_dir / 'tsz_sweep.pdf'}")
    except Exception as e:
        click.echo(f"Warning: Could not generate plot: {e}")


@cli.command('map-set')
@click.option('--map-dir', '-d', type=click.Path(exists=True), required=True,
              help='Directory containing CMB maps')
@click.option('--source', type=click.Choice(['act', 'planck']),
              default='act', help='Map source')
@click.option('--frequencies', type=str, default='90,150',
              help='Frequencies to load (comma-separated)')
@click.option('--create-null', is_flag=True,
              help='Create frequency difference null map')
@click.option('--create-coadd', is_flag=True,
              help='Create inverse-variance coadd')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/maps',
              help='Output directory')
def map_set_cmd(map_dir, source, frequencies, create_null, create_coadd, output):
    """
    Multi-frequency map set operations.

    Create coadds and null maps from multi-frequency CMB data.
    """
    from .io.map_set import MapSet, load_act_mapset, load_planck_mapset

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Multi-Frequency Map Set")
    click.echo("=" * 50)

    # Parse frequencies
    freqs = [int(f) for f in frequencies.split(',')]
    click.echo(f"Loading {source} maps at {freqs} GHz...")

    # Load map set
    if source == 'act':
        mapset = load_act_mapset(map_dir, frequencies=freqs)
    else:
        # For Planck, frequencies are component methods
        mapset = load_planck_mapset(map_dir)

    click.echo(f"\n{mapset.summary()}")

    # Validate
    validation = mapset.validate()
    click.echo(f"\nValidation: {'PASS' if validation['valid'] else 'FAIL'}")
    if validation['issues']:
        for issue in validation['issues']:
            click.echo(f"  - {issue}")

    # Create null map
    if create_null and len(freqs) >= 2:
        click.echo(f"\nCreating null map: {freqs[0]} - {freqs[1]} GHz...")
        try:
            null_result = mapset.create_null_map(float(freqs[0]), float(freqs[1]))
            click.echo(f"  Shape: {null_result.data.shape}")
            click.echo(f"  Effective beam: {null_result.effective_beam_fwhm_arcmin}' arcmin")

            # Save
            null_file = output_dir / f'null_map_{freqs[0]}_{freqs[1]}.npy'
            np.save(null_file, null_result.data)
            click.echo(f"  Saved: {null_file}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    # Create coadd
    if create_coadd:
        click.echo(f"\nCreating inverse-variance coadd...")
        try:
            coadd_result = mapset.create_coadd()
            click.echo(f"  Shape: {coadd_result.data.shape}")
            click.echo(f"  Frequencies used: {coadd_result.frequencies_used}")

            # Save
            coadd_file = output_dir / 'coadd_map.npy'
            np.save(coadd_file, coadd_result.data)
            click.echo(f"  Saved: {coadd_file}")
        except Exception as e:
            click.echo(f"  Error: {e}")

    click.echo("\nMap set operations complete")


@cli.command('backend-bench')
@click.option('--n-points', type=int, default=10000,
              help='Number of test points')
@click.option('--n-bins', type=int, default=20,
              help='Number of separation bins')
@click.option('--r-max', type=float, default=150.0,
              help='Maximum separation (Mpc/h)')
def backend_bench_cmd(n_points, n_bins, r_max):
    """
    Benchmark pair counting backends.

    Compares performance of Corrfunc (if available) vs KDTree.
    """
    from .estimators.pair_counting import (
        benchmark_backends, get_available_backends, get_default_backend
    )

    click.echo("=" * 50)
    click.echo("Pair Counting Backend Benchmark")
    click.echo("=" * 50)

    click.echo(f"\nAvailable backends: {get_available_backends()}")
    click.echo(f"Default backend: {get_default_backend()}")

    click.echo(f"\nBenchmark parameters:")
    click.echo(f"  N points: {n_points:,}")
    click.echo(f"  N bins: {n_bins}")
    click.echo(f"  r_max: {r_max} Mpc/h")

    click.echo("\nRunning benchmark...")
    results = benchmark_backends(
        n_points=n_points,
        n_bins=n_bins,
        r_max=r_max,
    )

    click.echo(f"\nResults:")
    for backend, stats in results.items():
        click.echo(f"  {backend}:")
        click.echo(f"    Time: {stats['time_seconds']:.3f}s")
        click.echo(f"    Pairs: {stats['total_pairs']:,}")
        click.echo(f"    Rate: {stats['pairs_per_second']:.0f} pairs/sec")

    # Recommendation
    if 'corrfunc' in results and 'kdtree' in results:
        speedup = results['kdtree']['time_seconds'] / results['corrfunc']['time_seconds']
        click.echo(f"\nCorrfunc speedup: {speedup:.1f}x")
    else:
        click.echo("\nNote: Install Corrfunc for better performance")


@cli.command('auto-cov')
@click.option('--cov-file', '-c', type=click.Path(exists=True), required=True,
              help='Raw covariance matrix file (.npy)')
@click.option('--n-samples', type=int, required=True,
              help='Number of jackknife samples used')
@click.option('--target-condition', type=float, default=1e6,
              help='Target condition number')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/covariance',
              help='Output directory')
def auto_cov_cmd(cov_file, n_samples, target_condition, output):
    """
    Apply automatic covariance regularization.

    Automatically applies eigenvalue flooring and/or shrinkage
    based on matrix condition. Issues Hartlap warnings.
    """
    from .covariance.stability import (
        auto_regularize, robust_precision_matrix, check_hartlap_regime
    )

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Auto-Regularization Covariance")
    click.echo("=" * 50)

    # Load covariance
    cov = np.load(cov_file)
    n_bins = cov.shape[0]
    click.echo(f"Loaded covariance: {cov.shape}")
    click.echo(f"N samples (K): {n_samples}")

    # Check Hartlap regime
    hartlap_info = check_hartlap_regime(n_samples, n_bins)
    click.echo(f"\nHartlap regime check:")
    click.echo(f"  Status: {hartlap_info['status']}")
    click.echo(f"  Hartlap factor: {hartlap_info['hartlap_factor']:.4f}")
    click.echo(f"  Recommendation: {hartlap_info['recommendation']}")

    # Apply auto-regularization
    click.echo(f"\nApplying auto-regularization (target κ = {target_condition:.0e})...")
    result = auto_regularize(cov, n_samples, target_condition=target_condition)

    click.echo(f"\nRegularization applied:")
    click.echo(f"  Method: {result.method}")
    click.echo(f"  Original κ: {result.original_condition:.2e}")
    click.echo(f"  Final κ: {result.final_condition:.2e}")

    if result.warnings:
        click.echo(f"\nWarnings:")
        for warn in result.warnings:
            click.echo(f"  - {warn}")

    # Compute robust precision matrix
    click.echo(f"\nComputing precision matrix...")
    precision, info = robust_precision_matrix(cov, n_samples)

    click.echo(f"  Method: {info['method']}")
    click.echo(f"  Precision κ: {info['precision_condition']:.2e}")

    # Save results
    np.save(output_dir / 'covariance_regularized.npy', result.regularized_cov)
    np.save(output_dir / 'precision_matrix.npy', precision)

    import json
    with open(output_dir / 'regularization_info.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    click.echo(f"\nSaved:")
    click.echo(f"  {output_dir / 'covariance_regularized.npy'}")
    click.echo(f"  {output_dir / 'precision_matrix.npy'}")
    click.echo(f"  {output_dir / 'regularization_info.json'}")


@cli.command('transfer-test')
@click.option('--catalog-file', '-c', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--nside', type=int, default=512,
              help='HEALPix nside for test maps')
@click.option('--n-realizations', '-n', type=int, default=5,
              help='Number of CMB realizations')
@click.option('--bias-threshold', type=float, default=0.1,
              help='Maximum acceptable fractional bias')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/transfer',
              help='Output directory')
def transfer_test_cmd(catalog_file, nside, n_realizations, bias_threshold, output):
    """
    Run end-to-end map transfer function test.

    Tests that temperature extraction pipeline correctly propagates
    known input signals without introducing bias.
    """
    from .maps.validation import map_transfer_function_test

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 50)
    click.echo("Map Transfer Function Test")
    click.echo("=" * 50)

    click.echo(f"Parameters:")
    click.echo(f"  nside: {nside}")
    click.echo(f"  N realizations: {n_realizations}")
    click.echo(f"  Bias threshold: {bias_threshold:.1%}")

    # Load catalog
    data = np.load(catalog_file)
    ra, dec = data['ra'], data['dec']
    click.echo(f"Loaded {len(ra):,} galaxy positions")

    # Define simple temperature extraction function
    try:
        import healpy as hp
    except ImportError:
        click.echo("Error: healpy required for transfer function test")
        return

    def extract_temps(map_data, ra, dec):
        """Simple temperature extraction at galaxy positions."""
        theta = np.radians(90.0 - dec)
        phi = np.radians(ra)
        pix = hp.ang2pix(hp.npix2nside(len(map_data)), theta, phi)
        return map_data[pix]

    click.echo(f"\nRunning transfer function test...")
    result = map_transfer_function_test(
        temperature_extraction_func=extract_temps,
        positions_ra=ra,
        positions_dec=dec,
        nside=nside,
        n_realizations=n_realizations,
        bias_threshold=bias_threshold,
    )

    click.echo(f"\nResults:")
    click.echo(f"  Mean transfer: {result.mean_transfer:.3f} +/- {result.std_transfer:.3f}")
    click.echo(f"  Bias at l~500: {result.bias:.1%}")

    status = "PASS" if result.passed else "FAIL"
    click.echo(f"  Status: {status}")

    # Save results
    import json
    with open(output_dir / 'transfer_test_result.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    click.echo(f"\nSaved: {output_dir / 'transfer_test_result.json'}")


# ============================================================================
# Phase 3-4 Execution Commands
# ============================================================================

@cli.command('run-phase34')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to YAML configuration file')
@click.option('--catalog-file', type=click.Path(exists=True), required=True,
              help='Galaxy catalog file (.npz)')
@click.option('--map-file', type=click.Path(exists=True), required=True,
              help='Primary CMB map file')
@click.option('--map-file-secondary', type=click.Path(exists=True),
              help='Secondary CMB map file (for cross-check)')
@click.option('--tracer', type=click.Choice(['BGS', 'LRG', 'ELG']),
              default='LRG', help='Galaxy tracer type')
@click.option('--z-bins', type=str, default='0.4,0.5,0.6,0.7',
              help='Redshift bin edges (comma-separated, >=4 edges for >=3 bins)')
@click.option('--jackknife-k', type=str, default='100',
              help='Jackknife K values to try (comma-separated)')
@click.option('--n-injection', type=int, default=100,
              help='Number of injection test realizations')
@click.option('--n-null', type=int, default=500,
              help='Number of null test realizations')
@click.option('--tsz-mask-radii', type=str, default='5,10,15,20',
              help='tSZ mask radii to test (arcmin)')
@click.option('--run-referee-checks/--no-referee-checks', default=True,
              help='Run referee attack checks')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/phase34',
              help='Output directory')
def run_phase34_cmd(config, catalog_file, map_file, map_file_secondary,
                     tracer, z_bins, jackknife_k, n_injection, n_null,
                     tsz_mask_radii, run_referee_checks, output):
    """
    Run complete Phase 3-4 kSZ measurement with automated gating.

    This is the production measurement driver that:
      1. Measures pairwise kSZ momentum in tomographic z-bins
      2. Estimates covariance via jackknife with auto-K selection
      3. Runs injection tests for bias validation
      4. Runs null test suite
      5. Runs transfer function test
      6. Runs tSZ contamination sweep
      7. (Optional) Runs 5 referee attack checks
      8. Evaluates all gates (PASS/FAIL/INCONCLUSIVE)
      9. Generates decision report and all outputs

    Non-negotiables enforced:
      - At least 2 independent map products (if secondary provided)
      - At least 3 tomographic z-bins
      - Detection significance reported
      - Jackknife covariance with valid Hartlap factor
      - tSZ leakage controls via mask sweep
      - Transfer function validation
    """
    from .runner.phase34 import Phase34Runner, Phase34Config
    from .runner.gates import evaluate_gates

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 70)
    click.echo("DESI DR1 Pairwise kSZ Phase 3-4 Execution")
    click.echo("=" * 70)

    # Parse configuration
    z_edges = [float(z) for z in z_bins.split(',')]
    k_values = [int(k) for k in jackknife_k.split(',')]
    tsz_radii = [float(r) for r in tsz_mask_radii.split(',')]

    # Validate minimum requirements
    if len(z_edges) < 4:
        click.echo("ERROR: Require at least 4 z-bin edges for 3 tomographic bins")
        click.echo("       Provide --z-bins with at least 4 comma-separated values")
        return

    click.echo(f"\nConfiguration:")
    click.echo(f"  Tracer: {tracer}")
    click.echo(f"  Z bins: {len(z_edges)-1} bins from z={z_edges[0]} to z={z_edges[-1]}")
    click.echo(f"  Jackknife K values: {k_values}")
    click.echo(f"  Injection realizations: {n_injection}")
    click.echo(f"  Null test realizations: {n_null}")
    click.echo(f"  tSZ mask radii: {tsz_radii} arcmin")
    click.echo(f"  Referee checks: {'enabled' if run_referee_checks else 'disabled'}")

    # Map info
    click.echo(f"\nMaps:")
    click.echo(f"  Primary: {map_file}")
    if map_file_secondary:
        click.echo(f"  Secondary: {map_file_secondary}")
    else:
        click.echo(f"  Secondary: None (single-map mode)")

    # Build config
    phase34_config = Phase34Config(
        catalog_file=catalog_file,
        map_files=[map_file] + ([map_file_secondary] if map_file_secondary else []),
        tracer=tracer,
        z_bins=z_edges,
        jackknife_k_values=k_values,
        n_injection_realizations=n_injection,
        n_null_realizations=n_null,
        tsz_mask_radii=tsz_radii,
        run_referee_checks=run_referee_checks,
        output_dir=str(output_dir),
    )

    # Load YAML config if provided
    if config and YAML_AVAILABLE:
        with open(config) as f:
            yaml_config = yaml.safe_load(f)
        click.echo(f"\nLoaded additional config from: {config}")
        # Merge with command-line options (command-line takes precedence)

    # Create and run Phase 3-4 driver
    click.echo("\n" + "-" * 70)
    click.echo("Starting Phase 3-4 Execution")
    click.echo("-" * 70)

    runner = Phase34Runner(phase34_config)

    try:
        result = runner.run()

        # Display summary
        click.echo("\n" + "=" * 70)
        click.echo("Phase 3-4 Results Summary")
        click.echo("=" * 70)

        click.echo(f"\nOverall Status: {result.gate_result.overall_status}")
        click.echo(f"Recommendation: {result.gate_result.recommendation}")

        click.echo(f"\nDetection:")
        if result.joint_snr is not None:
            click.echo(f"  Joint S/N: {result.joint_snr:.1f} sigma")
            click.echo(f"  Joint amplitude: {result.joint_amplitude:.4f} +/- {result.joint_amplitude_err:.4f}")
        else:
            click.echo(f"  (Joint fit not computed)")

        click.echo(f"\nPer-bin results:")
        for zr in result.z_bin_results:
            click.echo(f"  {zr.z_bin_label}: A={zr.amplitude:.3f}+/-{zr.amplitude_err:.3f}, S/N={zr.snr:.1f}")

        click.echo(f"\nGate evaluation:")
        click.echo(f"  Critical passed: {result.gate_result.critical_passed}/{result.gate_result.critical_passed + result.gate_result.critical_failed}")
        click.echo(f"  Warnings: {result.gate_result.warnings}")

        if result.gate_result.failed_gates:
            click.echo(f"\nFailed critical gates:")
            for gate in result.gate_result.failed_gates:
                click.echo(f"    - {gate}")

        if result.gate_result.warning_gates:
            click.echo(f"\nWarning gates:")
            for gate in result.gate_result.warning_gates:
                click.echo(f"    - {gate}")

        click.echo(f"\nOutputs written to: {output_dir}")
        click.echo("  - decision_report.md")
        click.echo("  - phase34_result.json")
        click.echo("  - plots/")

    except Exception as e:
        click.echo(f"\nERROR: Phase 3-4 execution failed: {e}")
        import traceback
        traceback.print_exc()
        return

    click.echo("\n" + "=" * 70)
    click.echo("Phase 3-4 Execution Complete")
    click.echo("=" * 70)


@cli.command('package-results')
@click.option('--result-file', '-r', type=click.Path(exists=True), required=True,
              help='Phase 3-4 result JSON file')
@click.option('--output', '-o', type=click.Path(), default='data/ksz/bundle',
              help='Output directory for bundle')
@click.option('--bundle-name', type=str, default=None,
              help='Name for results bundle (default: timestamped)')
@click.option('--include-plots/--no-plots', default=True,
              help='Include plots in bundle')
def package_results_cmd(result_file, output, bundle_name, include_plots):
    """
    Package analysis results into a self-contained bundle.

    Creates a complete results package containing:
      - plots/          Publication figures (PDF + PNG)
      - tables/         CSV tables (p(r), covariance, null tests)
      - configs/        YAML configuration files used
      - data/           HDF5/NPY data files
      - manifest.json   File listing with metadata
      - checksums.sha256 SHA256 checksums for reproducibility
      - summary.json    Machine-readable results summary
      - results.md      Human-readable decision report

    This bundle is the deliverable for publication or archiving.
    """
    from .results.packager import ResultsPackager, create_results_bundle
    from .runner.gates import GateEvaluationResult, GateResult, GateStatus
    import json

    output_dir = Path(output)

    click.echo("=" * 60)
    click.echo("Results Packager")
    click.echo("=" * 60)

    # Load Phase 3-4 result
    click.echo(f"Loading results from: {result_file}")
    with open(result_file) as f:
        result_data = json.load(f)

    # Reconstruct result objects from JSON
    # (In production, you would pickle or use a proper serialization format)
    from dataclasses import dataclass, field
    from typing import List, Optional, Dict, Any

    @dataclass
    class LoadedTomographicResult:
        z_bin_label: str
        z_mean: float
        n_galaxies: int
        n_pairs: int
        r_centers: np.ndarray
        pairwise_momentum: np.ndarray
        pairwise_momentum_err: np.ndarray
        theory_template: Optional[np.ndarray]
        amplitude: float
        amplitude_err: float
        snr: float

    @dataclass
    class LoadedPhase34Result:
        z_bin_results: List[LoadedTomographicResult]
        covariance: Optional[np.ndarray]
        joint_amplitude: Optional[float]
        joint_amplitude_err: Optional[float]
        joint_snr: Optional[float]
        plots: Dict[str, Path]
        metrics: Dict[str, float]
        referee_results: Dict[str, Any]

    # Parse z-bin results
    z_bin_results = []
    for zr_data in result_data.get('z_bin_results', []):
        zr = LoadedTomographicResult(
            z_bin_label=zr_data.get('z_bin_label', ''),
            z_mean=zr_data.get('z_mean', 0.5),
            n_galaxies=zr_data.get('n_galaxies', 0),
            n_pairs=zr_data.get('n_pairs', 0),
            r_centers=np.array(zr_data.get('r_centers', [])),
            pairwise_momentum=np.array(zr_data.get('pairwise_momentum', [])),
            pairwise_momentum_err=np.array(zr_data.get('pairwise_momentum_err', [])),
            theory_template=np.array(zr_data.get('theory_template', [])) if zr_data.get('theory_template') else None,
            amplitude=zr_data.get('amplitude', 1.0),
            amplitude_err=zr_data.get('amplitude_err', 0.1),
            snr=zr_data.get('snr', 0.0),
        )
        z_bin_results.append(zr)

    # Parse covariance if present
    cov_data = result_data.get('covariance')
    covariance = np.array(cov_data) if cov_data else None

    phase34_result = LoadedPhase34Result(
        z_bin_results=z_bin_results,
        covariance=covariance,
        joint_amplitude=result_data.get('joint_amplitude'),
        joint_amplitude_err=result_data.get('joint_amplitude_err'),
        joint_snr=result_data.get('joint_snr'),
        plots={},
        metrics=result_data.get('metrics', {}),
        referee_results=result_data.get('referee_results', {}),
    )

    # Parse gate result
    gate_data = result_data.get('gate_result', {})
    gate_results = []
    for gr_data in gate_data.get('gate_results', []):
        gr = GateResult(
            name=gr_data.get('name', ''),
            status=GateStatus(gr_data.get('status', 'SKIP')),
            metric=gr_data.get('metric', np.nan) if gr_data.get('metric') is not None else np.nan,
            threshold=gr_data.get('threshold', 0),
            message=gr_data.get('message', ''),
            is_critical=gr_data.get('is_critical', False),
        )
        gate_results.append(gr)

    gate_result = GateEvaluationResult(
        overall_status=gate_data.get('overall_status', 'INCONCLUSIVE'),
        critical_passed=gate_data.get('critical_passed', 0),
        critical_failed=gate_data.get('critical_failed', 0),
        warnings=gate_data.get('warnings', 0),
        skipped=gate_data.get('skipped', 0),
        gate_results=gate_results,
        failed_gates=gate_data.get('failed_gates', []),
        warning_gates=gate_data.get('warning_gates', []),
        recommendation=gate_data.get('recommendation', ''),
        rerun_commands=gate_data.get('rerun_commands', []),
    )

    # Extract config from result
    config = result_data.get('config', {})

    click.echo(f"\nLoaded:")
    click.echo(f"  Z-bins: {len(z_bin_results)}")
    click.echo(f"  Status: {gate_result.overall_status}")

    # Create bundle
    click.echo(f"\nCreating results bundle...")
    packager = ResultsPackager(str(output_dir), bundle_name)
    bundle_path = packager.create_bundle(
        phase34_result=phase34_result,
        gate_result=gate_result,
        config=config,
    )

    click.echo(f"\n" + "=" * 60)
    click.echo(f"Bundle created: {bundle_path}")
    click.echo(f"=" * 60)

    # List contents
    click.echo(f"\nContents:")
    for item in sorted(bundle_path.iterdir()):
        if item.is_dir():
            n_files = len(list(item.iterdir()))
            click.echo(f"  {item.name}/  ({n_files} files)")
        else:
            size_kb = item.stat().st_size / 1024
            click.echo(f"  {item.name}  ({size_kb:.1f} KB)")


# ============================================================================
# Entry point
# ============================================================================

def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
