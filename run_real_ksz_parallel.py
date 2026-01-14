#!/usr/bin/env python3
"""
DESI DR1 x Planck PR3 Pairwise kSZ Analysis - PARALLELIZED
============================================================
Real data, real measurement, proper parallelization.
"""

import numpy as np
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


# Must be at module level to be picklable
def process_chunk_standalone(chunk_info):
    """Process a chunk of galaxies for pair counting."""
    chunk_indices, pos_data, T_data, w_data, r_hat_data, r_edges = chunk_info

    tree_local = cKDTree(pos_data)
    n_bins = len(r_edges) - 1
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in chunk_indices:
        neighbors = tree_local.query_ball_point(pos_data[i], r_edges[-1])

        for j in neighbors:
            if j <= i:
                continue

            dr = pos_data[j] - pos_data[i]
            r = np.sqrt(np.sum(dr**2))

            if r < r_edges[0]:
                continue

            bin_idx = np.searchsorted(r_edges, r) - 1
            if bin_idx < 0 or bin_idx >= n_bins:
                continue

            r_hat_ij = dr / r
            c_ij = 0.5 * np.dot(r_hat_ij, r_hat_data[i] - r_hat_data[j])

            ww = w_data[i] * w_data[j]
            dT = T_data[i] - T_data[j]

            num[bin_idx] += ww * dT * c_ij
            den[bin_idx] += ww * c_ij**2
            counts[bin_idx] += 1

    return num, den, counts


def compute_pairwise_parallel(pos, T, w, r_bin_edges, n_workers=None, subsample=None):
    """
    Compute pairwise kSZ momentum with parallel processing.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    n = len(pos)

    # Subsample for speed if requested
    if subsample is not None and subsample < n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=subsample, replace=False)
        pos = pos[idx]
        T = T[idx]
        w = w[idx]
        n = len(pos)
        print(f"    Subsampled to {n:,} galaxies")

    # Unit vectors
    r_mag = np.sqrt(np.sum(pos**2, axis=1))
    r_hat = pos / r_mag[:, np.newaxis]

    # Split indices into chunks
    chunk_size = max(100, n // (n_workers * 10))
    chunks = [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]

    print(f"    Using {n_workers} workers, {len(chunks)} chunks")

    # Prepare arguments
    args_list = [(chunk, pos, T, w, r_hat, r_bin_edges) for chunk in chunks]

    # Process in parallel
    n_bins = len(r_bin_edges) - 1
    total_num = np.zeros(n_bins)
    total_den = np.zeros(n_bins)
    total_counts = np.zeros(n_bins, dtype=np.int64)

    with Pool(n_workers) as pool:
        results = pool.map(process_chunk_standalone, args_list)

    # Aggregate results
    for num, den, counts in results:
        total_num += num
        total_den += den
        total_counts += counts

    # Compute estimator
    with np.errstate(divide='ignore', invalid='ignore'):
        p_ksz = np.where(total_den > 0, total_num / total_den, 0)

    return p_ksz, total_counts, total_num, total_den


def theory_template(r, A=1.0, r0=50.0):
    """Simple theory template: p(r) = A * exp(-r/r0)"""
    return A * np.exp(-r / r0)


if __name__ == '__main__':
    print("=" * 70)
    print("DESI DR1 x Planck SMICA Pairwise kSZ Analysis (PARALLEL)")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"CPUs available: {cpu_count()}")
    print()

    # =========================================================================
    # 1. Load DESI DR1 LRG Catalogs
    # =========================================================================
    print("[1/8] Loading DESI DR1 LRG catalogs...")

    try:
        from astropy.io import fits
    except ImportError:
        import fitsio as fits
        USE_FITSIO = True
    else:
        USE_FITSIO = False

    ngc_file = Path("data/ksz/catalogs/LRG_NGC_clustering.dat.fits")
    sgc_file = Path("data/ksz/catalogs/LRG_SGC_clustering.dat.fits")

    if USE_FITSIO:
        ngc_data = fits.read(str(ngc_file))
        sgc_data = fits.read(str(sgc_file))
    else:
        with fits.open(ngc_file) as hdu:
            ngc_data = hdu[1].data
        with fits.open(sgc_file) as hdu:
            sgc_data = hdu[1].data

    ra = np.concatenate([ngc_data['RA'], sgc_data['RA']])
    dec = np.concatenate([ngc_data['DEC'], sgc_data['DEC']])
    z = np.concatenate([ngc_data['Z'], sgc_data['Z']])

    weight_sys = np.concatenate([ngc_data['WEIGHT_SYS'], sgc_data['WEIGHT_SYS']])
    weight_comp = np.concatenate([ngc_data['WEIGHT_COMP'], sgc_data['WEIGHT_COMP']])
    weight_zfail = np.concatenate([ngc_data['WEIGHT_ZFAIL'], sgc_data['WEIGHT_ZFAIL']])
    weights = weight_sys * weight_comp * weight_zfail

    print(f"  Total LRGs: {len(ra):,}")
    print(f"  NGC: {len(ngc_data):,}, SGC: {len(sgc_data):,}")
    print(f"  Redshift range: {z.min():.3f} - {z.max():.3f}")
    print(f"  Mean weight: {weights.mean():.3f}")
    print()

    # =========================================================================
    # 2. Load Planck SMICA CMB Map
    # =========================================================================
    print("[2/8] Loading Planck SMICA CMB map...")

    try:
        import healpy as hp
    except ImportError:
        print("ERROR: healpy required. Install with: pip install healpy")
        sys.exit(1)

    planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")
    cmb_map = hp.read_map(str(planck_file), field=0, verbose=False)
    nside = hp.npix2nside(len(cmb_map))

    cmb_map = cmb_map * 1e6  # Convert to μK

    print(f"  NSIDE: {nside}")
    print(f"  N pixels: {len(cmb_map):,}")
    print(f"  Map range: {cmb_map[cmb_map != hp.UNSEEN].min():.1f} to {cmb_map[cmb_map != hp.UNSEEN].max():.1f} μK")
    print()

    # =========================================================================
    # 3. Compute Comoving Positions
    # =========================================================================
    print("[3/8] Computing comoving positions...")

    from astropy.cosmology import Planck18
    cosmo = Planck18

    h = cosmo.H0.value / 100
    chi = cosmo.comoving_distance(z).value * h

    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    z_pos = chi * np.sin(dec_rad)
    positions = np.column_stack([x, y, z_pos])

    print(f"  Comoving distance range: {chi.min():.1f} - {chi.max():.1f} Mpc/h")
    print()

    # =========================================================================
    # 4. Extract CMB Temperatures at Galaxy Positions
    # =========================================================================
    print("[4/8] Extracting CMB temperatures at galaxy positions...")

    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)
    temperatures = cmb_map[pix]

    good = temperatures != hp.UNSEEN
    print(f"  Valid temperatures: {good.sum():,} / {len(temperatures):,}")

    ra = ra[good]
    dec = dec[good]
    z = z[good]
    weights = weights[good]
    positions = positions[good]
    temperatures = temperatures[good]
    chi = chi[good]

    print(f"  Temperature mean: {temperatures.mean():.2f} μK")
    print(f"  Temperature std: {temperatures.std():.2f} μK")
    print()

    # =========================================================================
    # 5. Define Redshift Bins and Separation Bins
    # =========================================================================
    print("[5/8] Setting up tomographic bins...")

    z_sorted = np.sort(z)
    n_gal = len(z)
    z_bin_edges = [
        z.min(),
        z_sorted[n_gal // 3],
        z_sorted[2 * n_gal // 3],
        z.max() + 0.001
    ]

    r_min, r_max = 20.0, 150.0
    n_r_bins = 13
    r_bin_edges = np.linspace(r_min, r_max, n_r_bins + 1)
    r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

    print(f"  Z bins: {[f'{e:.3f}' for e in z_bin_edges]}")
    print(f"  R bins: {r_min} - {r_max} Mpc/h ({n_r_bins} bins)")
    print()

    # =========================================================================
    # 6. Parallelized Pairwise Momentum Estimator
    # =========================================================================
    print("[6/8] Computing pairwise momentum (PARALLELIZED)...")

    results = {
        'z_bins': [],
        'r_centers': r_centers.tolist(),
        'r_bin_edges': r_bin_edges.tolist(),
        'analysis_date': datetime.now().isoformat(),
    }

    all_p_ksz = []
    all_p_err = []
    all_n_gal = []
    all_n_pairs = []

    SUBSAMPLE_PER_BIN = 50000

    n_workers = max(1, cpu_count() - 1)
    print(f"  Using {n_workers} parallel workers")
    print(f"  Subsampling to {SUBSAMPLE_PER_BIN:,} galaxies per bin")

    for i_z in range(len(z_bin_edges) - 1):
        z_lo, z_hi = z_bin_edges[i_z], z_bin_edges[i_z + 1]
        z_mask = (z >= z_lo) & (z < z_hi)

        pos_bin = positions[z_mask]
        T_bin = temperatures[z_mask]
        w_bin = weights[z_mask]
        z_bin_arr = z[z_mask]

        n_gal_bin = len(pos_bin)
        z_mean = z_bin_arr.mean()

        print(f"\n  Z bin {i_z + 1}: {z_lo:.3f} < z < {z_hi:.3f}")
        print(f"    N galaxies (full): {n_gal_bin:,}")
        print(f"    Mean z: {z_mean:.3f}")

        t0 = time.time()
        p_ksz, pair_counts, num, denom = compute_pairwise_parallel(
            pos_bin, T_bin, w_bin, r_bin_edges,
            n_workers=n_workers,
            subsample=SUBSAMPLE_PER_BIN
        )
        dt = time.time() - t0

        print(f"    Pair counting time: {dt:.1f}s")
        print(f"    Total pairs: {pair_counts.sum():,}")
        print(f"    p(r) range: {p_ksz.min():.4f} to {p_ksz.max():.4f} μK")

        with np.errstate(divide='ignore', invalid='ignore'):
            p_err = np.where(pair_counts > 0,
                             np.abs(p_ksz) / np.sqrt(pair_counts) + 0.01,
                             0.1)

        all_p_ksz.append(p_ksz)
        all_p_err.append(p_err)
        all_n_gal.append(n_gal_bin)
        all_n_pairs.append(int(pair_counts.sum()))

        results['z_bins'].append({
            'z_lo': float(z_lo),
            'z_hi': float(z_hi),
            'z_mean': float(z_mean),
            'n_galaxies': int(n_gal_bin),
            'n_galaxies_used': int(min(n_gal_bin, SUBSAMPLE_PER_BIN)),
            'n_pairs': int(pair_counts.sum()),
            'p_ksz': p_ksz.tolist(),
            'p_err': p_err.tolist(),
            'pair_counts': pair_counts.tolist(),
        })

    print()

    # =========================================================================
    # 7. Theory Template and Amplitude Fitting
    # =========================================================================
    print("[7/8] Fitting amplitude to theory template...")

    amplitudes = []
    amplitude_errors = []
    snrs = []

    for i_z, zbin in enumerate(results['z_bins']):
        p_data = np.array(zbin['p_ksz'])
        p_err = np.array(zbin['p_err'])

        p_template = theory_template(r_centers, A=1.0, r0=60.0)

        valid = (p_err > 0) & np.isfinite(p_data) & (p_err < 10)
        if valid.sum() < 3:
            amplitudes.append(0.0)
            amplitude_errors.append(1.0)
            snrs.append(0.0)
            continue

        w = 1.0 / p_err[valid]**2
        A_fit = np.sum(w * p_data[valid] * p_template[valid]) / np.sum(w * p_template[valid]**2)
        A_err = 1.0 / np.sqrt(np.sum(w * p_template[valid]**2))

        amplitudes.append(float(A_fit))
        amplitude_errors.append(float(A_err))
        snrs.append(float(np.abs(A_fit) / A_err))

        zbin['amplitude'] = float(A_fit)
        zbin['amplitude_err'] = float(A_err)
        zbin['snr'] = float(np.abs(A_fit) / A_err)

        z_lo, z_hi = zbin['z_lo'], zbin['z_hi']
        print(f"  Z bin {i_z + 1} ({z_lo:.2f}-{z_hi:.2f}): A = {A_fit:.4f} ± {A_err:.4f}, S/N = {snrs[-1]:.2f}")

    # Joint fit
    all_p = np.concatenate([np.array(zb['p_ksz']) for zb in results['z_bins']])
    all_e = np.concatenate([np.array(zb['p_err']) for zb in results['z_bins']])
    all_template = np.tile(theory_template(r_centers, A=1.0, r0=60.0), len(results['z_bins']))

    valid = (all_e > 0) & np.isfinite(all_p) & (all_e < 10)
    w_joint = 1.0 / all_e[valid]**2
    A_joint = np.sum(w_joint * all_p[valid] * all_template[valid]) / np.sum(w_joint * all_template[valid]**2)
    A_joint_err = 1.0 / np.sqrt(np.sum(w_joint * all_template[valid]**2))
    joint_snr = np.abs(A_joint) / A_joint_err

    print(f"\n  JOINT FIT: A = {A_joint:.4f} ± {A_joint_err:.4f}")
    print(f"  JOINT S/N: {joint_snr:.2f} sigma")

    results['joint_amplitude'] = float(A_joint)
    results['joint_amplitude_err'] = float(A_joint_err)
    results['joint_snr'] = float(joint_snr)

    print()

    # =========================================================================
    # 8. Null Test: Shuffle Temperatures
    # =========================================================================
    print("[8/8] Running null test (temperature shuffle)...")

    n_null = 10
    null_amplitudes = []

    z_lo, z_hi = z_bin_edges[0], z_bin_edges[1]
    z_mask = (z >= z_lo) & (z < z_hi)
    pos_null = positions[z_mask]
    T_null_base = temperatures[z_mask].copy()
    w_null = weights[z_mask]

    rng = np.random.default_rng(42)

    for i_null in range(n_null):
        T_null = T_null_base.copy()
        rng.shuffle(T_null)

        p_null, _, _, _ = compute_pairwise_parallel(
            pos_null, T_null, w_null, r_bin_edges,
            n_workers=n_workers,
            subsample=20000
        )

        p_template = theory_template(r_centers, A=1.0, r0=60.0)
        p_err_null = np.full_like(p_null, 0.1)
        valid = np.isfinite(p_null)
        if valid.sum() >= 3:
            w = 1.0 / p_err_null[valid]**2
            A_null = np.sum(w * p_null[valid] * p_template[valid]) / np.sum(w * p_template[valid]**2)
            null_amplitudes.append(A_null)

        print(f"    Null iteration {i_null + 1}/{n_null}: A = {null_amplitudes[-1]:.4f}")

    null_amplitudes = np.array(null_amplitudes)
    null_mean = null_amplitudes.mean()
    null_std = null_amplitudes.std()

    print(f"\n  Null test: A_null = {null_mean:.4f} ± {null_std:.4f}")
    print(f"  Data amplitude (bin 1): {amplitudes[0]:.4f}")

    if null_std > 0:
        sig_vs_null = np.abs(amplitudes[0] - null_mean) / null_std
    else:
        sig_vs_null = 0.0
    print(f"  Significance vs null: {sig_vs_null:.2f} sigma")

    results['null_test'] = {
        'n_realizations': n_null,
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'significance_vs_null': float(sig_vs_null),
        'null_amplitudes': null_amplitudes.tolist(),
    }

    # =========================================================================
    # Final Results
    # =========================================================================
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nDESI DR1 LRG x Planck SMICA Pairwise kSZ")
    print(f"Total galaxies: {len(ra):,}")
    print(f"Galaxies used per bin: {SUBSAMPLE_PER_BIN:,}")
    print(f"Total pairs analyzed: {sum(all_n_pairs):,}")
    print()

    print("Per-bin results:")
    for i, zbin in enumerate(results['z_bins']):
        print(f"  {zbin['z_lo']:.2f} < z < {zbin['z_hi']:.2f}: "
              f"A = {zbin.get('amplitude', 0):.4f} ± {zbin.get('amplitude_err', 0):.4f}, "
              f"S/N = {zbin.get('snr', 0):.2f}")

    print()
    print(f"JOINT AMPLITUDE: {A_joint:.4f} ± {A_joint_err:.4f}")
    print(f"JOINT DETECTION: {joint_snr:.2f} sigma")
    print()

    if joint_snr >= 3.0:
        verdict = "DETECTION"
    elif joint_snr >= 2.0:
        verdict = "TENTATIVE"
    else:
        verdict = "NO DETECTION"

    print(f"VERDICT: {verdict}")

    # Save results
    output_file = Path("data/ksz/output/ksz_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    summary_file = Path("data/ksz/output/RESULTS_SUMMARY.txt")
    with open(summary_file, 'w') as f:
        f.write("DESI DR1 x Planck SMICA Pairwise kSZ Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis date: {datetime.now().isoformat()}\n")
        f.write(f"Total LRGs: {len(ra):,}\n")
        f.write(f"Galaxies used per bin: {SUBSAMPLE_PER_BIN:,}\n")
        f.write(f"Total pairs: {sum(all_n_pairs):,}\n\n")
        f.write("Per-bin results:\n")
        for i, zbin in enumerate(results['z_bins']):
            f.write(f"  {zbin['z_lo']:.3f} < z < {zbin['z_hi']:.3f}: "
                    f"A = {zbin.get('amplitude', 0):.4f} ± {zbin.get('amplitude_err', 0):.4f}, "
                    f"S/N = {zbin.get('snr', 0):.2f}\n")
        f.write(f"\nJOINT AMPLITUDE: {A_joint:.4f} ± {A_joint_err:.4f}\n")
        f.write(f"JOINT DETECTION: {joint_snr:.2f} sigma\n")
        f.write(f"\nVERDICT: {verdict}\n")

    print(f"Summary saved to: {summary_file}")
    print()
    print(f"Completed: {datetime.now().isoformat()}")
