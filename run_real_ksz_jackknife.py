#!/usr/bin/env python3
"""
DESI DR1 x Planck PR3 Pairwise kSZ Analysis
============================================
With proper jackknife covariance estimation.
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


def compute_pairwise_parallel(pos, T, w, r_bin_edges, n_workers=None, verbose=True):
    """
    Compute pairwise kSZ momentum with parallel processing.
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    n = len(pos)

    # Unit vectors
    r_mag = np.sqrt(np.sum(pos**2, axis=1))
    r_hat = pos / r_mag[:, np.newaxis]

    # Split indices into chunks
    chunk_size = max(100, n // (n_workers * 10))
    chunks = [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]

    if verbose:
        print(f"    Using {n_workers} workers, {len(chunks)} chunks, {n} galaxies")

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


def jackknife_covariance(pos, T, w, ra, dec, r_bin_edges, n_jk=50, n_workers=None, subsample=None):
    """
    Compute pairwise momentum with jackknife covariance.

    Split sky into n_jk regions using HEALPix, compute p(r) excluding each region.
    """
    import healpy as hp

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    n_gal = len(pos)

    # Subsample if requested
    if subsample is not None and subsample < n_gal:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_gal, size=subsample, replace=False)
        pos = pos[idx]
        T = T[idx]
        w = w[idx]
        ra = ra[idx]
        dec = dec[idx]
        n_gal = len(pos)
        print(f"    Subsampled to {n_gal:,} galaxies for jackknife")

    # Assign galaxies to HEALPix regions
    nside_jk = 4  # 192 pixels, will use n_jk of them
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pixel_ids = hp.ang2pix(nside_jk, theta, phi)

    # Get unique pixel IDs and map to jackknife regions
    unique_pixels = np.unique(pixel_ids)
    n_actual_jk = min(n_jk, len(unique_pixels))

    # Map pixels to jackknife regions (round-robin assignment)
    pixel_to_jk = {pix: i % n_actual_jk for i, pix in enumerate(unique_pixels)}
    jk_ids = np.array([pixel_to_jk[pix] for pix in pixel_ids])

    print(f"    Jackknife regions: {n_actual_jk}")

    # Compute full sample p(r)
    p_full, counts_full, _, _ = compute_pairwise_parallel(pos, T, w, r_bin_edges, n_workers, verbose=False)

    # Compute jackknife samples
    n_bins = len(r_bin_edges) - 1
    p_jk = np.zeros((n_actual_jk, n_bins))

    for i_jk in range(n_actual_jk):
        # Exclude region i_jk
        mask = jk_ids != i_jk
        pos_jk = pos[mask]
        T_jk = T[mask]
        w_jk = w[mask]

        p_jk[i_jk], _, _, _ = compute_pairwise_parallel(pos_jk, T_jk, w_jk, r_bin_edges, n_workers, verbose=False)

        if (i_jk + 1) % 10 == 0:
            print(f"    Jackknife {i_jk + 1}/{n_actual_jk} complete")

    # Compute covariance from jackknife samples
    # Cov = (n_jk - 1) / n_jk * Σ (p_jk - p_mean)^2
    p_jk_mean = np.mean(p_jk, axis=0)
    cov = np.zeros((n_bins, n_bins))
    for i_jk in range(n_actual_jk):
        dp = p_jk[i_jk] - p_jk_mean
        cov += np.outer(dp, dp)
    cov *= (n_actual_jk - 1) / n_actual_jk

    # Error bars from diagonal
    p_err = np.sqrt(np.diag(cov))

    return p_full, p_err, cov, counts_full


def theory_template(r, A=1.0, r0=50.0):
    """Simple theory template: p(r) = A * exp(-r/r0)"""
    return A * np.exp(-r / r0)


if __name__ == '__main__':
    print("=" * 70)
    print("DESI DR1 x Planck SMICA Pairwise kSZ Analysis")
    print("WITH JACKKNIFE COVARIANCE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"CPUs available: {cpu_count()}")
    print()

    # =========================================================================
    # 1. Load DESI DR1 LRG Catalogs
    # =========================================================================
    print("[1/7] Loading DESI DR1 LRG catalogs...")

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
    print()

    # =========================================================================
    # 2. Load Planck SMICA CMB Map
    # =========================================================================
    print("[2/7] Loading Planck SMICA CMB map...")

    try:
        import healpy as hp
    except ImportError:
        print("ERROR: healpy required")
        sys.exit(1)

    planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")
    cmb_map = hp.read_map(str(planck_file), field=0, verbose=False)
    nside = hp.npix2nside(len(cmb_map))

    cmb_map = cmb_map * 1e6  # Convert to μK

    print(f"  NSIDE: {nside}, N pixels: {len(cmb_map):,}")
    print()

    # =========================================================================
    # 3. Compute Comoving Positions and Extract Temperatures
    # =========================================================================
    print("[3/7] Computing positions and extracting temperatures...")

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

    # Extract temperatures
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)
    temperatures = cmb_map[pix]

    # Mask bad pixels
    good = temperatures != hp.UNSEEN
    ra = ra[good]
    dec = dec[good]
    z = z[good]
    weights = weights[good]
    positions = positions[good]
    temperatures = temperatures[good]

    # Subtract mean to reduce large-scale bias
    temperatures = temperatures - temperatures.mean()

    print(f"  Valid galaxies: {len(ra):,}")
    print(f"  Temperature std: {temperatures.std():.2f} μK")
    print()

    # =========================================================================
    # 4. Define Bins
    # =========================================================================
    print("[4/7] Setting up bins...")

    z_sorted = np.sort(z)
    n_gal = len(z)
    z_bin_edges = [
        z.min(),
        z_sorted[n_gal // 3],
        z_sorted[2 * n_gal // 3],
        z.max() + 0.001
    ]

    r_min, r_max = 20.0, 150.0
    n_r_bins = 10  # Fewer bins for better S/N per bin
    r_bin_edges = np.linspace(r_min, r_max, n_r_bins + 1)
    r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

    print(f"  Z bins: {[f'{e:.3f}' for e in z_bin_edges]}")
    print(f"  R bins: {r_min} - {r_max} Mpc/h ({n_r_bins} bins)")
    print()

    # =========================================================================
    # 5. Compute Pairwise Momentum with Jackknife Covariance
    # =========================================================================
    print("[5/7] Computing pairwise momentum with jackknife covariance...")

    results = {
        'z_bins': [],
        'r_centers': r_centers.tolist(),
        'r_bin_edges': r_bin_edges.tolist(),
        'analysis_date': datetime.now().isoformat(),
    }

    n_workers = max(1, cpu_count() - 1)
    SUBSAMPLE_PER_BIN = 30000  # For tractable jackknife
    N_JK = 50  # Number of jackknife regions

    all_p_ksz = []
    all_p_err = []
    all_cov = []

    for i_z in range(len(z_bin_edges) - 1):
        z_lo, z_hi = z_bin_edges[i_z], z_bin_edges[i_z + 1]
        z_mask = (z >= z_lo) & (z < z_hi)

        pos_bin = positions[z_mask]
        T_bin = temperatures[z_mask]
        w_bin = weights[z_mask]
        ra_bin = ra[z_mask]
        dec_bin = dec[z_mask]
        z_bin_arr = z[z_mask]

        n_gal_bin = len(pos_bin)
        z_mean = z_bin_arr.mean()

        print(f"\n  Z bin {i_z + 1}: {z_lo:.3f} < z < {z_hi:.3f}")
        print(f"    N galaxies: {n_gal_bin:,}, mean z: {z_mean:.3f}")

        t0 = time.time()
        p_ksz, p_err, cov, pair_counts = jackknife_covariance(
            pos_bin, T_bin, w_bin, ra_bin, dec_bin, r_bin_edges,
            n_jk=N_JK,
            n_workers=n_workers,
            subsample=SUBSAMPLE_PER_BIN
        )
        dt = time.time() - t0

        print(f"    Jackknife time: {dt:.1f}s")
        print(f"    p(r) range: {p_ksz.min():.2f} to {p_ksz.max():.2f} μK")
        print(f"    Error range: {p_err.min():.2f} to {p_err.max():.2f} μK")

        all_p_ksz.append(p_ksz)
        all_p_err.append(p_err)
        all_cov.append(cov)

        results['z_bins'].append({
            'z_lo': float(z_lo),
            'z_hi': float(z_hi),
            'z_mean': float(z_mean),
            'n_galaxies': int(n_gal_bin),
            'n_pairs': int(pair_counts.sum()),
            'p_ksz': p_ksz.tolist(),
            'p_err': p_err.tolist(),
            'pair_counts': pair_counts.tolist(),
        })

    print()

    # =========================================================================
    # 6. Amplitude Fitting
    # =========================================================================
    print("[6/7] Fitting amplitude to theory template...")

    amplitudes = []
    amplitude_errors = []
    snrs = []

    for i_z, zbin in enumerate(results['z_bins']):
        p_data = np.array(zbin['p_ksz'])
        p_err = np.array(zbin['p_err'])

        # Theory template
        p_template = theory_template(r_centers, A=1.0, r0=60.0)

        # Weighted least squares
        valid = (p_err > 0) & np.isfinite(p_data)
        if valid.sum() < 3:
            amplitudes.append(0.0)
            amplitude_errors.append(999.0)
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
        print(f"  Z bin {i_z + 1} ({z_lo:.2f}-{z_hi:.2f}): A = {A_fit:.2f} ± {A_err:.2f} μK, S/N = {snrs[-1]:.2f}")

    # Joint fit
    all_p = np.concatenate([np.array(zb['p_ksz']) for zb in results['z_bins']])
    all_e = np.concatenate([np.array(zb['p_err']) for zb in results['z_bins']])
    all_template = np.tile(theory_template(r_centers, A=1.0, r0=60.0), len(results['z_bins']))

    valid = (all_e > 0) & np.isfinite(all_p)
    w_joint = 1.0 / all_e[valid]**2
    A_joint = np.sum(w_joint * all_p[valid] * all_template[valid]) / np.sum(w_joint * all_template[valid]**2)
    A_joint_err = 1.0 / np.sqrt(np.sum(w_joint * all_template[valid]**2))
    joint_snr = np.abs(A_joint) / A_joint_err

    print(f"\n  JOINT FIT: A = {A_joint:.2f} ± {A_joint_err:.2f} μK")
    print(f"  JOINT S/N: {joint_snr:.2f} sigma")

    results['joint_amplitude'] = float(A_joint)
    results['joint_amplitude_err'] = float(A_joint_err)
    results['joint_snr'] = float(joint_snr)

    # =========================================================================
    # 7. Final Results
    # =========================================================================
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"\nDESI DR1 LRG x Planck SMICA Pairwise kSZ")
    print(f"Total galaxies: {len(ra):,}")
    print()

    print("Per-bin results:")
    for i, zbin in enumerate(results['z_bins']):
        print(f"  {zbin['z_lo']:.2f} < z < {zbin['z_hi']:.2f}: "
              f"A = {zbin.get('amplitude', 0):.2f} ± {zbin.get('amplitude_err', 0):.2f} μK, "
              f"S/N = {zbin.get('snr', 0):.2f}")

    print()
    print(f"JOINT AMPLITUDE: {A_joint:.2f} ± {A_joint_err:.2f} μK")
    print(f"JOINT DETECTION: {joint_snr:.2f} sigma")
    print()

    if joint_snr >= 3.0:
        verdict = "DETECTION"
    elif joint_snr >= 2.0:
        verdict = "TENTATIVE"
    else:
        verdict = "NO DETECTION (noise-dominated)"

    print(f"VERDICT: {verdict}")
    print()
    print("NOTE: Expected kSZ amplitude is ~0.1-1 μK. Large amplitudes indicate")
    print("      systematic effects or noise fluctuations.")

    # Save results
    output_file = Path("data/ksz/output/ksz_results_jackknife.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    summary_file = Path("data/ksz/output/RESULTS_JACKKNIFE.txt")
    with open(summary_file, 'w') as f:
        f.write("DESI DR1 x Planck SMICA Pairwise kSZ Results (Jackknife)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis date: {datetime.now().isoformat()}\n")
        f.write(f"Total LRGs: {len(ra):,}\n")
        f.write(f"Jackknife regions: {N_JK}\n\n")
        f.write("Per-bin results:\n")
        for i, zbin in enumerate(results['z_bins']):
            f.write(f"  {zbin['z_lo']:.3f} < z < {zbin['z_hi']:.3f}: "
                    f"A = {zbin.get('amplitude', 0):.2f} ± {zbin.get('amplitude_err', 0):.2f} μK, "
                    f"S/N = {zbin.get('snr', 0):.2f}\n")
        f.write(f"\nJOINT AMPLITUDE: {A_joint:.2f} ± {A_joint_err:.2f} μK\n")
        f.write(f"JOINT DETECTION: {joint_snr:.2f} sigma\n")
        f.write(f"\nVERDICT: {verdict}\n")

    print(f"Summary saved to: {summary_file}")
    print()
    print(f"Completed: {datetime.now().isoformat()}")
