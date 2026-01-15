#!/usr/bin/env python3
"""
ITERATION 2: Mean-field subtraction with random catalog
========================================================
Apply ℓ > 300 filter + subtract mean-field using DESI randoms.

Mean-field: p̂_corrected(r) = p̂_data(r) − p̂_random(r)
"""

import numpy as np
import healpy as hp
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.cosmology import Planck18
from multiprocessing import Pool, cpu_count

# Configuration
CONFIG = {
    'iteration': 2,
    'n_gal_test': 30000,  # Increase sample size for better statistics
    'n_rand_ratio': 2,    # Use 2x as many randoms as galaxies
    'r_bins': np.linspace(20, 150, 11),
    'apply_mask': True,
    'remove_monopole_dipole': True,
    'ell_min': 300,  # HIGH-PASS: suppress large-scale CMB
    'ell_max': 3000,
    'random_seed': 42,
    'n_null': 30,
    'n_workers': max(1, cpu_count() - 1),
}

print("=" * 70)
print("ITERATION 2: MEAN-FIELD SUBTRACTION WITH RANDOM CATALOG")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print(f"Using {CONFIG['n_workers']} parallel workers")
print()

# =============================================================================
# 1. Load and Process CMB Map with MASK + HARMONIC FILTER
# =============================================================================
print("[1/9] Loading Planck SMICA map WITH mask...")

planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")
with fits.open(planck_file) as hdu:
    cmb_map_raw = hp.read_map(hdu, field=0, verbose=False) * 1e6  # μK
    tmask = hp.read_map(hdu, field=3, verbose=False)

nside = hp.npix2nside(len(cmb_map_raw))
print(f"  NSIDE: {nside}")
print(f"  Map raw std: {cmb_map_raw.std():.1f} μK")

# Apply mask
cmb_map = cmb_map_raw.copy()
cmb_map[tmask == 0] = hp.UNSEEN
valid_pix = cmb_map != hp.UNSEEN
print(f"  TMASK: {valid_pix.sum()/len(cmb_map)*100:.1f}% unmasked")

# Remove monopole/dipole
cmb_map_cleaned = hp.remove_dipole(cmb_map, gal_cut=0, fitval=True, verbose=False)
cmb_map = np.array(cmb_map_cleaned[0])
cmb_map[~valid_pix] = hp.UNSEEN
print(f"  After mono/dipole removal: mean={cmb_map[valid_pix].mean():.4f} μK")
print()

# =============================================================================
# 2. Apply Harmonic Filter (HIGH-PASS)
# =============================================================================
print(f"[2/9] Applying harmonic filter ({CONFIG['ell_min']} < ℓ < {CONFIG['ell_max']})...")

# For masked map, fill with constrained realization approach
# Use mean of valid pixels instead of zero (reduces ringing)
cmb_map_filled = cmb_map.copy()
mean_valid = cmb_map[valid_pix].mean()
cmb_map_filled[~valid_pix] = mean_valid  # Fill with mean instead of 0

# Convert to alm
lmax = min(3 * nside - 1, CONFIG['ell_max'] + 100)
alm = hp.map2alm(cmb_map_filled, lmax=lmax)

# Get ℓ for each alm
l_alm = hp.Alm.getlm(lmax)[0]

# Apply filter with smooth edges (avoid ringing)
ell_low = CONFIG['ell_min']
ell_high = CONFIG['ell_max']
delta_ell = 50  # Smooth transition width

# Smooth filter function
filter_func = np.ones(len(l_alm))
# Low-ell cutoff (smooth)
low_mask = l_alm < ell_low + delta_ell
filter_func[low_mask] = 0.5 * (1 + np.tanh((l_alm[low_mask] - ell_low) / (delta_ell/3)))
filter_func[l_alm < ell_low - delta_ell] = 0.0
# High-ell cutoff (smooth)
high_mask = l_alm > ell_high - delta_ell
filter_func[high_mask] = 0.5 * (1 - np.tanh((l_alm[high_mask] - ell_high) / (delta_ell/3)))
filter_func[l_alm > ell_high + delta_ell] = 0.0

alm_filtered = alm * filter_func

# Convert back to map
cmb_map_filtered = hp.alm2map(alm_filtered, nside, verbose=False)

# Re-apply mask
cmb_map_filtered[~valid_pix] = hp.UNSEEN

print(f"  Filtered map std: {cmb_map_filtered[valid_pix].std():.1f} μK")
print(f"  Suppression factor: {cmb_map[valid_pix].std() / cmb_map_filtered[valid_pix].std():.2f}x")
print()

# Use filtered map
cmb_map = cmb_map_filtered

# =============================================================================
# 3. Load DESI LRG Catalog
# =============================================================================
print("[3/9] Loading DESI LRG catalog...")

ngc_file = Path("data/ksz/catalogs/LRG_NGC_clustering.dat.fits")
sgc_file = Path("data/ksz/catalogs/LRG_SGC_clustering.dat.fits")
with fits.open(ngc_file) as hdu:
    ngc = hdu[1].data
with fits.open(sgc_file) as hdu:
    sgc = hdu[1].data

ra_full = np.concatenate([ngc['RA'], sgc['RA']])
dec_full = np.concatenate([ngc['DEC'], sgc['DEC']])
z_full = np.concatenate([ngc['Z'], sgc['Z']])
w_full = np.concatenate([ngc['WEIGHT_SYS'] * ngc['WEIGHT_COMP'] * ngc['WEIGHT_ZFAIL'],
                         sgc['WEIGHT_SYS'] * sgc['WEIGHT_COMP'] * sgc['WEIGHT_ZFAIL']])

# Random subsample
rng = np.random.default_rng(CONFIG['random_seed'])
n_test = min(CONFIG['n_gal_test'], len(ra_full))
idx = rng.choice(len(ra_full), size=n_test, replace=False)
ra, dec, z, weights = ra_full[idx], dec_full[idx], z_full[idx], w_full[idx]

print(f"  Total LRGs: {len(ra_full):,}")
print(f"  Test sample: {n_test:,}")
print()

# =============================================================================
# 4. Load DESI Random Catalog
# =============================================================================
print("[4/9] Loading DESI LRG random catalog...")

random_file = Path("data/ksz/randoms/LRG_0_full.ran.fits")
with fits.open(random_file) as hdu:
    rand_data = hdu[1].data

ra_rand_full = rand_data['RA']
dec_rand_full = rand_data['DEC']
z_rand_full = rand_data['Z']

# Compute weights for randoms (WEIGHT_SYS * WEIGHT_COMP * WEIGHT_ZFAIL if available)
# For randoms, weights are typically simpler
if 'WEIGHT' in rand_data.columns.names:
    w_rand_full = rand_data['WEIGHT']
else:
    # Use uniform weights for randoms
    w_rand_full = np.ones(len(ra_rand_full))

print(f"  Total randoms: {len(ra_rand_full):,}")

# Subsample randoms (2x galaxies)
n_rand = min(CONFIG['n_rand_ratio'] * n_test, len(ra_rand_full))
idx_rand = rng.choice(len(ra_rand_full), size=n_rand, replace=False)
ra_rand, dec_rand, z_rand = ra_rand_full[idx_rand], dec_rand_full[idx_rand], z_rand_full[idx_rand]
w_rand = w_rand_full[idx_rand]

print(f"  Random sample: {n_rand:,}")
print()

# =============================================================================
# 5. Extract Temperatures for Galaxies and Randoms
# =============================================================================
print("[5/9] Extracting temperatures...")

def extract_temperatures(ra, dec, cmb_map, nside):
    """Extract CMB temperatures at positions."""
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(nside, theta, phi)
    return cmb_map[pix]

# Galaxies
temperatures = extract_temperatures(ra, dec, cmb_map, nside)
in_mask = temperatures != hp.UNSEEN
n_masked = (~in_mask).sum()
print(f"  Galaxies masked: {n_masked} ({n_masked/len(temperatures)*100:.1f}%)")

ra = ra[in_mask]
dec = dec[in_mask]
z = z[in_mask]
weights = weights[in_mask]
temperatures = temperatures[in_mask]

print(f"  Valid galaxies: {len(ra):,}")
print(f"  Galaxy T mean: {temperatures.mean():.2f} μK")
print(f"  Galaxy T std: {temperatures.std():.2f} μK")

# Randoms
temperatures_rand = extract_temperatures(ra_rand, dec_rand, cmb_map, nside)
in_mask_rand = temperatures_rand != hp.UNSEEN
n_masked_rand = (~in_mask_rand).sum()
print(f"  Randoms masked: {n_masked_rand} ({n_masked_rand/len(temperatures_rand)*100:.1f}%)")

ra_rand = ra_rand[in_mask_rand]
dec_rand = dec_rand[in_mask_rand]
z_rand = z_rand[in_mask_rand]
w_rand = w_rand[in_mask_rand]
temperatures_rand = temperatures_rand[in_mask_rand]

print(f"  Valid randoms: {len(ra_rand):,}")
print(f"  Random T mean: {temperatures_rand.mean():.2f} μK")
print(f"  Random T std: {temperatures_rand.std():.2f} μK")
print()

# =============================================================================
# 6. Compute Comoving Positions
# =============================================================================
print("[6/9] Computing comoving positions...")

cosmo = Planck18
h = cosmo.H0.value / 100

def compute_positions(ra, dec, z, cosmo, h):
    """Compute 3D comoving positions."""
    chi = cosmo.comoving_distance(z).value * h
    ra_rad, dec_rad = np.radians(ra), np.radians(dec)
    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    z_pos = chi * np.sin(dec_rad)
    positions = np.column_stack([x, y, z_pos])
    r_mag = np.sqrt(np.sum(positions**2, axis=1))
    r_hat = positions / r_mag[:, np.newaxis]
    return positions, r_hat, chi

positions, r_hat, chi = compute_positions(ra, dec, z, cosmo, h)
positions_rand, r_hat_rand, chi_rand = compute_positions(ra_rand, dec_rand, z_rand, cosmo, h)

print(f"  Galaxy comoving range: {chi.min():.1f} - {chi.max():.1f} Mpc/h")
print(f"  Random comoving range: {chi_rand.min():.1f} - {chi_rand.max():.1f} Mpc/h")
print()

# =============================================================================
# 7. Pairwise Estimator (Parallelized)
# =============================================================================
print("[7/9] Computing pairwise estimator...")

r_bin_edges = CONFIG['r_bins']
r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
n_bins = len(r_bin_edges) - 1

def compute_pairwise_chunk(args):
    """Compute pairwise for a chunk of galaxies."""
    i_start, i_end, pos, T, w, r_hat, r_bin_edges, tree = args
    n_bins = len(r_bin_edges) - 1
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    for i in range(i_start, i_end):
        neighbors = tree.query_ball_point(pos[i], r_bin_edges[-1])
        for j in neighbors:
            if j <= i:
                continue
            dr = pos[j] - pos[i]
            r = np.sqrt(np.sum(dr**2))
            if r < r_bin_edges[0]:
                continue
            bin_idx = np.searchsorted(r_bin_edges, r) - 1
            if bin_idx < 0 or bin_idx >= n_bins:
                continue
            r_hat_ij = dr / r
            c_ij = 0.5 * np.dot(r_hat_ij, r_hat[i] - r_hat[j])
            ww = w[i] * w[j]
            dT = T[i] - T[j]
            num[bin_idx] += ww * dT * c_ij
            den[bin_idx] += ww * c_ij**2
            counts[bin_idx] += 1

    return num, den, counts

def compute_pairwise(pos, T, w, r_hat, r_bin_edges, n_workers=1, label=""):
    """Compute pairwise estimator with optional parallelization."""
    tree = cKDTree(pos)
    n = len(pos)
    n_bins = len(r_bin_edges) - 1

    if n_workers > 1 and n > 5000:
        # Parallel execution
        chunk_size = n // n_workers
        chunks = []
        for k in range(n_workers):
            i_start = k * chunk_size
            i_end = n if k == n_workers - 1 else (k + 1) * chunk_size
            chunks.append((i_start, i_end, pos, T, w, r_hat, r_bin_edges, tree))

        with Pool(n_workers) as pool:
            results = pool.map(compute_pairwise_chunk, chunks)

        num = np.sum([r[0] for r in results], axis=0)
        den = np.sum([r[1] for r in results], axis=0)
        counts = np.sum([r[2] for r in results], axis=0)
    else:
        # Serial execution
        num = np.zeros(n_bins)
        den = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=np.int64)

        for i in range(n):
            neighbors = tree.query_ball_point(pos[i], r_bin_edges[-1])
            for j in neighbors:
                if j <= i:
                    continue
                dr = pos[j] - pos[i]
                r = np.sqrt(np.sum(dr**2))
                if r < r_bin_edges[0]:
                    continue
                bin_idx = np.searchsorted(r_bin_edges, r) - 1
                if bin_idx < 0 or bin_idx >= n_bins:
                    continue
                r_hat_ij = dr / r
                c_ij = 0.5 * np.dot(r_hat_ij, r_hat[i] - r_hat[j])
                ww = w[i] * w[j]
                dT = T[i] - T[j]
                num[bin_idx] += ww * dT * c_ij
                den[bin_idx] += ww * c_ij**2
                counts[bin_idx] += 1

            if label and i > 0 and i % 5000 == 0:
                print(f"    {label}: {i:,}/{n:,} done...")

    with np.errstate(divide='ignore', invalid='ignore'):
        p_ksz = np.where(den > 0, num / den, 0)
    return p_ksz, counts, num, den

print("  Computing p_data (galaxies)...")
p_data, counts_data, num_data, den_data = compute_pairwise(
    positions, temperatures, weights, r_hat, r_bin_edges,
    n_workers=CONFIG['n_workers'], label="Galaxies"
)
print(f"    Total pairs: {counts_data.sum():,}")

print("  Computing p_random (mean-field)...")
p_rand, counts_rand, num_rand, den_rand = compute_pairwise(
    positions_rand, temperatures_rand, w_rand, r_hat_rand, r_bin_edges,
    n_workers=CONFIG['n_workers'], label="Randoms"
)
print(f"    Total random pairs: {counts_rand.sum():,}")

# Mean-field subtraction
p_corrected = p_data - p_rand

print()
print("  Raw p_data vs p_rand vs p_corrected:")
for i in range(n_bins):
    print(f"    r={r_centers[i]:.0f}: p_data={p_data[i]:.2f}, p_rand={p_rand[i]:.2f}, p_corr={p_corrected[i]:.2f} μK")
print()

# =============================================================================
# 8. Null Tests + Results
# =============================================================================
print(f"[8/9] Running {CONFIG['n_null']} null tests...")

null_amplitudes = []
template = np.exp(-r_centers / 60.0)
valid = counts_data > 100

def run_null_test(seed):
    """Run single null test with shuffled temperatures."""
    rng_null = np.random.default_rng(seed)
    T_shuffled = temperatures.copy()
    rng_null.shuffle(T_shuffled)
    p_null, _, _, _ = compute_pairwise(positions, T_shuffled, weights, r_hat, r_bin_edges, n_workers=1)

    # Fit amplitude
    if valid.sum() >= 3:
        w = counts_data[valid].astype(float)
        A = np.sum(w * p_null[valid] * template[valid]) / np.sum(w * template[valid]**2)
        return A
    return np.nan

# Run null tests in parallel
with Pool(CONFIG['n_workers']) as pool:
    null_amplitudes = pool.map(run_null_test, range(CONFIG['n_null']))

null_amplitudes = np.array([a for a in null_amplitudes if not np.isnan(a)])

# Corrected amplitude
if valid.sum() >= 3:
    w = counts_data[valid].astype(float)
    A_corrected = np.sum(w * p_corrected[valid] * template[valid]) / np.sum(w * template[valid]**2)
    A_raw = np.sum(w * p_data[valid] * template[valid]) / np.sum(w * template[valid]**2)
else:
    A_corrected = np.nan
    A_raw = np.nan

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"{'r (Mpc/h)':>12} | {'p_data':>10} | {'p_rand':>10} | {'p_corr':>10} | {'N_pairs':>12}")
print("-" * 62)
for i in range(n_bins):
    print(f"{r_centers[i]:12.1f} | {p_data[i]:10.2f} | {p_rand[i]:10.2f} | {p_corrected[i]:10.2f} | {counts_data[i]:12,}")

print()
print(f"Median |p_data|: {np.median(np.abs(p_data)):.2f} μK")
print(f"Median |p_rand|: {np.median(np.abs(p_rand)):.2f} μK")
print(f"Median |p_corrected|: {np.median(np.abs(p_corrected)):.2f} μK")
print()
print(f"Raw amplitude (A_data): {A_raw:.2f} μK")
print(f"Corrected amplitude (A_corr): {A_corrected:.2f} μK")
print(f"Null mean: {null_amplitudes.mean():.2f} μK")
print(f"Null std: {null_amplitudes.std():.2f} μK")
print(f"Significance (corrected): {(A_corrected - null_amplitudes.mean()) / null_amplitudes.std():.2f} sigma")
print()

# =============================================================================
# 9. Run Ledger Entry
# =============================================================================
ledger_entry = {
    'iteration': 2,
    'timestamp': datetime.now().isoformat(),
    'n_gal': len(ra),
    'n_rand': len(ra_rand),
    'n_pairs_data': int(counts_data.sum()),
    'n_pairs_rand': int(counts_rand.sum()),
    'map_product': 'Planck SMICA R3.00',
    'masking': 'TMASK',
    'monopole_dipole': 'removed',
    'ell_filter': f"{CONFIG['ell_min']}-{CONFIG['ell_max']} (smooth edges)",
    'temp_extraction': 'single pixel',
    'mean_field_subtraction': 'DESI LRG randoms',
    'median_abs_p_data': float(np.median(np.abs(p_data))),
    'median_abs_p_rand': float(np.median(np.abs(p_rand))),
    'median_abs_p_corrected': float(np.median(np.abs(p_corrected))),
    'amplitude_raw': float(A_raw),
    'amplitude_corrected': float(A_corrected),
    'null_mean': float(null_amplitudes.mean()),
    'null_std': float(null_amplitudes.std()),
    'significance': float((A_corrected - null_amplitudes.mean()) / null_amplitudes.std()),
}

print("=" * 70)
print("RUN LEDGER ENTRY")
print("=" * 70)
for k, v in ledger_entry.items():
    print(f"  {k}: {v}")

# Save results
output_dir = Path("data/ksz/output/iteration2")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'config': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in CONFIG.items()},
    'ledger': ledger_entry,
    'r_centers': r_centers.tolist(),
    'p_data': p_data.tolist(),
    'p_rand': p_rand.tolist(),
    'p_corrected': p_corrected.tolist(),
    'pair_counts_data': counts_data.tolist(),
    'pair_counts_rand': counts_rand.tolist(),
    'null_amplitudes': null_amplitudes.tolist(),
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"Results saved to: {output_dir}/results.json")
print(f"Completed: {datetime.now().isoformat()}")
