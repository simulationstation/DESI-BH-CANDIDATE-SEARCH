#!/usr/bin/env python3
"""
ITERATION 1: High-pass harmonic filter
=======================================
Apply ℓ > 300 filter to suppress primary CMB.
Primary CMB dominates at large scales (low ℓ).
"""

import numpy as np
import healpy as hp
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.cosmology import Planck18

# Configuration
CONFIG = {
    'iteration': 1,
    'n_gal_test': 20000,  # Increase sample size
    'r_bins': np.linspace(20, 150, 11),
    'apply_mask': True,
    'remove_monopole_dipole': True,
    'ell_min': 300,  # HIGH-PASS: suppress large-scale CMB
    'ell_max': 3000,  # Avoid beam/noise dominated scales
    'random_seed': 42,
    'n_null': 20,
}

print("=" * 70)
print("ITERATION 1: HIGH-PASS HARMONIC FILTER (ℓ > 300)")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print()

# =============================================================================
# 1. Load and Process CMB Map with MASK + HARMONIC FILTER
# =============================================================================
print("[1/7] Loading Planck SMICA map WITH mask...")

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
print(f"[2/7] Applying harmonic filter ({CONFIG['ell_min']} < ℓ < {CONFIG['ell_max']})...")

# For masked map, we need to fill masked pixels first
cmb_map_filled = cmb_map.copy()
cmb_map_filled[~valid_pix] = 0.0  # Fill masked with zero

# Convert to alm
lmax = min(3 * nside - 1, CONFIG['ell_max'] + 100)
alm = hp.map2alm(cmb_map_filled, lmax=lmax)

# Get ℓ for each alm
l_alm = hp.Alm.getlm(lmax)[0]

# Apply filter: zero out ℓ < ell_min and ℓ > ell_max
filter_mask = (l_alm < CONFIG['ell_min']) | (l_alm > CONFIG['ell_max'])
alm_filtered = alm.copy()
alm_filtered[filter_mask] = 0.0

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
print("[3/7] Loading DESI LRG catalog...")

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
# 4. Extract Temperatures
# =============================================================================
print("[4/7] Extracting temperatures...")

theta = np.radians(90.0 - dec)
phi = np.radians(ra)
pix = hp.ang2pix(nside, theta, phi)
temperatures = cmb_map[pix]

# Mask check
in_mask = temperatures != hp.UNSEEN
n_masked = (~in_mask).sum()
print(f"  Masked galaxies: {n_masked} ({n_masked/len(temperatures)*100:.1f}%)")

ra = ra[in_mask]
dec = dec[in_mask]
z = z[in_mask]
weights = weights[in_mask]
temperatures = temperatures[in_mask]

print(f"  Valid galaxies: {len(ra):,}")
print(f"  T mean: {temperatures.mean():.2f} μK")
print(f"  T std: {temperatures.std():.2f} μK")
print()

# =============================================================================
# 5. Compute Comoving Positions
# =============================================================================
print("[5/7] Computing comoving positions...")

cosmo = Planck18
h = cosmo.H0.value / 100
chi = cosmo.comoving_distance(z).value * h

ra_rad, dec_rad = np.radians(ra), np.radians(dec)
x = chi * np.cos(dec_rad) * np.cos(ra_rad)
y = chi * np.cos(dec_rad) * np.sin(ra_rad)
z_pos = chi * np.sin(dec_rad)
positions = np.column_stack([x, y, z_pos])
r_mag = np.sqrt(np.sum(positions**2, axis=1))
r_hat = positions / r_mag[:, np.newaxis]

print(f"  Comoving range: {chi.min():.1f} - {chi.max():.1f} Mpc/h")
print()

# =============================================================================
# 6. Pairwise Estimator
# =============================================================================
print("[6/7] Computing pairwise estimator...")

r_bin_edges = CONFIG['r_bins']
r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
n_bins = len(r_bin_edges) - 1

def compute_pairwise(pos, T, w, r_hat, r_bin_edges):
    tree = cKDTree(pos)
    n_bins = len(r_bin_edges) - 1
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    n = len(pos)

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

        if i > 0 and i % 5000 == 0:
            print(f"    {i:,}/{n:,} galaxies...")

    with np.errstate(divide='ignore', invalid='ignore'):
        p_ksz = np.where(den > 0, num / den, 0)
    return p_ksz, counts

p_real, pair_counts = compute_pairwise(positions, temperatures, weights, r_hat, r_bin_edges)

print(f"  Total pairs: {pair_counts.sum():,}")
print()

# =============================================================================
# 7. Null Tests + Results
# =============================================================================
print(f"[7/7] Running {CONFIG['n_null']} null tests...")

null_amplitudes = []
template = np.exp(-r_centers / 60.0)
valid = pair_counts > 100

for i in range(CONFIG['n_null']):
    T_shuffled = temperatures.copy()
    rng.shuffle(T_shuffled)
    p_null, _ = compute_pairwise(positions, T_shuffled, weights, r_hat, r_bin_edges)
    if valid.sum() >= 3:
        w = pair_counts[valid].astype(float)
        A = np.sum(w * p_null[valid] * template[valid]) / np.sum(w * template[valid]**2)
        null_amplitudes.append(A)
    if (i+1) % 5 == 0:
        print(f"    {i+1}/{CONFIG['n_null']} null tests done")

null_amplitudes = np.array(null_amplitudes)

# Real amplitude
if valid.sum() >= 3:
    w = pair_counts[valid].astype(float)
    A_real = np.sum(w * p_real[valid] * template[valid]) / np.sum(w * template[valid]**2)
else:
    A_real = np.nan

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"{'r (Mpc/h)':>12} | {'p(r) (μK)':>12} | {'N_pairs':>12}")
print("-" * 42)
for i in range(n_bins):
    print(f"{r_centers[i]:12.1f} | {p_real[i]:12.2f} | {pair_counts[i]:12,}")

print()
print(f"Median |p(r)|: {np.median(np.abs(p_real)):.2f} μK")
print(f"Real amplitude: {A_real:.2f} μK")
print(f"Null mean: {null_amplitudes.mean():.2f} μK")
print(f"Null std: {null_amplitudes.std():.2f} μK")
print(f"Significance: {(A_real - null_amplitudes.mean()) / null_amplitudes.std():.2f} sigma")
print()

# =============================================================================
# Run Ledger Entry
# =============================================================================
ledger_entry = {
    'iteration': 1,
    'timestamp': datetime.now().isoformat(),
    'n_gal': len(ra),
    'n_pairs': int(pair_counts.sum()),
    'map_product': 'Planck SMICA R3.00',
    'masking': 'TMASK',
    'monopole_dipole': 'removed',
    'ell_filter': f"{CONFIG['ell_min']}-{CONFIG['ell_max']}",
    'temp_extraction': 'single pixel',
    'mean_field_subtraction': 'None',
    'median_abs_p': float(np.median(np.abs(p_real))),
    'amplitude_fit': float(A_real),
    'null_mean': float(null_amplitudes.mean()),
    'null_std': float(null_amplitudes.std()),
    'significance': float((A_real - null_amplitudes.mean()) / null_amplitudes.std()),
}

print("=" * 70)
print("RUN LEDGER ENTRY")
print("=" * 70)
for k, v in ledger_entry.items():
    print(f"  {k}: {v}")

# Save results
output_dir = Path("data/ksz/output/iteration1")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'config': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in CONFIG.items()},
    'ledger': ledger_entry,
    'r_centers': r_centers.tolist(),
    'p_ksz': p_real.tolist(),
    'pair_counts': pair_counts.tolist(),
    'null_amplitudes': null_amplitudes.tolist(),
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"Results saved to: {output_dir}/results.json")
print(f"Completed: {datetime.now().isoformat()}")
