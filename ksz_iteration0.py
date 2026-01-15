#!/usr/bin/env python3
"""
ITERATION 0: Diagnostic kSZ Pipeline
=====================================
Address top issues:
1. Apply Planck TMASK (galactic + point source mask)
2. Remove monopole/dipole from MAP before extraction
3. Verify c_ij formula
4. Small-scale test first

Run Ledger tracking included.
"""

import numpy as np
import healpy as hp
import json
import sys
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.cosmology import Planck18

# Configuration
CONFIG = {
    'iteration': 0,
    'n_gal_test': 10000,  # Small test first
    'r_bins': np.linspace(20, 150, 11),  # 10 bins
    'apply_mask': True,
    'remove_monopole_dipole': True,
    'ell_min': None,  # No filtering yet
    'ell_max': None,
    'random_seed': 42,
}

print("=" * 70)
print("ITERATION 0: DIAGNOSTIC kSZ PIPELINE")
print("=" * 70)
print(f"Started: {datetime.now().isoformat()}")
print()

# =============================================================================
# 1. Load and Process CMB Map with MASK
# =============================================================================
print("[1/6] Loading Planck SMICA map WITH mask...")

planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")

# Read temperature map and mask from same file
with fits.open(planck_file) as hdu:
    # SMICA is in K_CMB, need to convert to μK
    cmb_map_raw = hp.read_map(hdu, field=0, verbose=False) * 1e6  # μK
    # TMASK is field 3 (0=I, 1=Q, 2=U, 3=TMASK, 4=PMASK, ...)
    tmask = hp.read_map(hdu, field=3, verbose=False)

nside = hp.npix2nside(len(cmb_map_raw))
print(f"  NSIDE: {nside}")
print(f"  Map raw: min={cmb_map_raw.min():.1f}, max={cmb_map_raw.max():.1f}, std={cmb_map_raw.std():.1f} μK")
print(f"  TMASK: {tmask.sum()/len(tmask)*100:.1f}% unmasked")

# Apply mask
if CONFIG['apply_mask']:
    cmb_map = cmb_map_raw.copy()
    cmb_map[tmask == 0] = hp.UNSEEN
    valid_pix = cmb_map != hp.UNSEEN
    print(f"  After masking: {valid_pix.sum():,} valid pixels ({valid_pix.sum()/len(cmb_map)*100:.1f}%)")
else:
    cmb_map = cmb_map_raw.copy()
    valid_pix = np.ones(len(cmb_map), dtype=bool)

# Remove monopole and dipole from MAP
if CONFIG['remove_monopole_dipole']:
    print("  Removing monopole and dipole from map...")
    # hp.remove_dipole fits and removes monopole+dipole
    cmb_map_cleaned = hp.remove_dipole(cmb_map, gal_cut=0, fitval=True, verbose=False)
    if isinstance(cmb_map_cleaned, tuple):
        cmb_map = np.array(cmb_map_cleaned[0])  # Convert masked array to regular array
        mono, dipole = cmb_map_cleaned[1], cmb_map_cleaned[2]
        print(f"  Removed monopole: {mono:.2f} μK")
        print(f"  Removed dipole: {dipole}")
    else:
        cmb_map = np.array(cmb_map_cleaned)  # Convert masked array to regular array
    # Re-apply hp.UNSEEN to masked regions
    cmb_map[~valid_pix] = hp.UNSEEN
    print(f"  Map cleaned: mean={cmb_map[cmb_map != hp.UNSEEN].mean():.4f} μK")
else:
    print("  Skipping monopole/dipole removal")

print()

# =============================================================================
# 2. Load DESI LRG Catalog (small subset for test)
# =============================================================================
print("[2/6] Loading DESI LRG catalog (small subset)...")

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

print(f"  Total LRGs: {len(ra_full):,}")

# Subsample randomly for test
rng = np.random.default_rng(CONFIG['random_seed'])
n_test = min(CONFIG['n_gal_test'], len(ra_full))
idx = rng.choice(len(ra_full), size=n_test, replace=False)
ra = ra_full[idx]
dec = dec_full[idx]
z = z_full[idx]
weights = w_full[idx]

print(f"  Test sample: {n_test:,} galaxies")
print(f"  z range: {z.min():.3f} - {z.max():.3f}")
print()

# =============================================================================
# 3. Extract Temperatures at Galaxy Positions (with mask check)
# =============================================================================
print("[3/6] Extracting temperatures at galaxy positions...")

# Convert to HEALPix coordinates
theta = np.radians(90.0 - dec)
phi = np.radians(ra)
pix = hp.ang2pix(nside, theta, phi)

# Extract temperatures
temperatures = cmb_map[pix]

# Check which galaxies land on masked pixels
in_mask = temperatures != hp.UNSEEN
n_masked = (~in_mask).sum()
print(f"  Galaxies on masked pixels: {n_masked} ({n_masked/len(temperatures)*100:.1f}%)")

# Keep only unmasked galaxies
ra = ra[in_mask]
dec = dec[in_mask]
z = z[in_mask]
weights = weights[in_mask]
temperatures = temperatures[in_mask]

print(f"  Valid galaxies: {len(ra):,}")
print(f"  T at galaxies: mean={temperatures.mean():.2f}, std={temperatures.std():.2f} μK")
print()

# =============================================================================
# 4. Compute Comoving Positions
# =============================================================================
print("[4/6] Computing comoving positions...")

cosmo = Planck18
h = cosmo.H0.value / 100
chi = cosmo.comoving_distance(z).value * h  # Mpc/h

ra_rad = np.radians(ra)
dec_rad = np.radians(dec)
x = chi * np.cos(dec_rad) * np.cos(ra_rad)
y = chi * np.cos(dec_rad) * np.sin(ra_rad)
z_pos = chi * np.sin(dec_rad)
positions = np.column_stack([x, y, z_pos])

# Unit vectors from observer to galaxies
r_mag = np.sqrt(np.sum(positions**2, axis=1))
r_hat = positions / r_mag[:, np.newaxis]

print(f"  Comoving distance range: {chi.min():.1f} - {chi.max():.1f} Mpc/h")
print()

# =============================================================================
# 5. Compute Pairwise Estimator (verify c_ij)
# =============================================================================
print("[5/6] Computing pairwise estimator...")

r_bin_edges = CONFIG['r_bins']
r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])
n_bins = len(r_bin_edges) - 1

# Build KD-tree
tree = cKDTree(positions)

# Accumulators
numerator = np.zeros(n_bins)
denominator = np.zeros(n_bins)
pair_counts = np.zeros(n_bins, dtype=np.int64)

# Debug: track c_ij distribution
c_ij_samples = []

n = len(positions)
for i in range(n):
    neighbors = tree.query_ball_point(positions[i], r_bin_edges[-1])

    for j in neighbors:
        if j <= i:
            continue

        # Separation vector and magnitude
        dr = positions[j] - positions[i]
        r = np.sqrt(np.sum(dr**2))

        if r < r_bin_edges[0]:
            continue

        bin_idx = np.searchsorted(r_bin_edges, r) - 1
        if bin_idx < 0 or bin_idx >= n_bins:
            continue

        # c_ij: geometric projection factor
        # Standard definition: c_ij = (1/2) * r̂_ij · (r̂_i - r̂_j)
        r_hat_ij = dr / r  # Unit vector from i to j
        c_ij = 0.5 * np.dot(r_hat_ij, r_hat[i] - r_hat[j])

        # Weights and temperature difference
        ww = weights[i] * weights[j]
        dT = temperatures[i] - temperatures[j]

        # Accumulate
        numerator[bin_idx] += ww * dT * c_ij
        denominator[bin_idx] += ww * c_ij**2
        pair_counts[bin_idx] += 1

        # Sample c_ij for diagnostics
        if len(c_ij_samples) < 10000:
            c_ij_samples.append(c_ij)

    if i > 0 and i % 2000 == 0:
        print(f"    Processed {i:,}/{n:,} galaxies...")

c_ij_samples = np.array(c_ij_samples)

# Compute estimator
with np.errstate(divide='ignore', invalid='ignore'):
    p_ksz = np.where(denominator > 0, numerator / denominator, 0)

print(f"  Total pairs: {pair_counts.sum():,}")
print(f"  c_ij stats: mean={c_ij_samples.mean():.4f}, std={c_ij_samples.std():.4f}")
print(f"  c_ij range: [{c_ij_samples.min():.4f}, {c_ij_samples.max():.4f}]")
print()

# =============================================================================
# 6. Results and Diagnostics
# =============================================================================
print("[6/6] Results...")
print()
print("=" * 70)
print("PAIRWISE MOMENTUM p(r)")
print("=" * 70)
print(f"{'r (Mpc/h)':>12} | {'p(r) (μK)':>12} | {'N_pairs':>12}")
print("-" * 42)
for i in range(n_bins):
    print(f"{r_centers[i]:12.1f} | {p_ksz[i]:12.2f} | {pair_counts[i]:12,}")

print()
print(f"Median |p(r)|: {np.median(np.abs(p_ksz)):.2f} μK")
print(f"Mean |p(r)|: {np.mean(np.abs(p_ksz)):.2f} μK")
print(f"Max |p(r)|: {np.max(np.abs(p_ksz)):.2f} μK")
print()

# Simple amplitude fit (for comparison)
template = np.exp(-r_centers / 60.0)
valid = pair_counts > 100
if valid.sum() >= 3:
    # Simple weighted fit assuming uniform errors for now
    w = pair_counts[valid].astype(float)
    A_fit = np.sum(w * p_ksz[valid] * template[valid]) / np.sum(w * template[valid]**2)
    print(f"Amplitude fit (crude): A = {A_fit:.2f} μK")
else:
    A_fit = np.nan
    print("Insufficient pairs for amplitude fit")

print()

# =============================================================================
# Run Ledger Entry
# =============================================================================
ledger_entry = {
    'iteration': 0,
    'timestamp': datetime.now().isoformat(),
    'n_gal': len(ra),
    'map_product': 'Planck SMICA R3.00',
    'masking': 'TMASK (galactic+PS)' if CONFIG['apply_mask'] else 'None',
    'monopole_dipole': 'removed' if CONFIG['remove_monopole_dipole'] else 'not removed',
    'ell_filter': f"{CONFIG['ell_min']}-{CONFIG['ell_max']}" if CONFIG['ell_min'] else 'None',
    'temp_extraction': 'single pixel',
    'mean_field_subtraction': 'None',
    'median_abs_p': float(np.median(np.abs(p_ksz))),
    'mean_abs_p': float(np.mean(np.abs(p_ksz))),
    'amplitude_fit': float(A_fit) if not np.isnan(A_fit) else None,
    'n_pairs_total': int(pair_counts.sum()),
    'conclusion': 'Iteration 0 complete - check amplitude scale',
}

print("=" * 70)
print("RUN LEDGER ENTRY")
print("=" * 70)
for k, v in ledger_entry.items():
    print(f"  {k}: {v}")

# Save results
output_dir = Path("data/ksz/output/iteration0")
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'config': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in CONFIG.items()},
    'ledger': ledger_entry,
    'r_centers': r_centers.tolist(),
    'p_ksz': p_ksz.tolist(),
    'pair_counts': pair_counts.tolist(),
    'c_ij_mean': float(c_ij_samples.mean()),
    'c_ij_std': float(c_ij_samples.std()),
}

with open(output_dir / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f"Results saved to: {output_dir}/results.json")
print()
print(f"Completed: {datetime.now().isoformat()}")
