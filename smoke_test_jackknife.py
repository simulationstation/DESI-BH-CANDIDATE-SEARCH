#!/usr/bin/env python3
"""
Smoke test for jackknife kSZ analysis on small subset of real data.
"""

import numpy as np
import sys
import time
from pathlib import Path
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count

print("=" * 60)
print("SMOKE TEST: Jackknife kSZ on Real Data (TINY SUBSET)")
print("=" * 60)
print()

# =============================================================================
# Load small subset of real data
# =============================================================================
print("[1] Loading TINY subset of real data...")

from astropy.io import fits
import healpy as hp
from astropy.cosmology import Planck18

ngc_file = Path("data/ksz/catalogs/LRG_NGC_clustering.dat.fits")
planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")

with fits.open(ngc_file) as hdu:
    data = hdu[1].data

# RANDOMLY sample 15000 galaxies across the survey (not contiguous!)
N_TEST = 15000
rng = np.random.default_rng(42)
idx = rng.choice(len(data), size=N_TEST, replace=False)
ra = data['RA'][idx]
dec = data['DEC'][idx]
z = data['Z'][idx]
weights = data['WEIGHT_SYS'][idx] * data['WEIGHT_COMP'][idx]

cmb_map = hp.read_map(str(planck_file), field=0, verbose=False)
cmb_map = cmb_map * 1e6  # μK
nside = hp.npix2nside(len(cmb_map))

# Compute positions
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

# Subtract mean
temperatures = temperatures - temperatures.mean()

print(f"  N galaxies: {len(ra)}")
print(f"  Temperature std: {temperatures.std():.2f} μK")
print()

# =============================================================================
# Pairwise momentum function
# =============================================================================
def compute_pairwise(pos, T, w, r_bin_edges):
    """Simple single-threaded pairwise for smoke test."""
    n = len(pos)
    tree = cKDTree(pos)

    r_mag = np.sqrt(np.sum(pos**2, axis=1))
    r_hat = pos / r_mag[:, np.newaxis]

    n_bins = len(r_bin_edges) - 1
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

    with np.errstate(divide='ignore', invalid='ignore'):
        p_ksz = np.where(den > 0, num / den, 0)

    return p_ksz, counts

# =============================================================================
# Compute with simple jackknife
# =============================================================================
print("[2] Computing pairwise momentum with jackknife...")

r_bin_edges = np.linspace(20, 150, 11)  # 10 bins
r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

# Full sample
t0 = time.time()
p_full, counts_full = compute_pairwise(positions, temperatures, weights, r_bin_edges)
t_full = time.time() - t0

print(f"  Full sample time: {t_full:.1f}s")
print(f"  Total pairs: {counts_full.sum():,}")
print(f"  p(r) full: {p_full.round(2)}")
print()

# Simple 10-fold jackknife
n_jk = 10
n_bins = len(r_bin_edges) - 1
n_gal = len(positions)

indices = np.arange(n_gal)
np.random.shuffle(indices)
chunk_size = n_gal // n_jk

p_jk = np.zeros((n_jk, n_bins))
print(f"[3] Running {n_jk}-fold jackknife...")

for i_jk in range(n_jk):
    # Exclude chunk i_jk
    exclude_start = i_jk * chunk_size
    exclude_end = exclude_start + chunk_size
    mask = np.ones(n_gal, dtype=bool)
    mask[indices[exclude_start:exclude_end]] = False

    p_jk[i_jk], _ = compute_pairwise(positions[mask], temperatures[mask], weights[mask], r_bin_edges)

    print(f"    Jackknife {i_jk + 1}/{n_jk}")

# Compute jackknife error
p_err = np.sqrt((n_jk - 1) * np.var(p_jk, axis=0))

print()
print("[4] RESULTS:")
print("=" * 60)
print()
print(f"r (Mpc/h) | p(r) (μK) | σ_p (μK) | S/N")
print("-" * 60)
for i in range(n_bins):
    snr = p_full[i] / p_err[i] if p_err[i] > 0 else 0
    print(f"{r_centers[i]:8.1f} | {p_full[i]:9.2f} | {p_err[i]:8.2f} | {snr:6.2f}")

print()

# Fit simple amplitude
template = np.exp(-r_centers / 60.0)
valid = p_err > 0
w = 1.0 / p_err[valid]**2
A_fit = np.sum(w * p_full[valid] * template[valid]) / np.sum(w * template[valid]**2)
A_err = 1.0 / np.sqrt(np.sum(w * template[valid]**2))
snr_total = np.abs(A_fit) / A_err

print(f"Template amplitude: A = {A_fit:.2f} ± {A_err:.2f} μK")
print(f"Template S/N: {snr_total:.2f}")
print()

# Sanity checks
print("[5] SANITY CHECKS:")
print("-" * 60)

checks_passed = 0
checks_total = 4

# Check 1: Errors should be O(50-1000) μK (CMB variance / sqrt(N_pairs))
mean_err = np.mean(p_err)
if 10 < mean_err < 1000:
    print(f"✓ Errors are reasonable: {mean_err:.1f} μK")
    checks_passed += 1
else:
    print(f"✗ Errors look wrong: {mean_err:.1f} μK (expected 10-1000 μK)")

# Check 2: Amplitude should be consistent with noise (within 10x error)
if np.abs(A_fit) < 10 * A_err:
    print(f"✓ Amplitude consistent with noise: {A_fit:.1f} ± {A_err:.1f} μK")
    checks_passed += 1
else:
    print(f"✗ Amplitude too large compared to error: {A_fit:.1f} ± {A_err:.1f} μK")

# Check 3: S/N should be modest with tiny sample
if np.abs(snr_total) < 10:
    print(f"✓ S/N is modest: {snr_total:.2f}")
    checks_passed += 1
else:
    print(f"✗ S/N suspiciously high: {snr_total:.2f} (expected <10 with {N_TEST} galaxies)")

# Check 4: Pair counts should be > 0
if counts_full.sum() > 1000:
    print(f"✓ Pair counts reasonable: {counts_full.sum():,}")
    checks_passed += 1
else:
    print(f"✗ Too few pairs: {counts_full.sum()}")

print()
print(f"RESULT: {checks_passed}/{checks_total} checks passed")
print()

if checks_passed == checks_total:
    print("✓ SMOKE TEST PASSED - Ready for full analysis")
else:
    print("✗ SMOKE TEST FAILED - Need to fix issues before full run")
