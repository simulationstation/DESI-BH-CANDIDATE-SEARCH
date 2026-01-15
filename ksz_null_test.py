#!/usr/bin/env python3
"""
Quick null test: shuffle temperatures and compute pairwise.
Should give amplitude consistent with zero.
"""

import numpy as np
import healpy as hp
import json
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
from astropy.io import fits
from astropy.cosmology import Planck18

print("=" * 70)
print("NULL TEST: Temperature Shuffle")
print("=" * 70)

# Load data (same as iteration 0)
planck_file = Path("data/ksz/maps/COM_CMB_IQU-smica_2048_R3.00_full.fits")
with fits.open(planck_file) as hdu:
    cmb_map_raw = hp.read_map(hdu, field=0, verbose=False) * 1e6
    tmask = hp.read_map(hdu, field=3, verbose=False)

nside = hp.npix2nside(len(cmb_map_raw))
cmb_map = cmb_map_raw.copy()
cmb_map[tmask == 0] = hp.UNSEEN
valid_pix = cmb_map != hp.UNSEEN

# Remove monopole/dipole
cmb_map_cleaned = hp.remove_dipole(cmb_map, gal_cut=0, fitval=True, verbose=False)
cmb_map = np.array(cmb_map_cleaned[0])
cmb_map[~valid_pix] = hp.UNSEEN

# Load DESI
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

# Subsample
rng = np.random.default_rng(42)
n_test = 10000
idx = rng.choice(len(ra_full), size=n_test, replace=False)
ra, dec, z, weights = ra_full[idx], dec_full[idx], z_full[idx], w_full[idx]

# Extract temperatures
theta = np.radians(90.0 - dec)
phi = np.radians(ra)
pix = hp.ang2pix(nside, theta, phi)
temperatures = cmb_map[pix]

# Mask check
in_mask = temperatures != hp.UNSEEN
ra = ra[in_mask]
dec = dec[in_mask]
z = z[in_mask]
weights = weights[in_mask]
temperatures = temperatures[in_mask]

print(f"Valid galaxies: {len(ra)}")

# Compute positions
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

r_bin_edges = np.linspace(20, 150, 11)
r_centers = 0.5 * (r_bin_edges[:-1] + r_bin_edges[1:])

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

    with np.errstate(divide='ignore', invalid='ignore'):
        p_ksz = np.where(den > 0, num / den, 0)
    return p_ksz, counts

# Compute real
print("\nComputing REAL pairwise...")
p_real, counts = compute_pairwise(positions, temperatures, weights, r_hat, r_bin_edges)
print(f"Median |p| (real): {np.median(np.abs(p_real)):.2f} μK")

# Run null tests
n_null = 20
null_amplitudes = []

print(f"\nRunning {n_null} null tests (shuffle T)...")
for i in range(n_null):
    T_shuffled = temperatures.copy()
    rng.shuffle(T_shuffled)
    p_null, _ = compute_pairwise(positions, T_shuffled, weights, r_hat, r_bin_edges)

    # Fit amplitude
    template = np.exp(-r_centers / 60.0)
    valid = counts > 100
    w = counts[valid].astype(float)
    A = np.sum(w * p_null[valid] * template[valid]) / np.sum(w * template[valid]**2)
    null_amplitudes.append(A)

    if (i+1) % 5 == 0:
        print(f"  {i+1}/{n_null} done")

null_amplitudes = np.array(null_amplitudes)

# Real amplitude
template = np.exp(-r_centers / 60.0)
valid = counts > 100
w = counts[valid].astype(float)
A_real = np.sum(w * p_real[valid] * template[valid]) / np.sum(w * template[valid]**2)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Real amplitude: {A_real:.2f} μK")
print(f"Null mean: {null_amplitudes.mean():.2f} μK")
print(f"Null std: {null_amplitudes.std():.2f} μK")
print(f"Significance: {(A_real - null_amplitudes.mean()) / null_amplitudes.std():.2f} sigma")
print()

if np.abs(null_amplitudes.mean()) < 2 * null_amplitudes.std():
    print("✓ Null test PASSES - shuffled amplitudes are consistent with zero")
else:
    print("✗ Null test FAILS - shuffled amplitudes have significant offset")
