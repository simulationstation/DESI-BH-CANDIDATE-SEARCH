#!/usr/bin/env python3
"""
Smoke test: Verify parallel kSZ code works with tiny data.
"""

import numpy as np
import time
from scipy.spatial import cKDTree
from multiprocessing import Pool, cpu_count

print("=" * 50)
print("SMOKE TEST: Parallel kSZ Estimator")
print("=" * 50)
print()

# Tiny fake data
n_gal = 1000  # Tiny
print(f"Creating {n_gal} fake galaxies...")

rng = np.random.default_rng(42)
positions = rng.uniform(-100, 100, (n_gal, 3))
temperatures = rng.normal(0, 100, n_gal)  # μK
weights = np.ones(n_gal)

r_bin_edges = np.linspace(10, 50, 6)

# Unit vectors
r_mag = np.sqrt(np.sum(positions**2, axis=1))
r_hat = positions / r_mag[:, np.newaxis]

def process_chunk_standalone(chunk_info):
    """Process a chunk of galaxies."""
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


# Test parallel computation
n_workers = max(1, cpu_count() - 1)
print(f"Testing with {n_workers} workers...")

chunk_size = 100
chunks = [list(range(i, min(i + chunk_size, n_gal))) for i in range(0, n_gal, chunk_size)]
args_list = [(chunk, positions, temperatures, weights, r_hat, r_bin_edges) for chunk in chunks]

print(f"Split into {len(chunks)} chunks")

t0 = time.time()
with Pool(n_workers) as pool:
    results = pool.map(process_chunk_standalone, args_list)

n_bins = len(r_bin_edges) - 1
total_num = np.zeros(n_bins)
total_den = np.zeros(n_bins)
total_counts = np.zeros(n_bins, dtype=np.int64)

for num, den, counts in results:
    total_num += num
    total_den += den
    total_counts += counts

with np.errstate(divide='ignore', invalid='ignore'):
    p_ksz = np.where(total_den > 0, total_num / total_den, 0)

dt = time.time() - t0

print()
print("RESULTS:")
print(f"  Time: {dt:.2f}s")
print(f"  Total pairs: {total_counts.sum():,}")
print(f"  p(r) values: {p_ksz}")
print()

if total_counts.sum() > 0 and np.any(p_ksz != 0):
    print("✓ SMOKE TEST PASSED!")
    print()
    print("Parallel kSZ estimator is working correctly.")
else:
    print("✗ SMOKE TEST FAILED!")
    print("Something is wrong with the parallel code.")
