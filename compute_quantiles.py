#!/usr/bin/env python3
"""
compute_quantiles.py - Compute distribution summaries for Tier A candidates

This script loads triage CSVs and computes quantiles to inform gating thresholds.
"""

import numpy as np
from pathlib import Path


def load_tier_a(csv_path):
    """Load Tier A candidates from triage CSV."""
    candidates = []
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        col_idx = {name: i for i, name in enumerate(header)}

        for line in f:
            parts = line.strip().split(',')
            if parts[col_idx['tier']] == 'A':
                candidates.append({
                    'targetid': int(parts[col_idx['targetid']]),
                    'n_epochs': int(parts[col_idx['n_epochs']]),
                    'mjd_span': float(parts[col_idx['mjd_span']]),
                    'S': float(parts[col_idx['S']]),
                    'S_robust': float(parts[col_idx['S_robust']]),
                    'd_max': float(parts[col_idx['d_max']]),
                    'rv_err_median_kms': float(parts[col_idx['rv_err_median_kms']]),
                    'delta_rv_kms': float(parts[col_idx['delta_rv_kms']])
                })
    return candidates


def print_quantiles(values, name):
    """Print quantile summary."""
    percentiles = [50, 80, 90, 95, 99]
    print(f"  {name}:")
    print(f"    min:  {np.min(values):.2f}")
    for p in percentiles:
        print(f"    {p}th:  {np.percentile(values, p):.2f}")
    print(f"    max:  {np.max(values):.2f}")


def main():
    base_dir = Path('.')

    for survey in ['bright', 'dark']:
        csv_path = base_dir / f'data/derived/triage_candidates_{survey}.csv'
        candidates = load_tier_a(csv_path)

        print(f"\n{'='*70}")
        print(f"TIER A DISTRIBUTION SUMMARY: main-{survey}")
        print(f"{'='*70}")
        print(f"Total Tier A candidates: {len(candidates)}")

        if len(candidates) == 0:
            continue

        # Extract arrays
        S_robust = np.array([c['S_robust'] for c in candidates])
        d_max = np.array([c['d_max'] for c in candidates])
        mjd_span = np.array([c['mjd_span'] for c in candidates])
        n_epochs = np.array([c['n_epochs'] for c in candidates])
        rv_err_med = np.array([c['rv_err_median_kms'] for c in candidates])

        print_quantiles(S_robust, "S_robust")
        print_quantiles(d_max, "d_max")
        print_quantiles(mjd_span, "mjd_span (days)")
        print_quantiles(n_epochs, "n_epochs")
        print_quantiles(rv_err_med, "rv_err_median (km/s)")

        # Test different threshold combinations
        print(f"\n--- THRESHOLD IMPACT ANALYSIS ---")

        thresholds = [
            {'S_robust_min': 5, 'd_max_max': None, 'mjd_span_min': 0.5},
            {'S_robust_min': 10, 'd_max_max': None, 'mjd_span_min': 0.5},
            {'S_robust_min': 10, 'd_max_max': 200, 'mjd_span_min': 0.5},
            {'S_robust_min': 10, 'd_max_max': 100, 'mjd_span_min': 0.5},
            {'S_robust_min': 15, 'd_max_max': None, 'mjd_span_min': 0.5},
            {'S_robust_min': 15, 'd_max_max': 150, 'mjd_span_min': 0.5},
            {'S_robust_min': 20, 'd_max_max': None, 'mjd_span_min': 0.5},
            {'S_robust_min': 20, 'd_max_max': 150, 'mjd_span_min': 0.5},
        ]

        print(f"{'S_robust>=':<12} {'d_max<=':<10} {'mjd>=':<8} {'Count':<8} {'Survivors':<10}")
        print("-" * 50)

        for t in thresholds:
            count = 0
            for c in candidates:
                if c['S_robust'] >= t['S_robust_min']:
                    if c['mjd_span'] >= t['mjd_span_min']:
                        if t['d_max_max'] is None or c['d_max'] <= t['d_max_max']:
                            count += 1

            d_max_str = str(t['d_max_max']) if t['d_max_max'] else "None"
            print(f"{t['S_robust_min']:<12} {d_max_str:<10} {t['mjd_span_min']:<8} {count:<8}")

        # Correlation analysis
        print(f"\n--- CORRELATION: S_robust vs d_max ---")
        # Check if high S_robust implies high d_max
        high_S = S_robust >= np.percentile(S_robust, 80)
        high_d = d_max >= np.percentile(d_max, 80)
        both_high = np.sum(high_S & high_d)
        print(f"  Candidates with S_robust >= 80th percentile: {np.sum(high_S)}")
        print(f"  Of those, also with d_max >= 80th percentile: {both_high}")

        # List candidates with high S_robust but moderate d_max
        print(f"\n--- BEST CANDIDATES: S_robust >= 15, d_max <= 150 ---")
        good = [c for c in candidates if c['S_robust'] >= 15 and c['d_max'] <= 150]
        good.sort(key=lambda x: x['S_robust'], reverse=True)

        if len(good) > 0:
            print(f"{'TARGETID':<22} {'N':<4} {'MJD_span':<10} {'S_robust':<10} {'d_max':<10}")
            print("-" * 60)
            for c in good[:20]:
                print(f"{c['targetid']:<22} {c['n_epochs']:<4} {c['mjd_span']:<10.1f} "
                      f"{c['S_robust']:<10.1f} {c['d_max']:<10.1f}")
        else:
            print("  (none)")


if __name__ == '__main__':
    main()
