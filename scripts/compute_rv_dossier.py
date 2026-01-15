#!/usr/bin/env python3
"""
compute_rv_dossier.py - Compute RV metrics for a target using exact pipeline definitions

Computes: S, S_min_LOO, S_robust, d_max (leverage), weighted mean, chi2 constant model
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RVEpoch:
    mjd: float
    rv: float      # km/s
    rv_err: float  # km/s

def compute_S(epochs: List[RVEpoch]) -> float:
    """
    Compute global RV significance metric S.

    S = ΔRV_max / sqrt(sum(σ_RV,i²))

    Symbol table:
    | Symbol      | Definition                           | Units  |
    |-------------|--------------------------------------|--------|
    | S           | Global RV significance               | -      |
    | ΔRV_max     | max(RV) - min(RV)                   | km/s   |
    | σ_RV,i      | RV uncertainty for epoch i           | km/s   |
    """
    rvs = np.array([e.rv for e in epochs])
    errs = np.array([e.rv_err for e in epochs])

    delta_rv_max = np.max(rvs) - np.min(rvs)
    noise_term = np.sqrt(np.sum(errs**2))

    return delta_rv_max / noise_term

def compute_S_LOO(epochs: List[RVEpoch]) -> Tuple[float, int]:
    """
    Compute leave-one-out minimum significance.

    S_min_LOO = min over all i of S(epochs excluding i)

    Returns: (S_min_LOO, index of epoch that when removed gives minimum)
    """
    n = len(epochs)
    if n < 3:
        return 0.0, -1

    S_values = []
    for i in range(n):
        subset = [e for j, e in enumerate(epochs) if j != i]
        S_values.append(compute_S(subset))

    min_idx = np.argmin(S_values)
    return S_values[min_idx], min_idx

def compute_leverage_d_max(epochs: List[RVEpoch]) -> Tuple[float, int]:
    """
    Compute leverage metric d_max.

    d_max = max over all i of |RV_i - RV_mean| / σ_RV,i

    This identifies epochs that have outsized influence on the variability signal.

    Symbol table:
    | Symbol   | Definition                              | Units  |
    |----------|----------------------------------------|--------|
    | d_max    | Maximum normalized deviation            | -      |
    | RV_i     | RV at epoch i                          | km/s   |
    | RV_mean  | Weighted mean RV                       | km/s   |
    | σ_RV,i   | RV uncertainty at epoch i              | km/s   |
    """
    rvs = np.array([e.rv for e in epochs])
    errs = np.array([e.rv_err for e in epochs])

    # Weighted mean
    weights = 1.0 / errs**2
    rv_wmean = np.sum(weights * rvs) / np.sum(weights)

    # Normalized deviations
    d_values = np.abs(rvs - rv_wmean) / errs

    max_idx = np.argmax(d_values)
    return d_values[max_idx], max_idx

def compute_weighted_mean_and_chi2(epochs: List[RVEpoch]) -> Tuple[float, float, float, int]:
    """
    Compute weighted mean RV and chi² for constant-RV model.

    χ² = Σ [(RV_i - RV_wmean)² / σ_RV,i²]

    Symbol table:
    | Symbol    | Definition                           | Units  |
    |-----------|--------------------------------------|--------|
    | RV_wmean  | Weighted mean RV                     | km/s   |
    | χ²        | Chi-squared statistic                | -      |
    | dof       | Degrees of freedom (N-1)             | -      |
    | p_const   | p-value for constant model           | -      |

    Returns: (weighted_mean, chi2, reduced_chi2, dof)
    """
    from scipy import stats

    rvs = np.array([e.rv for e in epochs])
    errs = np.array([e.rv_err for e in epochs])

    weights = 1.0 / errs**2
    rv_wmean = np.sum(weights * rvs) / np.sum(weights)

    chi2 = np.sum((rvs - rv_wmean)**2 / errs**2)
    dof = len(epochs) - 1
    reduced_chi2 = chi2 / dof if dof > 0 else np.inf

    return rv_wmean, chi2, reduced_chi2, dof

def check_night_consistency(epochs: List[RVEpoch], mjd_threshold: float = 0.5) -> dict:
    """
    Check consistency of epochs observed on the same night.

    For epochs within mjd_threshold days of each other, compute:
    - difference in RV
    - significance of difference
    """
    results = []
    n = len(epochs)

    for i in range(n):
        for j in range(i+1, n):
            dt = abs(epochs[i].mjd - epochs[j].mjd)
            if dt < mjd_threshold:
                diff = epochs[i].rv - epochs[j].rv
                err_diff = np.sqrt(epochs[i].rv_err**2 + epochs[j].rv_err**2)
                sig = abs(diff) / err_diff
                results.append({
                    'epoch_i': i,
                    'epoch_j': j,
                    'mjd_i': epochs[i].mjd,
                    'mjd_j': epochs[j].mjd,
                    'dt_days': dt,
                    'rv_diff_kms': diff,
                    'err_diff_kms': err_diff,
                    'significance': sig,
                    'consistent': sig < 3.0
                })

    return results

def compute_full_dossier(epochs: List[RVEpoch]) -> dict:
    """Compute all metrics for a target."""

    rvs = np.array([e.rv for e in epochs])
    errs = np.array([e.rv_err for e in epochs])
    mjds = np.array([e.mjd for e in epochs])

    # Helper to convert numpy types to native Python for JSON serialization
    def to_native(x):
        if hasattr(x, 'item'):
            return x.item()
        return x

    # Basic stats
    delta_rv_max = np.max(rvs) - np.min(rvs)
    K_est = delta_rv_max / 2.0
    mjd_span = np.max(mjds) - np.min(mjds)

    # Core metrics
    S = compute_S(epochs)
    S_min_LOO, loo_drop_idx = compute_S_LOO(epochs)
    S_robust = min(S, S_min_LOO)
    d_max, d_max_idx = compute_leverage_d_max(epochs)

    # Weighted mean and chi2
    rv_wmean, chi2, chi2_red, dof = compute_weighted_mean_and_chi2(epochs)

    # Night consistency
    night_checks = check_night_consistency(epochs)
    # Convert night_checks to native Python types
    for check in night_checks:
        for k, v in check.items():
            if hasattr(v, 'item'):
                check[k] = v.item()
            elif isinstance(v, np.bool_):
                check[k] = bool(v)

    return {
        'n_epochs': len(epochs),
        'mjd_span_days': to_native(mjd_span),
        'delta_rv_max_kms': to_native(delta_rv_max),
        'K_est_kms': to_native(K_est),
        'rv_weighted_mean_kms': to_native(rv_wmean),
        'chi2_constant': to_native(chi2),
        'chi2_reduced': to_native(chi2_red),
        'dof': to_native(dof),
        'S': to_native(S),
        'S_min_LOO': to_native(S_min_LOO),
        'S_robust': to_native(S_robust),
        'loo_drop_epoch_idx': to_native(loo_drop_idx),
        'd_max': to_native(d_max),
        'd_max_epoch_idx': to_native(d_max_idx),
        'high_leverage': bool(d_max > 100),
        'loo_significant_drop': bool(S_min_LOO < S * 0.5),
        'night_consistency_checks': night_checks,
        'epochs': [{'mjd': e.mjd, 'rv_kms': e.rv, 'rv_err_kms': e.rv_err} for e in epochs]
    }

def main():
    # Gaia DR3 3802130935635096832 epochs (EXACT VALUES)
    epochs = [
        RVEpoch(mjd=59568.48825, rv=-86.39, rv_err=0.55),
        RVEpoch(mjd=59605.38003, rv=59.68, rv_err=0.83),
        RVEpoch(mjd=59607.37393, rv=26.43, rv_err=1.06),
        RVEpoch(mjd=59607.38852, rv=25.16, rv_err=1.11),
    ]

    print("=" * 70)
    print("RV DOSSIER: Gaia DR3 3802130935635096832")
    print("=" * 70)
    print()

    # Print epoch table
    print("PER-EPOCH RV TABLE")
    print("-" * 50)
    print(f"{'#':<3} {'MJD':<14} {'RV (km/s)':<12} {'σRV (km/s)':<12}")
    print("-" * 50)
    for i, e in enumerate(epochs):
        print(f"{i:<3} {e.mjd:<14.5f} {e.rv:<12.2f} {e.rv_err:<12.2f}")
    print()

    # Compute dossier
    dossier = compute_full_dossier(epochs)

    # Print derived quantities
    print("DERIVED QUANTITIES")
    print("-" * 50)
    print(f"ΔRV_max         = {dossier['delta_rv_max_kms']:.2f} km/s")
    print(f"K_est = ΔRV/2   = {dossier['K_est_kms']:.2f} km/s")
    print(f"MJD span        = {dossier['mjd_span_days']:.2f} days")
    print(f"RV weighted mean= {dossier['rv_weighted_mean_kms']:.2f} km/s")
    print()

    # Print chi2 analysis
    print("CONSTANT-RV MODEL TEST")
    print("-" * 50)
    print(f"χ² (constant)   = {dossier['chi2_constant']:.2f}")
    print(f"χ²_reduced      = {dossier['chi2_reduced']:.2f}")
    print(f"dof             = {dossier['dof']}")
    print(f"  → Constant RV rejected at high significance (χ²_red >> 1)")
    print()

    # Print significance metrics
    print("RV SELECTION METRICS (PIPELINE DEFINITIONS)")
    print("-" * 50)
    print(f"S               = {dossier['S']:.4f}")
    print(f"S_min_LOO       = {dossier['S_min_LOO']:.4f}  (dropping epoch {dossier['loo_drop_epoch_idx']})")
    print(f"S_robust        = {dossier['S_robust']:.4f}  = min(S, S_min_LOO)")
    print()
    print(f"d_max           = {dossier['d_max']:.4f}  (epoch {dossier['d_max_epoch_idx']})")
    print(f"High leverage?  = {dossier['high_leverage']}  (d_max > 100)")
    print(f"LOO drops sig?  = {dossier['loo_significant_drop']}  (S_min_LOO < 0.5*S)")
    print()

    # Night consistency
    print("NIGHT CONSISTENCY CHECK")
    print("-" * 50)
    if dossier['night_consistency_checks']:
        for check in dossier['night_consistency_checks']:
            print(f"Epochs {check['epoch_i']} & {check['epoch_j']}: Δt = {check['dt_days']*24:.2f} hr")
            print(f"  ΔRV = {check['rv_diff_kms']:.2f} ± {check['err_diff_kms']:.2f} km/s")
            print(f"  Significance = {check['significance']:.2f}σ")
            print(f"  Consistent (< 3σ)? {check['consistent']}")
    else:
        print("No same-night observations found.")
    print()

    # Save to JSON
    with open('candidate_dossier.json', 'w') as f:
        json.dump(dossier, f, indent=2)
    print("Saved: candidate_dossier.json")

    return dossier

if __name__ == "__main__":
    main()
