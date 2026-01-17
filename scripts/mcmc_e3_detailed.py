#!/usr/bin/env python3
"""
Run MCMC orbital fitting on E3_dwd_lisa candidates and generate detailed reports.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, '/home/primary/DESI-BH-CANDIDATE-SEARCH')

# Physical constants
G_SI = 6.674e-11  # m^3 kg^-1 s^-2
M_SUN = 1.989e30  # kg
DAY_S = 86400     # seconds per day

OUTPUT_DIR = '/home/primary/DESI-BH-CANDIDATE-SEARCH/detailed_reports_e3'

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """Solve Kepler's equation M = E - e*sin(E) for E."""
    M = np.atleast_1d(M)
    E = M.copy()
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E = E + dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def rv_model_keplerian(t, P, K, e, omega, T0, gamma):
    """Keplerian radial velocity model."""
    M = 2 * np.pi * (t - T0) / P
    M = M % (2 * np.pi)
    E = solve_kepler(M, e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))
    rv = gamma + K * (np.cos(nu + omega) + e * np.cos(omega))
    return rv


def mass_function(P_days, K_kms):
    """Compute mass function f(M) in solar masses."""
    P_s = P_days * DAY_S
    K_ms = K_kms * 1000
    f_M = (P_s * K_ms**3) / (2 * np.pi * G_SI)
    return f_M / M_SUN


def min_companion_mass(f_M, M1):
    """Solve for minimum companion mass (sin i = 1)."""
    from scipy.optimize import brentq
    def equation(M2):
        return M2**3 / (M1 + M2)**2 - f_M
    try:
        return brentq(equation, 0.01, 100)
    except:
        return np.nan


class CandidateMCMC:
    """Run MCMC for a single candidate."""

    def __init__(self, targetid, t_obs, rv_obs, rv_err, candidate_info, M1_assumed=0.5):
        self.targetid = targetid
        self.t_obs = np.array(t_obs)
        self.rv_obs = np.array(rv_obs)
        self.rv_err = np.array(rv_err)
        self.candidate_info = candidate_info
        self.M1 = M1_assumed

        # Get initial guesses from fast screen
        self.P_init = candidate_info.get('best_period', 20.0)
        self.K_init = candidate_info.get('best_K', 80.0)

    def log_likelihood(self, theta):
        """Gaussian log-likelihood for RV data."""
        P, K, e, omega, T0, gamma = theta
        rv_model = rv_model_keplerian(self.t_obs, P, K, e, omega, T0, gamma)
        chi2 = np.sum(((self.rv_obs - rv_model) / self.rv_err)**2)
        return -0.5 * chi2

    def log_prior(self, theta):
        """Log-prior for orbital parameters."""
        P, K, e, omega, T0, gamma = theta

        # Adaptive bounds based on data
        rv_range = np.ptp(self.rv_obs)
        rv_mean = np.mean(self.rv_obs)
        baseline = np.ptp(self.t_obs)

        # Period bounds: 0.5 days to 3x baseline
        P_min, P_max = 0.5, max(200, 3 * baseline)

        # K bounds: based on RV range
        K_min, K_max = 5, max(300, rv_range)

        # Gamma bounds: based on RV mean
        gamma_min = rv_mean - 200
        gamma_max = rv_mean + 200

        if not (P_min < P < P_max):
            return -np.inf
        if not (K_min < K < K_max):
            return -np.inf
        if not (0 <= e < 0.9):
            return -np.inf
        if not (0 <= omega < 2 * np.pi):
            return -np.inf
        if not (self.t_obs.min() - P_max < T0 < self.t_obs.max() + P_max):
            return -np.inf
        if not (gamma_min < gamma < gamma_max):
            return -np.inf

        # Log-uniform prior on P
        return -np.log(P)

    def log_probability(self, theta):
        """Log-posterior = log-prior + log-likelihood."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_mcmc(self, n_walkers=32, n_steps=3000, n_burn=500):
        """Run MCMC sampling."""
        import emcee

        ndim = 6

        # Initialize walkers around fast-screen solution
        pos = np.zeros((n_walkers, ndim))

        rv_mean = np.mean(self.rv_obs)

        for i in range(n_walkers):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                pos[i, 0] = self.P_init * np.random.uniform(0.5, 2.0)  # P
                pos[i, 1] = self.K_init * np.random.uniform(0.5, 1.5)  # K
                pos[i, 2] = np.random.uniform(0, 0.5)  # e
                pos[i, 3] = np.random.uniform(0, 2*np.pi)  # omega
                pos[i, 4] = self.t_obs.mean() + np.random.uniform(-50, 50)  # T0
                pos[i, 5] = rv_mean + np.random.uniform(-30, 30)  # gamma

                if np.isfinite(self.log_probability(pos[i])):
                    valid = True
                attempts += 1

            if not valid:
                # Fall back to safe defaults
                pos[i] = [20, 50, 0.1, np.pi, self.t_obs.mean(), rv_mean]

        # Run sampler (single process to avoid nested parallelism)
        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_probability)
        sampler.run_mcmc(pos, n_steps, progress=False)

        # Get samples
        samples = sampler.get_chain(discard=n_burn, flat=True)
        log_probs = sampler.get_log_prob(discard=n_burn, flat=True)

        return samples, log_probs

    def analyze_samples(self, samples, log_probs):
        """Analyze MCMC samples and compute derived quantities."""
        labels = ['P', 'K', 'e', 'omega', 'T0', 'gamma']

        results = {}

        # Summary statistics
        for i, label in enumerate(labels):
            results[label] = {
                'median': float(np.median(samples[:, i])),
                'p16': float(np.percentile(samples[:, i], 16)),
                'p84': float(np.percentile(samples[:, i], 84)),
                'mean': float(np.mean(samples[:, i])),
                'std': float(np.std(samples[:, i])),
            }

        # Mass function and M2_min for each sample
        f_M_samples = np.array([mass_function(s[0], s[1]) for s in samples])
        M2_min_samples = np.array([min_companion_mass(f, self.M1) for f in f_M_samples])

        results['f_M'] = {
            'median': float(np.nanmedian(f_M_samples)),
            'p16': float(np.nanpercentile(f_M_samples, 16)),
            'p84': float(np.nanpercentile(f_M_samples, 84)),
        }

        valid_M2 = M2_min_samples[np.isfinite(M2_min_samples)]
        if len(valid_M2) > 0:
            results['M2_min'] = {
                'median': float(np.nanmedian(valid_M2)),
                'p16': float(np.nanpercentile(valid_M2, 16)),
                'p84': float(np.nanpercentile(valid_M2, 84)),
            }

            # Probabilities
            results['probabilities'] = {
                'P_M2_gt_1.4': float(np.sum(valid_M2 > 1.4) / len(valid_M2) * 100),
                'P_M2_gt_3.0': float(np.sum(valid_M2 > 3.0) / len(valid_M2) * 100),
                'P_M2_gt_5.0': float(np.sum(valid_M2 > 5.0) / len(valid_M2) * 100),
            }
        else:
            results['M2_min'] = {'median': np.nan, 'p16': np.nan, 'p84': np.nan}
            results['probabilities'] = {'P_M2_gt_1.4': 0, 'P_M2_gt_3.0': 0, 'P_M2_gt_5.0': 0}

        # Best fit
        best_idx = np.argmax(log_probs)
        results['best_fit'] = {
            'P': float(samples[best_idx, 0]),
            'K': float(samples[best_idx, 1]),
            'e': float(samples[best_idx, 2]),
            'omega': float(samples[best_idx, 3]),
            'T0': float(samples[best_idx, 4]),
            'gamma': float(samples[best_idx, 5]),
            'log_L': float(log_probs[best_idx]),
        }

        # Convergence check
        results['convergence'] = {
            'n_samples': len(samples),
            'acceptance_rate': float(len(np.unique(samples[:, 0])) / len(samples)),
        }

        return results, samples, M2_min_samples

    def create_plots(self, samples, M2_min_samples, output_prefix):
        """Create diagnostic plots."""
        try:
            # Corner plot (P, K, e only)
            import corner
            fig = corner.corner(
                samples[:, :3],
                labels=['P (days)', 'K (km/s)', 'e'],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
            )
            fig.savefig(f'{output_prefix}_corner.png', dpi=100)
            plt.close()
        except Exception as e:
            print(f"  Corner plot failed: {e}")

        # RV curve plot
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Time series
            best_idx = np.argmax(samples[:, 0] > 0)  # Just get any valid sample
            best = samples[np.argmax([self.log_probability(s) for s in samples[:100]])]

            t_grid = np.linspace(self.t_obs.min() - 10, self.t_obs.max() + 10, 200)

            # Plot a few posterior samples
            for i in np.random.choice(len(samples), min(50, len(samples)), replace=False):
                rv_model = rv_model_keplerian(t_grid, *samples[i])
                ax1.plot(t_grid, rv_model, 'b-', alpha=0.05, lw=0.5)

            # Plot data
            ax1.errorbar(self.t_obs, self.rv_obs, yerr=self.rv_err,
                        fmt='ko', markersize=8, capsize=3)
            ax1.set_xlabel('MJD')
            ax1.set_ylabel('RV (km/s)')
            ax1.set_title(f'Target {self.targetid}')
            ax1.grid(True, alpha=0.3)

            # M2_min histogram
            valid_M2 = M2_min_samples[np.isfinite(M2_min_samples)]
            if len(valid_M2) > 10:
                ax2.hist(valid_M2, bins=30, density=True, alpha=0.7, color='green')
                ax2.axvline(1.4, color='purple', linestyle='--', lw=2, label='NS')
                ax2.axvline(3.0, color='black', linestyle='--', lw=2, label='BH')
                ax2.axvline(np.median(valid_M2), color='red', linestyle='-', lw=2)
                ax2.set_xlabel('M2_min (M☉)')
                ax2.set_ylabel('Density')
                ax2.set_title(f'M2_min = {np.median(valid_M2):.2f} M☉')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_prefix}_rv.png', dpi=100)
            plt.close()
        except Exception as e:
            print(f"  RV plot failed: {e}")


def load_desi_rv_data():
    """Load DESI RV data and return lookup dictionary."""
    from astropy.io import fits

    print("Loading DESI RV data...")

    epochs_by_target = {}

    for path, program in [
        ('data/raw/rvpix_exp-main-bright.fits', 'bright'),
        ('data/raw/rvpix_exp-main-dark.fits', 'dark')
    ]:
        full_path = f'/home/primary/DESI-BH-CANDIDATE-SEARCH/{path}'
        if not os.path.exists(full_path):
            print(f"  Warning: {path} not found")
            continue

        print(f"  Loading {path}...")
        with fits.open(full_path) as hdu:
            rvtab = hdu['RVTAB'].data
            fibermap = hdu['FIBERMAP'].data

            # Quality filter
            mask = np.ones(len(rvtab), dtype=bool)
            mask &= np.isfinite(rvtab['VRAD'])
            mask &= np.isfinite(rvtab['VRAD_ERR'])
            mask &= rvtab['VRAD_ERR'] > 0
            mask &= rvtab['VRAD_ERR'] < 50
            mask &= np.abs(rvtab['VRAD']) < 500

            indices = np.where(mask)[0]

            for idx in indices:
                tid = int(rvtab['TARGETID'][idx])
                if tid not in epochs_by_target:
                    epochs_by_target[tid] = {'mjd': [], 'rv': [], 'rv_err': []}

                epochs_by_target[tid]['mjd'].append(float(fibermap['MJD'][idx]))
                epochs_by_target[tid]['rv'].append(float(rvtab['VRAD'][idx]))
                epochs_by_target[tid]['rv_err'].append(float(rvtab['VRAD_ERR'][idx]))

    print(f"  Loaded epochs for {len(epochs_by_target)} targets")
    return epochs_by_target


def process_candidate(args):
    """Process a single candidate (for parallel execution)."""
    candidate, epochs_by_target, rank = args

    targetid = candidate['targetid']

    print(f"\n[{rank}] Processing target {targetid}...")

    # Get RV data
    if targetid not in epochs_by_target:
        print(f"  No RV data found for {targetid}")
        return None

    data = epochs_by_target[targetid]

    # Sort by MJD
    sort_idx = np.argsort(data['mjd'])
    t_obs = np.array(data['mjd'])[sort_idx]
    rv_obs = np.array(data['rv'])[sort_idx]
    rv_err = np.array(data['rv_err'])[sort_idx]

    if len(t_obs) < 3:
        print(f"  Only {len(t_obs)} epochs, skipping")
        return None

    print(f"  {len(t_obs)} epochs, ΔRV = {np.ptp(rv_obs):.1f} km/s")

    # Get M1 estimate from candidate metadata
    M1 = candidate.get('metadata', {}).get('M1', 0.5)

    try:
        # Run MCMC
        mcmc = CandidateMCMC(targetid, t_obs, rv_obs, rv_err, candidate, M1)
        samples, log_probs = mcmc.run_mcmc(n_walkers=24, n_steps=2000, n_burn=400)

        # Analyze
        results, samples, M2_samples = mcmc.analyze_samples(samples, log_probs)

        # Add input data to results
        results['input'] = {
            'targetid': targetid,
            'n_epochs': len(t_obs),
            'mjd': t_obs.tolist(),
            'rv': rv_obs.tolist(),
            'rv_err': rv_err.tolist(),
            'delta_rv': float(np.ptp(rv_obs)),
            'baseline_days': float(np.ptp(t_obs)),
            'M1_assumed': M1,
        }

        # Add fast-screen comparison
        results['fast_screen'] = {
            'period': candidate.get('best_period'),
            'K': candidate.get('best_K'),
            'm2_min': candidate.get('m2_min'),
            'total_score': candidate.get('total_score'),
        }

        # Create plots
        output_prefix = os.path.join(OUTPUT_DIR, f'{targetid}')
        mcmc.create_plots(samples, M2_samples, output_prefix)

        # Save individual results
        with open(f'{output_prefix}_mcmc.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)

        print(f"  MCMC: P={results['P']['median']:.1f}d, K={results['K']['median']:.1f}km/s, M2_min={results['M2_min']['median']:.2f}M☉")

        return results

    except Exception as e:
        print(f"  MCMC failed: {e}")
        traceback.print_exc()
        return None


def generate_summary_report(all_results, candidates):
    """Generate summary markdown report."""

    report = []
    report.append("# E3_dwd_lisa MCMC Verification Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"\nTotal candidates analyzed: {len([r for r in all_results if r is not None])}/{len(candidates)}")
    report.append("")

    # Summary table
    report.append("## Summary Table")
    report.append("")
    report.append("| Rank | TargetID | N_epochs | ΔRV | P_mcmc | K_mcmc | M2_min | Pr(NS+) | Pr(BH) | Status |")
    report.append("|------|----------|----------|-----|--------|--------|--------|---------|--------|--------|")

    valid_results = []
    for i, (result, candidate) in enumerate(zip(all_results, candidates)):
        if result is None:
            report.append(f"| {i+1} | {candidate['targetid']} | - | - | - | - | - | - | - | FAILED |")
            continue

        valid_results.append((i, result, candidate))

        P_med = result['P']['median']
        K_med = result['K']['median']
        M2_med = result['M2_min']['median']
        pr_ns = result['probabilities']['P_M2_gt_1.4']
        pr_bh = result['probabilities']['P_M2_gt_3.0']

        # Classify
        if M2_med > 3.0 and pr_bh > 50:
            status = "**BH CANDIDATE**"
        elif M2_med > 1.4 and pr_ns > 50:
            status = "NS candidate"
        elif K_med > 200:
            status = "High-K (suspect)"
        else:
            status = "Review needed"

        report.append(f"| {i+1} | {result['input']['targetid']} | {result['input']['n_epochs']} | {result['input']['delta_rv']:.1f} | {P_med:.1f} | {K_med:.1f} | {M2_med:.2f} | {pr_ns:.0f}% | {pr_bh:.0f}% | {status} |")

    report.append("")

    # Top candidates section
    report.append("## Top Candidates (Pr(BH) > 30%)")
    report.append("")

    top_bh = [(i, r, c) for i, r, c in valid_results
              if r['probabilities']['P_M2_gt_3.0'] > 30 and r['K']['median'] < 200]

    if not top_bh:
        report.append("No candidates with Pr(BH) > 30% and reasonable K < 200 km/s.")
    else:
        for i, result, candidate in sorted(top_bh, key=lambda x: -x[1]['probabilities']['P_M2_gt_3.0']):
            report.append(f"### Target {result['input']['targetid']}")
            report.append("")
            report.append(f"- **MCMC Period**: {result['P']['median']:.2f} days (68% CI: {result['P']['p16']:.2f} - {result['P']['p84']:.2f})")
            report.append(f"- **MCMC K**: {result['K']['median']:.1f} km/s (68% CI: {result['K']['p16']:.1f} - {result['K']['p84']:.1f})")
            report.append(f"- **Eccentricity**: {result['e']['median']:.3f}")
            report.append(f"- **M2_min**: {result['M2_min']['median']:.2f} M☉ (68% CI: {result['M2_min']['p16']:.2f} - {result['M2_min']['p84']:.2f})")
            report.append(f"- **Pr(M2 > 1.4 M☉)**: {result['probabilities']['P_M2_gt_1.4']:.1f}%")
            report.append(f"- **Pr(M2 > 3.0 M☉)**: {result['probabilities']['P_M2_gt_3.0']:.1f}%")
            report.append("")
            report.append(f"Fast-screen comparison: P={candidate.get('best_period', 'N/A'):.1f}d, K={candidate.get('best_K', 'N/A'):.1f}km/s")
            report.append("")
            report.append(f"![RV Plot]({result['input']['targetid']}_rv.png)")
            report.append("")

    # Suspicious candidates
    report.append("## Suspicious Candidates (K > 200 km/s)")
    report.append("")
    report.append("These candidates have very high K values that may indicate period aliasing or systematic issues.")
    report.append("")

    suspicious = [(i, r, c) for i, r, c in valid_results if r['K']['median'] > 200]

    for i, result, candidate in suspicious[:5]:
        report.append(f"- **{result['input']['targetid']}**: K={result['K']['median']:.0f} km/s, P={result['P']['median']:.1f}d - needs verification")

    report.append("")

    # Statistics
    report.append("## Statistics")
    report.append("")

    if valid_results:
        K_values = [r['K']['median'] for _, r, _ in valid_results]
        M2_values = [r['M2_min']['median'] for _, r, _ in valid_results if not np.isnan(r['M2_min']['median'])]

        report.append(f"- Candidates with K < 100 km/s: {sum(1 for k in K_values if k < 100)}")
        report.append(f"- Candidates with K 100-200 km/s: {sum(1 for k in K_values if 100 <= k < 200)}")
        report.append(f"- Candidates with K > 200 km/s: {sum(1 for k in K_values if k >= 200)}")
        report.append("")
        report.append(f"- Candidates with M2_min > 3 M☉: {sum(1 for m in M2_values if m > 3)}")
        report.append(f"- Candidates with M2_min 1.4-3 M☉: {sum(1 for m in M2_values if 1.4 <= m <= 3)}")
        report.append(f"- Candidates with M2_min < 1.4 M☉: {sum(1 for m in M2_values if m < 1.4)}")

    report.append("")
    report.append("---")
    report.append("*Report generated by mcmc_e1_detailed.py*")

    return "\n".join(report)


def main():
    print("=" * 70)
    print("E3_dwd_lisa MCMC Verification")
    print("=" * 70)
    print()

    # Load E1 results
    e1_path = '/home/primary/DESI-BH-CANDIDATE-SEARCH/runs/real_run_20260115_183016/candidates/E3_dwd_lisa_results.json'

    print(f"Loading E1 results from {e1_path}")
    with open(e1_path) as f:
        e1_data = json.load(f)

    candidates = e1_data['candidates']
    print(f"Found {len(candidates)} candidates")

    # Load DESI data
    epochs_by_target = load_desi_rv_data()

    # Process candidates (top 20 by score for now to save time)
    # Can be adjusted to process all 50
    n_to_process = min(30, len(candidates))
    print(f"\nProcessing top {n_to_process} candidates by score...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []
    for i, candidate in enumerate(candidates[:n_to_process]):
        result = process_candidate((candidate, epochs_by_target, i+1))
        all_results.append(result)

    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(all_results, candidates[:n_to_process])

    report_path = os.path.join(OUTPUT_DIR, 'MCMC_VERIFICATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")

    # Summary stats
    successful = sum(1 for r in all_results if r is not None)
    print(f"\nCompleted: {successful}/{n_to_process} candidates successfully processed")

    print("\nDone!")


if __name__ == "__main__":
    main()
