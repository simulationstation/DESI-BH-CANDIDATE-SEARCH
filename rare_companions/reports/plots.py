"""
Plot generator for candidates.
"""

import os
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PlotGenerator:
    """
    Generate diagnostic plots for candidates.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def rv_timeseries_plot(self, targetid: int, mjd: np.ndarray,
                          rv: np.ndarray, rv_err: np.ndarray,
                          best_period: float = None,
                          best_K: float = None,
                          gamma: float = None) -> Optional[str]:
        """
        Generate RV time series plot.

        Parameters
        ----------
        targetid : int
            Target identifier
        mjd : array
            MJD timestamps
        rv : array
            RV values (km/s)
        rv_err : array
            RV errors (km/s)
        best_period : float, optional
            Best-fit period for phase folding
        best_K : float, optional
            Best-fit semi-amplitude
        gamma : float, optional
            Systemic velocity

        Returns
        -------
        str
            Path to saved plot
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Time series
        ax1 = axes[0]
        ax1.errorbar(mjd, rv, yerr=rv_err, fmt='o', capsize=3,
                     color='C0', markersize=6)
        ax1.set_xlabel('MJD')
        ax1.set_ylabel('RV (km/s)')
        ax1.set_title(f'Target {targetid} - RV Time Series')

        # Phase folded (if period available)
        ax2 = axes[1]
        if best_period and best_period > 0:
            phase = ((mjd - mjd.min()) % best_period) / best_period
            ax2.errorbar(phase, rv, yerr=rv_err, fmt='o', capsize=3,
                         color='C0', markersize=6)

            # Plot model if K and gamma available
            if best_K and gamma is not None:
                phase_model = np.linspace(0, 1, 100)
                rv_model = gamma + best_K * np.sin(2 * np.pi * phase_model)
                ax2.plot(phase_model, rv_model, 'r-', alpha=0.7, label=f'P={best_period:.1f}d')
                ax2.legend()

            ax2.set_xlabel('Phase')
            ax2.set_ylabel('RV (km/s)')
            ax2.set_title(f'Phase Folded (P={best_period:.2f} d)')
        else:
            ax2.text(0.5, 0.5, 'No period available',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Phase Folded')

        plt.tight_layout()

        path = os.path.join(self.output_dir, f'rv_plot_{targetid}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def negative_space_summary(self, targetid: int,
                              sed_data: Dict = None,
                              photometry_data: Dict = None,
                              xray_radio_data: Dict = None,
                              imaging_data: Dict = None) -> Optional[str]:
        """
        Generate negative-space summary plot.

        Shows SED, photometric variability, X-ray/radio constraints,
        and imaging analysis in one figure.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # SED panel
        ax_sed = axes[0, 0]
        ax_sed.set_title('SED Constraints')
        if sed_data:
            # Plot available photometry
            bands = []
            mags = []
            if sed_data.get('j_mag'):
                bands.append('J')
                mags.append(sed_data['j_mag'])
            if sed_data.get('h_mag'):
                bands.append('H')
                mags.append(sed_data['h_mag'])
            if sed_data.get('k_mag'):
                bands.append('K')
                mags.append(sed_data['k_mag'])
            if sed_data.get('w1_mag'):
                bands.append('W1')
                mags.append(sed_data['w1_mag'])
            if sed_data.get('w2_mag'):
                bands.append('W2')
                mags.append(sed_data['w2_mag'])

            if bands:
                ax_sed.scatter(range(len(bands)), mags, s=100)
                ax_sed.set_xticks(range(len(bands)))
                ax_sed.set_xticklabels(bands)
                ax_sed.set_ylabel('Magnitude')
                ax_sed.invert_yaxis()

            # Note IR excess status
            ir_status = 'IR excess' if sed_data.get('ir_excess') else 'No IR excess'
            ax_sed.text(0.05, 0.95, ir_status, transform=ax_sed.transAxes,
                        va='top', fontsize=10)
        else:
            ax_sed.text(0.5, 0.5, 'No SED data', ha='center', va='center',
                        transform=ax_sed.transAxes)

        # Photometry panel
        ax_phot = axes[0, 1]
        ax_phot.set_title('Photometric Variability')
        if photometry_data:
            tess = photometry_data.get('tess', {})
            ztf = photometry_data.get('ztf', {})

            text_lines = []
            if tess.get('has_data'):
                text_lines.append(f"TESS: {tess.get('scatter_ppm', 0):.0f} ppm scatter")
                text_lines.append(f"  Flag: {tess.get('variability_flag', 'unknown')}")
            if ztf.get('has_data'):
                text_lines.append(f"ZTF: {ztf.get('scatter_g_mmag', 0):.1f} mmag (g)")
                text_lines.append(f"  Flag: {ztf.get('variability_flag', 'unknown')}")

            ax_phot.text(0.1, 0.9, '\n'.join(text_lines), transform=ax_phot.transAxes,
                         va='top', fontsize=10, family='monospace')
        else:
            ax_phot.text(0.5, 0.5, 'No photometry data', ha='center', va='center',
                         transform=ax_phot.transAxes)
        ax_phot.axis('off')

        # X-ray/Radio panel
        ax_xray = axes[1, 0]
        ax_xray.set_title('X-ray/Radio Constraints')
        if xray_radio_data:
            text_lines = []
            if xray_radio_data.get('has_xray_detection'):
                text_lines.append(f"X-ray: DETECTED ({xray_radio_data.get('xray_catalog', 'unknown')})")
            else:
                text_lines.append(f"X-ray: Non-detection")
                text_lines.append(f"  Upper limit: {xray_radio_data.get('xray_upper_limit', 0):.1e} erg/s")

            if xray_radio_data.get('has_radio_detection'):
                text_lines.append(f"Radio: DETECTED ({xray_radio_data.get('radio_catalog', 'unknown')})")
            else:
                text_lines.append(f"Radio: Non-detection")

            text_lines.append(f"Consistent with quiescent: {xray_radio_data.get('consistent_with_quiescent', 'unknown')}")

            ax_xray.text(0.1, 0.9, '\n'.join(text_lines), transform=ax_xray.transAxes,
                         va='top', fontsize=10, family='monospace')
        else:
            ax_xray.text(0.5, 0.5, 'No X-ray/radio data', ha='center', va='center',
                         transform=ax_xray.transAxes)
        ax_xray.axis('off')

        # Imaging panel
        ax_img = axes[1, 1]
        ax_img.set_title('Imaging Analysis')
        if imaging_data:
            text_lines = [
                f"Ellipticity: {imaging_data.get('ellipticity', 0):.3f}",
                f"Asymmetry: {imaging_data.get('asymmetry', 0):.3f}",
                f"Secondary peaks: {imaging_data.get('n_secondary_peaks', 0)}",
                f"Blend score: {imaging_data.get('blend_score', 0):.2f}",
                f"Blend flag: {imaging_data.get('blend_flag', 'unknown')}"
            ]
            ax_img.text(0.1, 0.9, '\n'.join(text_lines), transform=ax_img.transAxes,
                        va='top', fontsize=10, family='monospace')
        else:
            ax_img.text(0.5, 0.5, 'No imaging data', ha='center', va='center',
                        transform=ax_img.transAxes)
        ax_img.axis('off')

        plt.suptitle(f'Target {targetid} - Negative Space Summary', fontsize=14)
        plt.tight_layout()

        path = os.path.join(self.output_dir, f'negative_space_{targetid}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path

    def gaia_astrometry_summary(self, targetid: int,
                               gaia_data: Dict) -> Optional[str]:
        """Generate Gaia astrometry summary plot."""
        if not HAS_MATPLOTLIB or not gaia_data:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        # Create text summary
        lines = [
            f"Source ID: {gaia_data.get('source_id', 'N/A')}",
            f"",
            f"Parallax: {gaia_data.get('parallax', 0):.3f} ± {gaia_data.get('parallax_err', 0):.3f} mas",
            f"Distance: {gaia_data.get('distance_pc', 'N/A')} pc",
            f"",
            f"RUWE: {gaia_data.get('ruwe', 0):.3f}",
            f"AEN: {gaia_data.get('aen', 0):.3f} mas (sig: {gaia_data.get('aen_sig', 0):.1f})",
            f"",
            f"IPD frac multi-peak: {gaia_data.get('ipd_frac_multi_peak', 0):.1f}%",
            f"IPD harmonic amp: {gaia_data.get('ipd_gof_harmonic_amp', 0):.3f}",
            f"",
            f"G mag: {gaia_data.get('phot_g_mean_mag', 0):.2f}",
            f"BP-RP: {gaia_data.get('bp_rp', 0):.2f}",
            f"",
            f"Astrometric binary flag: {gaia_data.get('is_astrometric_binary', False)}",
            f"Blend flag: {gaia_data.get('is_blended', False)}"
        ]

        ax.text(0.1, 0.95, '\n'.join(lines), transform=ax.transAxes,
                va='top', fontsize=11, family='monospace')
        ax.axis('off')
        ax.set_title(f'Target {targetid} - Gaia Astrometry Summary')

        path = os.path.join(self.output_dir, f'gaia_summary_{targetid}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        return path
