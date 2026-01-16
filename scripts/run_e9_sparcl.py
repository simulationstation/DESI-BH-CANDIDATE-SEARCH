#!/usr/bin/env python3
"""
E9 Laser Search using SPARCL.

Search DESI coadded spectra for anomalous monochromatic emission.
"""

import numpy as np
import pandas as pd
from sparcl.client import SparclClient
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class AnomalyCandidate:
    sparcl_id: str
    targetid: int
    ra: float
    dec: float
    wavelength_peak: float
    flux_peak: float
    snr: float
    fwhm_angstrom: float
    spectype: str
    is_airglow: bool
    notes: str

AIRGLOW_LINES = {
    5577.34: 'OI_5577', 5889.95: 'NaD1', 5895.92: 'NaD2',
    6300.30: 'OI_6300', 6363.78: 'OI_6364', 6533.04: 'OI_6533',
    6863.96: 'OH', 7316.29: 'OH', 7340.89: 'OH', 7571.75: 'OI_7774',
    7750.64: 'OH', 7794.11: 'OH', 7913.71: 'OH', 7993.33: 'OH',
    8344.60: 'OH', 8399.17: 'OH', 8430.17: 'OH', 8827.10: 'OH', 8885.85: 'OH',
}

def is_near_airglow(wavelength: float, tolerance: float = 3.0) -> Tuple[bool, str]:
    for line_wave, line_name in AIRGLOW_LINES.items():
        if abs(wavelength - line_wave) < tolerance:
            return True, line_name
    return False, ""

def compute_residuals(flux: np.ndarray, ivar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    continuum = gaussian_filter1d(flux, sigma=50)
    residuals = flux - continuum
    block_size = 200
    noise = np.zeros_like(flux)
    for i in range(0, len(flux), block_size):
        block = residuals[i:i+block_size]
        mad = np.median(np.abs(block - np.median(block))) * 1.4826
        noise[i:i+block_size] = max(mad, 0.01)
    ivar_sigma = np.where(ivar > 0, 1.0/np.sqrt(ivar), np.inf)
    total_sigma = np.sqrt(noise**2 + ivar_sigma**2)
    significance = residuals / total_sigma
    return residuals, significance

def find_peaks(wavelength: np.ndarray, flux: np.ndarray, significance: np.ndarray,
               threshold: float = 8.0, max_fwhm_ang: float = 5.0) -> List[dict]:
    peaks = []
    above = significance > threshold
    i = 0
    while i < len(above):
        if above[i]:
            start = i
            while i < len(above) and above[i]:
                i += 1
            end = i
            peak_idx = start + np.argmax(significance[start:end])
            peak_wave = wavelength[peak_idx]
            peak_flux = flux[peak_idx]
            peak_snr = significance[peak_idx]
            half_max = flux[peak_idx] / 2
            left = peak_idx
            while left > 0 and flux[left] > half_max:
                left -= 1
            right = peak_idx
            while right < len(flux)-1 and flux[right] > half_max:
                right += 1
            fwhm_pix = right - left
            dlambda = np.median(np.diff(wavelength))
            fwhm_ang = fwhm_pix * dlambda
            if fwhm_ang < max_fwhm_ang:
                peaks.append({
                    'wavelength': peak_wave, 'flux': peak_flux,
                    'snr': peak_snr, 'fwhm_ang': fwhm_ang
                })
        i += 1
    return peaks

def search_spectrum(sparcl_id: str, spec: dict) -> List[AnomalyCandidate]:
    candidates = []
    wavelength = np.array(spec['wavelength'])
    flux = np.array(spec['flux'])
    ivar = np.array(spec['ivar'])
    mask = np.array(spec['mask']) if 'mask' in spec else np.zeros_like(flux)
    good = (mask == 0) & np.isfinite(flux) & (ivar > 0)
    if good.sum() < 1000:
        return candidates
    residuals, significance = compute_residuals(flux, ivar)
    peaks = find_peaks(wavelength, flux, significance, threshold=8.0, max_fwhm_ang=5.0)
    for peak in peaks:
        is_ag, ag_name = is_near_airglow(peak['wavelength'])
        candidates.append(AnomalyCandidate(
            sparcl_id=sparcl_id, targetid=spec.get('targetid', 0),
            ra=spec.get('ra', 0), dec=spec.get('dec', 0),
            wavelength_peak=peak['wavelength'], flux_peak=peak['flux'],
            snr=peak['snr'], fwhm_angstrom=peak['fwhm_ang'],
            spectype=spec.get('spectype', 'UNKNOWN'), is_airglow=is_ag,
            notes=f"Near {ag_name}" if is_ag else "CANDIDATE"
        ))
    return candidates

def main():
    print("=" * 60)
    print("E9 LASER SEARCH (SPARCL VERSION)")
    print("=" * 60)

    client = SparclClient()
    print("\nConnected to SPARCL")

    all_candidates = []
    processed = 0

    # Search different object types to get variety
    for spectype in ['GALAXY', 'STAR', 'QSO']:
        print(f"\n--- Searching {spectype} spectra ---")

        try:
            result = client.find(
                outfields=['sparcl_id', 'ra', 'dec', 'spectype', 'targetid'],
                constraints={'data_release': ['DESI-EDR'], 'spectype': [spectype]},
                limit=3000
            )
            print(f"Found {len(result.records)} {spectype} spectra")
        except Exception as e:
            print(f"Error finding {spectype}: {e}")
            continue

        if not result.records:
            continue

        # Process in batches
        batch_size = 100
        for i in range(0, len(result.records), batch_size):
            batch_records = result.records[i:i+batch_size]
            ids = [rec['sparcl_id'] for rec in batch_records]
            metadata = {rec['sparcl_id']: rec for rec in batch_records}

            try:
                spectra = client.retrieve(
                    uuid_list=ids,
                    include=['sparcl_id', 'flux', 'wavelength', 'ivar', 'mask']
                )
            except Exception as e:
                print(f"  Retrieve error: {e}")
                continue

            batch_cands = []
            for spec in spectra.records:
                sid = spec['sparcl_id']
                spec.update(metadata.get(sid, {}))
                cands = search_spectrum(sid, spec)
                batch_cands.extend(cands)

            non_airglow = [c for c in batch_cands if not c.is_airglow]
            all_candidates.extend(non_airglow)
            processed += len(spectra.records)

            if i % 500 == 0:
                print(f"  Processed {i+len(batch_records)}/{len(result.records)}, "
                      f"found {len(non_airglow)} candidates")

    print("\n" + "=" * 60)
    print(f"SEARCH COMPLETE")
    print(f"Processed: {processed} spectra")
    print(f"Candidates: {len(all_candidates)}")
    print("=" * 60)

    if all_candidates:
        df = pd.DataFrame([vars(c) for c in all_candidates])
        outfile = 'data/e9_laser_candidates_sparcl.csv'
        os.makedirs('data', exist_ok=True)
        df.to_csv(outfile, index=False)
        print(f"\nResults saved to {outfile}")

        print("\nTop 20 candidates by SNR:")
        df_sorted = df.sort_values('snr', ascending=False).head(20)
        for _, row in df_sorted.iterrows():
            print(f"  λ={row['wavelength_peak']:.1f}Å SNR={row['snr']:.1f} "
                  f"FWHM={row['fwhm_angstrom']:.2f}Å {row['spectype']}")
    else:
        print("\nNo candidates found.")

if __name__ == '__main__':
    main()
