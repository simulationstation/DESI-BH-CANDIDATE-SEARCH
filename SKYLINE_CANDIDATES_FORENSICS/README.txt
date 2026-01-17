DESI SKYLINE ANOMALY CANDIDATES - FORENSIC ANALYSIS
====================================================

This folder contains results from a search for unexplained monochromatic 
emission features in DESI spectroscopic data ("space laser" candidates).

FILES:
------
e9_laser_candidates_sparcl.csv  - 1455 raw candidates from SPARCL query
e9_priority_candidates.csv      - 89 priority candidates (filtered by SNR)
e9_forensic_results.csv         - 89 forensic verification results

FORENSIC METHOD:
----------------
Each candidate's spectral profile was analyzed to distinguish:
- COSMIC_RAY: Sharp needle spike (wing ratio < 0.1) - detector artifact
- REAL_SIGNAL: Resolved Gaussian (wing ratio > 0.2) - astrophysical emission
- AMBIGUOUS: Intermediate profile (0.1 < wing ratio < 0.2)

RESULTS SUMMARY:
----------------
Total analyzed: 89
  COSMIC_RAY:   13
  REAL_SIGNAL:  70 (mostly galaxies with expected emission lines)
  AMBIGUOUS:    6

STAR candidates (most interesting for anomaly search):
  Total: 13
  COSMIC_RAY: 10
  REAL_SIGNAL: 1
  AMBIGUOUS: 2

HIGH-PRIORITY CANDIDATES:
-------------------------
1. REAL_SIGNAL: targetid 39627997694657850
   Wavelength: 9502.4 Å | SNR: 46.2 | Wing ratio: 0.24
   RA: 132.788138 | Dec: +8.718857
   Gaia: G=16.7, BP-RP=0.86 (F/G star), d~2058 pc, RUWE=2.11
   Notes: Nearest known line is [SIII] 9532 Å but offset is -930 km/s (unusual)

2. AMBIGUOUS: targetid 39633497471584886 (TWO features from same star)
   Feature A: 6848.0 Å | SNR: 23.6 | Wing ratio: 0.15
   Feature B: 6143.2 Å | SNR: 22.2 | Wing ratio: 0.17
   RA: 119.573328 | Dec: +70.945221
   Gaia: G=17.4, BP-RP=1.58 (K/M star), d~1028 pc, RUWE=1.00
   Notes: Neither wavelength matches known atomic/molecular lines
