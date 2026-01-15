# DESI DR1 Radial Velocity Variability Search for Unseen Companions

A conservative, reproducible search for statistically significant radial-velocity (RV) variability in public DESI Data Release 1 Milky Way Survey (MWS) data. Using only per-epoch RV measurements, we identify stars whose RV variability exceeds measurement noise and remains robust under leave-one-out tests. To distinguish potential non-interacting compact companions (Black Holes, Neutron Stars, White Dwarfs) from mundane binaries, we implement a **"Negative Space" multi-messenger validation pipeline**.

**Result: Identification of Gaia DR3 3802130935635096832**, a high-priority candidate system displaying large radial-velocity variations (ΔRV ≈ 146 km/s) and significant astrometric wobble (RUWE = 1.95), yet exhibiting no infrared excess, photometric variability, or ultraviolet emission. These properties are consistent with a massive, invisible companion.

---

## Motivation

Quiet compact companions (white dwarfs, neutron stars, black holes) are expected to be numerous in the Milky Way, yet difficult to identify when not accreting or otherwise luminous. Radial-velocity monitoring offers a gravity-only detection method, and DESI DR1 provides per-epoch RV measurements for millions of stars as part of the Milky Way Survey.

---

## Data

We use public DESI DR1 MWS per-epoch RV products from `main-bright` and `main-dark` programs. For each epoch we extract:

- Heliocentric radial velocity (RV)
- RV uncertainty (σ_RV)
- Observation time (MJD)
- DESI target identifier
- Gaia SOURCE_ID (when available)

---

## Selection Method

### Per-Epoch Quality Cuts

1. Finite RV and σ_RV
2. σ_RV < 10 km/s
3. |RV| < 500 km/s

### RV Variability Metric

```
ΔRV_max = max(RV) - min(RV)

S = ΔRV_max / sqrt(Σ σ_RV,i²)
```

### Robustness Diagnostics

To guard against single-epoch artifacts, we compute a leave-one-out minimum significance (S_min,LOO). For targets with N ≥ 3 epochs:

```
S_robust = min(S, S_min,LOO)
```

We restrict our sample to targets with N ≥ 3 epochs and select those with S_robust ≥ 10.

---

## Multi-Messenger Validation: The "Negative Space" Pipeline

To isolate potential compact objects from the initial RV-variable shortlist, we applied a secondary validation pipeline designed to identify systems with **strong gravity** but **missing light**. A candidate is considered a high-confidence dark companion only if it satisfies:

1. **Significant Gravity (Gaia DR3):** Astrometric wobble indicative of an orbit, defined as RUWE > 1.4
2. **No Infrared Excess (WISE):** To rule out M-dwarf companions, we require W1 - W2 < 0.1
3. **Photometric Silence (TESS/ZTF):** Time-domain photometry rules out deep eclipses or contact binary features
4. **Ultraviolet Silence (GALEX):** UV imaging rules out hot, young white dwarfs
5. **Clean Source Isolation (Legacy Survey):** Deep imaging residuals ensure astrometric signal is not due to contamination

---

## Results

The initial search yielded 21 candidate systems. Following the validation pipeline, one target emerged as a high-probability dark companion candidate.

### Follow-up Candidate Systems (Top 10)

| Rank | TargetID | Gaia Source ID | N | S_robust | RUWE | Verdict |
|------|----------|----------------|---|----------|------|---------|
| 1 | 39627745210139276 | 3802130935635096832 | 4 | 100.0 | 1.95 | **Dark Companion** |
| 2 | 39628001431785529 | 2759088365339967488 | 4 | 91.1 | 1.02 | Binary |
| 3 | 39632991214896712 | 1480681355298504960 | 3 | 81.1 | 0.98 | Binary |
| 4 | 39633437979575384 | 1584997005586641280 | 3 | 71.7 | 0.94 | Binary |
| 5 | 39627714713356667 | 3826086648304166400 | 4 | 55.4 | 1.10 | Binary |
| 6 | 39627830035744797 | 3891388499304470656 | 3 | 52.2 | 1.05 | Binary |
| 7 | 39627681263782079 | 6914501041337922944 | 3 | 49.1 | 0.89 | Binary |
| 8 | 39633025553665365 | 1375654252266254080 | 3 | 44.8 | 3.20 | Likely Binary |
| 9 | 39627720727987709 | 3827093418703158272 | 7 | 43.8 | 1.15 | Binary |
| 10 | 39627782317149427 | 3652971286995183488 | 5 | 42.7 | 1.20 | Binary |

---

## Top Dark Companion Candidate: Gaia DR3 3802130935635096832

This system represents the most significant detection in our sample. It exhibits violent radial velocity variations (ΔRV ≈ 146 km/s) over a baseline of 39 days.

### The "Money Plot": Gravity vs Silence

![Money Plot - RV vs TESS](money_plot.png)

*Left: High radial velocity amplitude (146 km/s) indicates a massive unseen companion yanking the visible star. Right: Completely flat TESS light curve over 6 years shows no eclipses or ellipsoidal variations - the companion emits no detectable light.*

### Physical Parameters

| Property | Value | Interpretation |
|----------|-------|----------------|
| **RA, Dec** | 164.5235, -1.6602 | Hydra constellation |
| **RUWE** | 1.95 | Strong astrometric wobble (>1.4 threshold) |
| **Astrometric Excess Noise** | 16.5σ | Highly significant orbital motion |
| **RV Amplitude** | 146.1 km/s | Large velocity variations |
| **W1-W2 Color** | 0.052 | No infrared excess (dark companion) |
| **SIMBAD** | NO MATCH | Unknown object |

### Visual Validation

The source appears isolated in both standard DSS imaging and deep Legacy Survey imaging, confirming the high RUWE is intrinsic to binary motion and not due to source blending.

![Aladin Sky View](dot.png)

*Aladin Sky Atlas (DSS) view. The source appears isolated.*

![Legacy Survey View](dot2.png)

*Legacy Survey Viewer (DECaLS) deep imaging. The source remains a clean, single point source.*

### Ultraviolet Constraints

To constrain the nature of the unseen companion, we inspected GALEX Near-Ultraviolet (NUV) imaging. A hot white dwarf companion (T_eff ≳ 10,000 K) would exhibit significant UV excess. The target is **undetected in the ultraviolet**, ruling out a hot, young white dwarf companion.

![GALEX NUV](nodot3.png)

*GALEX NUV imaging of the target field. The target is undetected in the ultraviolet.*

### TESS Photometry Analysis

We analyzed 6 sectors of TESS photometry spanning 2,200 days (6 years) to search for eclipses or ellipsoidal variations.

| Parameter | Value |
|-----------|-------|
| TESS Sectors | 6 (S9, S45, S46, S62, S72, S89) |
| Total Data Points | 37,832 |
| Time Baseline | 2,200 days |
| Light Curve Scatter | 6.32 ppt |
| Periodic Signal | **NONE DETECTED** |

![TESS Analysis](tess_analysis_result.png)

*Left: DESI Radial Velocity measurements showing significant variability. Right: TESS photometry showing a lack of eclipses or ellipsoidal modulation, consistent with a non-interacting dark companion.*

---

## Discussion

The discovery of **Gaia DR3 3802130935635096832** highlights the power of combining spectroscopic RVs with astrometric, photometric, and ultraviolet validation. Since the companion is optically and structurally dark (no IR excess, no eclipses), we can constrain its nature based on the lack of tidal interaction and UV emission.

Assuming a standard K-dwarf primary (~0.7 M☉), the high velocity semi-amplitude (K ≈ 73 km/s) requires a massive companion.

The **absence of GALEX UV emission** critically rules out the most common false positive for high-mass companions: the hot white dwarf. This restricts the candidate nature to either:
- An evolved, cold white dwarf
- A neutron star
- A stellar-mass black hole

**Conclusion: This system requires immediate spectroscopic monitoring to determine the orbital period and dynamic mass of the invisible companion.**

---

## Usage

### 1. Initial RV Candidate Analysis

```bash
python analyze_rv_candidates.py --data-root data --max-rows 1000000
```

### 2. Robust Triage with Leave-One-Out

```bash
python triage_rv_candidates.py --data-root data
```

### 3. Cross-match with Gaia NSS and SIMBAD

```bash
python crossmatch_nss_simbad.py
```

### 4. Build Priority Packet

```bash
python build_priority_packet.py
```

### 5. Multi-wavelength Validation

```bash
python verify_candidates.py
```

### 6. TESS Photometry Analysis

```bash
python analyze_tess_photometry.py
```

---

## Dependencies

```
numpy
fitsio (or astropy)
matplotlib
astroquery
lightkurve
pandas
```

---

## Data Requirements

DESI DR1 MWS per-epoch RV files:
- `rvpix_exp-main-bright.fits`
- `rvpix_exp-main-dark.fits`

Download from: https://data.desi.lbl.gov/public/dr1/

---

## Notes

- Gaia DR3 NSS incompleteness (particularly for short/intermediate periods) means absence of NSS classification does not indicate singleness
- Overlap with known variable classes is handled through annotation rather than removal
- Further observations required to determine orbital parameters or companion masses
- **Spectroscopic follow-up recommended** for the top candidate to obtain orbital solution

---

## Author

Aiden Smith (A.I Sloperator)

---

## License

For use with publicly released DESI data. See DESI data policies for usage terms.
