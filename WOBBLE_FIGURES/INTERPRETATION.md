# Wobble Analysis V3 - Same-Night Consistency Filtering

## Methodology

This analysis applies the paper's "same-night consistency" principle to PS1 astrometry:
observations taken on the same night should give consistent centroid measurements.
Data failing this test is flagged as problematic.

### Same-Night Consistency Test

1. Group observations by integer MJD (night) and filter
2. For groups with 2+ observations, identify outliers (>3σ from median in pixel space)
3. Calculate intra-night scatter for each group
4. Groups with scatter >50 mas are flagged as inconsistent

### Results

| Category | Count | Description |
|----------|-------|-------------|
| Good | 33 | Consistent same-night pairs |
| Bad | 5 | Outliers + inconsistent pairs |
| Total | 38 | All observations with centroids |

### Per-Filter Breakdown (Good Data)

| Filter | N epochs |
|--------|----------|
| g | 10 |
| i | 23 |

### RMS of Good Data

| Direction | RMS (mas) |
|-----------|-----------|
| RA (X) | 18.5 |
| Dec (Y) | 33.7 |
| Total | 38.4 |

### Files Generated

- `wobble_timeseries.gif` - Animated centroid track over 4.8 years (33 good obs)
- `wobble_star.gif` - Star marker animation with all positions shown
- `same_night_consistency.png` - Same-night consistency test results
- `same_night_good_data.png` - Good data centroid positions by filter
