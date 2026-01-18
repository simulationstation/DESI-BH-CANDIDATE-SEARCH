# Wobble Analysis V2 - Systematic-Aware Reanalysis

## What the Blink Animation Is Actually Showing

The multi-epoch PS1 blink visualization shows apparent centroid wander at the tens-of-mas level.
This is **consistent with seeing, filter, and blend systematics** for a G≈17 source, and
**does not constrain Gaia-scale (~mas) astrometric perturbations**.

### Key Finding: The Animation Shows Systematics, Not Orbital Wobble

The dominant drivers of apparent motion in the animation are:
1. **Filter-dependent centroid shifts** - Different filters have systematic offsets
2. **Seeing/PSF variations** - Blending with the 0.69" neighbor causes seeing-dependent centroids
3. **Normal PS1 astrometric noise** - ~100 mas precision for G=17.3 sources

### Per-Filter Statistics

| Filter | N epochs | X RMS (mas) | Y RMS (mas) | Total RMS (mas) |
|--------|----------|-------------|-------------|-----------------|
| g | 10 | 11.7 | 21.2 | 24.2 |
| i | 22 | 17.9 | 21.4 | 27.9 |

### Neighbor-Axis Projection Test

The Gaia-resolved neighbor lies at PA = -13.5° and separation = 0.69".

| Direction | RMS (mas) |
|-----------|-----------|
| Along neighbor axis | 32.6 |
| Orthogonal to neighbor | 18.2 |

**Interpretation:** Motion preferentially along neighbor axis

⚠️ **The motion is preferentially along the neighbor direction**, suggesting blend/seeing-driven
centroid bias rather than orbital wobble. When seeing worsens, the PSF wings of the neighbor
contaminate the target centroid, pulling it toward the neighbor.

### Conservative Paper-Ready Interpretation

> "The multi-epoch PS1 blink visualization shows apparent centroid wander at the tens-of-mas
> level, consistent with seeing/filter/blend systematics for a G≈17 source, and does not
> constrain Gaia-scale (≈mas) astrometric perturbations; AO imaging is required for direct
> wobble detection."

### Suggested Figure Caption

> **Figure X.** Multi-epoch Pan-STARRS1 imaging of Gaia DR3 3802130935635096832. The animation
> shows centroid wander at ~37 mas RMS, consistent with normal PS1 astrometric precision and
> filter/seeing-dependent systematics given the 0.69" Gaia-resolved neighbor. This level of
> scatter cannot constrain the ~0.9 mas astrometric excess noise detected by Gaia.

### Files Generated

- `blink_g_only.gif` - g-band only animation
- `blink_i_only.gif` - i-band only animation
- `blink_MIXED_FILTERS_systematics_expected.gif` - Mixed filters with warning
- `frames/<filter>/frame_###.png` - Individual frames
- `neighbor_axis_projection.png` - Projection analysis
- `filter_comparison.png` - Per-filter statistics
