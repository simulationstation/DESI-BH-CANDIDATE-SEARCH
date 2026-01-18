#!/usr/bin/env python3
"""Create blink animation from PS1 images."""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.io import fits
from astropy.visualization import ZScaleInterval

OUTPUT_DIR = "/home/primary/DESI-BH-CANDIDATE-SEARCH/wobble_imaging_3802130935635096832"
WOBBLE_FIGURES = "/home/primary/DESI-BH-CANDIDATE-SEARCH/WOBBLE_FIGURES"


def main():
    # Load metadata
    meta_file = os.path.join(OUTPUT_DIR, 'data', 'ps1_metadata.json')
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    # Sort by MJD
    metadata.sort(key=lambda x: x.get('mjd', 0) or 0)

    # Load images
    images = []
    dates = []
    filters_list = []

    for meta in metadata[:20]:  # Limit to 20 for animation
        filepath = meta.get('local')
        if not filepath or not os.path.exists(filepath):
            continue

        try:
            with fits.open(filepath) as hdu:
                data = hdu[0].data
                if data is None:
                    continue
                # Handle NaN
                data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
                images.append(data)
                dates.append(meta.get('date', f"MJD {meta.get('mjd', 0):.1f}"))
                filters_list.append(meta.get('filter', 'unknown').split('.')[0])
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    if len(images) < 2:
        print("Not enough images for animation")
        return

    print(f"[BLINK] Creating animation with {len(images)} frames")

    # Normalize all images to common scale
    zscale = ZScaleInterval()
    normalized = []
    for img in images:
        try:
            vmin, vmax = zscale.get_limits(img)
            norm_img = (img - vmin) / (vmax - vmin + 1e-10)
            norm_img = np.clip(norm_img, 0, 1)
            normalized.append(norm_img)
        except:
            normalized.append(img / (np.max(img) + 1e-10))

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(normalized[0], origin='lower', cmap='gray', vmin=0, vmax=1)

    # Mark center
    ny, nx = normalized[0].shape
    center_marker, = ax.plot(nx/2, ny/2, 'r+', markersize=15, markeredgewidth=2)

    title = ax.set_title(f'Frame 1: {dates[0]} ({filters_list[0]})', fontsize=12)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    def update(frame):
        im.set_data(normalized[frame])
        title.set_text(f'Frame {frame+1}: {dates[frame]} ({filters_list[frame]})')
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(normalized),
                                   interval=500, blit=False)

    # Save as GIF
    gif_path = os.path.join(WOBBLE_FIGURES, 'blink_animation.gif')
    try:
        ani.save(gif_path, writer='pillow', fps=2)
        print(f"[BLINK] Saved animation to {gif_path}")
    except Exception as e:
        print(f"[BLINK] Could not save GIF: {e}")
        # Save as static multi-panel instead
        create_static_panels(normalized, dates, filters_list)

    plt.close(fig)

    # Also create a static comparison figure
    create_static_panels(normalized, dates, filters_list)


def create_static_panels(normalized, dates, filters_list):
    """Create static multi-panel figure."""
    n_panels = min(9, len(normalized))
    nrows = 3
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < n_panels:
            ax.imshow(normalized[i], origin='lower', cmap='gray', vmin=0, vmax=1)
            # Mark center
            ny, nx = normalized[i].shape
            ax.plot(nx/2, ny/2, 'r+', markersize=10, markeredgewidth=1.5)
            ax.set_title(f'{dates[i][:10]}\n({filters_list[i]})', fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Gaia DR3 3802130935635096832 - Multi-Epoch PS1 Images', fontsize=14)
    plt.tight_layout()

    panel_path = os.path.join(WOBBLE_FIGURES, 'multi_epoch_panels.png')
    fig.savefig(panel_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[BLINK] Saved multi-epoch panels to {panel_path}")


if __name__ == "__main__":
    main()
