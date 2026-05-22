"""Visualization routines for ROI overlays on MNI template brains."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .models import AtlasSpec, RenderConfig, RoiSelection


def plot_roi_overlay(
    atlas: AtlasSpec,
    selection: RoiSelection,
    config: RenderConfig | None = None,
) -> Path | None:
    """Render selected ROIs as a colorized overlay on an MNI152 template brain.

    Each selected ROI is assigned a distinct color from the chosen colormap.
    The figure is saved to config.output_path if provided, otherwise shown
    interactively.

    Returns the saved output path, or None if displayed interactively.
    """
    from nilearn import datasets, plotting
    from nilearn.image import resample_to_img

    config = config or RenderConfig()

    # MNI152 background template (2 mm isotropic)
    template = datasets.load_mni152_template(resolution=2)

    # Build a labeled image: remap selected IDs to sequential 1..N so the
    # chosen colormap produces evenly distributed, distinct colors regardless
    # of the raw atlas ID values (which may be sparse, e.g. 3, 47, 112).
    atlas_data = np.rint(atlas.image.get_fdata()).astype(np.int32)
    labeled = np.zeros_like(atlas_data, dtype=np.int32)
    for rank, roi_id in enumerate(selection.ids, start=1):
        labeled[atlas_data == roi_id] = rank

    labeled_img = nib.Nifti1Image(labeled, affine=atlas.image.affine)

    # Resample to template space using nearest-neighbor to preserve integer labels.
    roi_resampled = resample_to_img(labeled_img, template, interpolation="nearest")

    n_rois = len(selection.ids)
    cmap = config.cmap or ("tab10" if n_rois <= 10 else "tab20")

    display = plotting.plot_roi(
        roi_resampled,
        bg_img=template,
        display_mode=config.display_mode,
        cut_coords=config.cut_coords,
        cmap=cmap,
        alpha=config.alpha,
        title=config.title,
        colorbar=False,
    )

    if config.output_path is not None:
        out = Path(config.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        display.savefig(str(out), dpi=config.dpi)
        display.close()
        return out

    return None

