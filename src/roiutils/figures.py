"""Visualization routines for ROI overlays on MNI template brains."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Patch

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

    template = datasets.load_mni152_template(resolution=config.template_resolution)

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
    discrete_cmap = cm.get_cmap(cmap, max(n_rois, 1))
    legend_handles = _build_legend_handles(atlas, selection, discrete_cmap)

    if config.figure_style == "quadrants":
        figure = _plot_quadrants(
            roi_resampled,
            template,
            config,
            cmap,
            legend_handles,
        )
        return _finalize_figure(figure, config)

    if config.figure_style == "xyz_strips":
        figure = _plot_xyz_strips(
            roi_resampled,
            template,
            config,
            cmap,
        )
        if config.show_legend and legend_handles:
            figure.legend(
                handles=legend_handles,
                loc=config.legend_loc,
                ncol=max(1, config.legend_ncol),
                frameon=True,
                fontsize=8,
            )
        return _finalize_figure(figure, config)

    display = plotting.plot_roi(
        roi_resampled,
        bg_img=template,
        display_mode=config.display_mode,
        cut_coords=config.cut_coords,
        cmap=cmap,
        alpha=config.alpha,
        title=config.title,
        colorbar=False,
        annotate=not config.clean,
        draw_cross=not config.clean,
    )

    if config.show_legend and legend_handles:
        figure = display.frame_axes.figure
        figure.legend(
            handles=legend_handles,
            loc=config.legend_loc,
            ncol=max(1, config.legend_ncol),
            frameon=True,
            fontsize=8,
        )

    if config.output_path is not None:
        out = Path(config.output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        display.savefig(str(out), dpi=config.dpi)
        display.close()
        return out

    return None


def _build_legend_handles(
    atlas: AtlasSpec,
    selection: RoiSelection,
    discrete_cmap,
) -> list[Patch]:
    labels_by_id = dict(atlas.labels_by_id)
    labels_by_id.update(selection.labels_by_id)
    handles: list[Patch] = []
    for index, roi_id in enumerate(selection.ids):
        color = discrete_cmap(index)
        label = labels_by_id.get(roi_id, f"ROI {roi_id}")
        handles.append(Patch(facecolor=color, edgecolor="black", label=f"{roi_id}: {label}"))
    return handles


def _plot_quadrants(roi_img, template, config: RenderConfig, cmap: str, legend_handles: list[Patch]):
    from nilearn import plotting

    slices = list(config.quadrant_slices or (("x", -10), ("y", 20), ("z", 8)))
    if len(slices) != 3:
        raise ValueError("quadrant_slices must contain exactly three (axis, coord) pairs.")

    figure = Figure(figsize=(12, 10), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    flat_axes = axes.flatten()

    for axis_obj, (axis_name, coord) in zip(flat_axes[:3], slices):
        display = plotting.plot_roi(
            roi_img,
            bg_img=template,
            display_mode=axis_name,
            cut_coords=[coord],
            cmap=cmap,
            alpha=config.alpha,
            colorbar=False,
            axes=axis_obj,
            annotate=not config.clean,
            draw_cross=not config.clean,
        )
        display.close()

    legend_ax = flat_axes[3]
    legend_ax.set_facecolor("white")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    if config.title:
        figure.suptitle(config.title)
    if config.show_legend and legend_handles:
        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=max(1, config.legend_ncol),
            frameon=True,
            fontsize=9,
        )

    figure.tight_layout()
    return figure


def _plot_xyz_strips(roi_img, template, config: RenderConfig, cmap: str):
    from nilearn import plotting

    strip_cut_coords = dict(config.strip_cut_coords or {})
    figure = Figure(figsize=(14, 12), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)

    for axis_obj, axis_name in zip(axes, ("z", "x", "y")):
        axis_cuts = strip_cut_coords.get(axis_name, config.cut_coords)
        display = plotting.plot_roi(
            roi_img,
            bg_img=template,
            display_mode=axis_name,
            cut_coords=axis_cuts,
            cmap=cmap,
            alpha=config.alpha,
            colorbar=False,
            axes=axis_obj,
            annotate=not config.clean,
            draw_cross=False,
        )
        display.close()

    if config.title:
        figure.suptitle(config.title)
    figure.tight_layout()
    return figure


def _finalize_figure(figure, config: RenderConfig) -> Path | None:
    if config.output_path is not None:
        out = Path(config.output_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(out), dpi=config.dpi, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return out

    figure.show()
    return None

