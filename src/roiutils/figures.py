"""Visualization routines for ROI overlays on MNI template brains."""

from __future__ import annotations

from pathlib import Path
import warnings

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
    _validate_render_config(config)

    template = datasets.load_mni152_template(resolution=config.template_resolution)

    # Build a labeled image: remap selected IDs to sequential 1..N so the
    # chosen colormap produces evenly distributed, distinct colors regardless
    # of the raw atlas ID values (which may be sparse, e.g. 3, 47, 112).
    atlas_data = np.rint(atlas.image.get_fdata()).astype(np.int32)
    missing_ids = [roi_id for roi_id in selection.ids if not np.any(atlas_data == roi_id)]
    if missing_ids:
        warnings.warn(
            "Selected ROI IDs not found in atlas image: "
            + ", ".join(str(x) for x in missing_ids)
            + ". They will not appear in the rendered figure.",
            stacklevel=2,
        )

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


def _validate_render_config(config: RenderConfig) -> None:
    valid_styles = {"standard", "quadrants", "xyz_strips"}
    if config.figure_style not in valid_styles:
        raise ValueError(
            f"Unsupported figure_style '{config.figure_style}'. Expected one of: {', '.join(sorted(valid_styles))}."
        )

    if config.template_resolution not in {1, 2}:
        raise ValueError("template_resolution must be 1 or 2 mm for the MNI template.")

    if config.figure_style == "quadrants":
        slices = list(config.quadrant_slices or (("x", -10), ("y", 20), ("z", 8)))
        if len(slices) != 3:
            raise ValueError("quadrant_slices must contain exactly three (axis, coord) pairs.")

        for axis_name, coord in slices:
            if axis_name not in {"x", "y", "z"}:
                raise ValueError("quadrant_slices axes must be 'x', 'y', or 'z'.")
            if not isinstance(coord, (int, float)):
                raise ValueError("quadrant_slices coordinates must be numeric.")

    if config.strip_cut_coords is not None:
        invalid_axes = set(config.strip_cut_coords) - {"x", "y", "z"}
        if invalid_axes:
            raise ValueError(
                "strip_cut_coords keys must be limited to 'x', 'y', and 'z'."
            )


def _render_slice_to_array(roi_img, template, axis_name, coord, cmap, alpha, annotate, draw_cross):
    """Render one slice to a numpy RGB array via an off-screen Agg canvas."""
    from nilearn import plotting

    fig = Figure(figsize=(6, 6), dpi=200, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")

    plotting.plot_roi(
        roi_img,
        bg_img=template,
        display_mode=axis_name,
        cut_coords=[coord],
        cmap=cmap,
        alpha=alpha,
        colorbar=False,
        axes=ax,
        annotate=annotate,
        draw_cross=draw_cross,
        black_bg=True,
    )

    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return buf[:, :, :3].copy()


def _crop_black_border(img: np.ndarray, margin: int = 10) -> np.ndarray:
    """Remove near-black borders, leaving a small pixel margin."""
    mask = np.any(img > 15, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return img
    r0 = max(0, rows[0] - margin)
    r1 = min(img.shape[0], rows[-1] + margin + 1)
    c0 = max(0, cols[0] - margin)
    c1 = min(img.shape[1], cols[-1] + margin + 1)
    return img[r0:r1, c0:c1]


def _pad_to_size(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center the image on a black canvas of target size."""
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    r0 = (target_h - h) // 2
    c0 = (target_w - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = img
    return out


def _plot_quadrants(roi_img, template, config: RenderConfig, cmap: str, legend_handles: list[Patch]):
    slices = list(config.quadrant_slices or (("x", -10), ("y", 20), ("z", 8)))

    # 1. Render each slice independently and crop black borders.
    arrays = []
    for axis_name, coord in slices:
        arr = _render_slice_to_array(
            roi_img, template, axis_name, coord,
            cmap, config.alpha, not config.clean, not config.clean,
        )
        arrays.append(_crop_black_border(arr, margin=10))

    # 2. Uniform panel size: bounding box that fits all three slices.
    max_h = max(a.shape[0] for a in arrays)
    max_w = max(a.shape[1] for a in arrays)
    padded = [_pad_to_size(a, max_h, max_w) for a in arrays]

    # 3. Figure size driven purely by panel proportions — no extra whitespace.
    panel_w_in = 5.5
    panel_h_in = panel_w_in * (max_h / max_w)
    figure = Figure(figsize=(panel_w_in * 2, panel_h_in * 2), facecolor="white")
    FigureCanvasAgg(figure)

    gs = figure.add_gridspec(2, 2, wspace=0, hspace=0, left=0, right=1, top=1, bottom=0)
    ax_list = [figure.add_subplot(gs[r, c]) for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]]

    # 4. Place brain images.
    for ax, img in zip(ax_list[:3], padded):
        ax.imshow(img, aspect="auto", interpolation="bilinear")
        ax.set_axis_off()

    # 5. Legend panel — always single column, sized so 10 rows fill the quadrant.
    legend_ax = ax_list[3]
    legend_ax.set_facecolor("white")
    legend_ax.set_axis_off()

    if config.show_legend and legend_handles:
        n_items = len(legend_handles)
        # Font size calculated so 10 rows fill the panel; shrink only if >10 items.
        rows_for_sizing = max(n_items, 10)
        # Empirical line factor: each row occupies ~2.0× fontsize in points
        fontsize = (panel_h_in * 72) / (rows_for_sizing * 2.0)
        fontsize = max(5.0, min(fontsize, 24.0))

        legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=1,
            frameon=True,
            fontsize=fontsize,
            handlelength=1.2,
            handleheight=0.8,
            borderpad=0.8,
            labelspacing=0.5,
        )

    if config.title:
        figure.suptitle(config.title, fontsize=12)

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

