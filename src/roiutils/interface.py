"""Public user-facing API for roiutils."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import nibabel as nib

from .atlas_io import load_atlas as _load_atlas
from .figures import plot_roi_overlay as _plot_roi_overlay
from .models import AtlasSpec, RenderConfig, RoiSelection, SelectionConfig, SelectionInput
from .roi_select import build_roi_mask as _build_roi_mask
from .roi_select import resolve_roi_selection as _resolve_roi_selection


def load_atlas(
    atlas_path: str | Path,
    labels_by_id: Mapping[int, str],
    *,
    template_name: str = "MNI152",
) -> AtlasSpec:
    """Load and validate an indexed atlas and its label mapping."""
    return _load_atlas(atlas_path, labels_by_id, template_name=template_name)


def resolve_roi_selection(
    atlas: AtlasSpec,
    selection: SelectionInput,
    *,
    strict: bool = True,
) -> RoiSelection:
    """Resolve ROI IDs or labels to a normalized ROI selection object."""
    config = SelectionConfig(strict=strict)
    return _resolve_roi_selection(atlas, selection, config=config)


def build_roi_mask(atlas: AtlasSpec, selection: RoiSelection) -> nib.Nifti1Image:
    """Build a binary mask image for selected ROIs."""
    return _build_roi_mask(atlas, selection)


def plot_roi_overlay(
    atlas: AtlasSpec,
    selection: RoiSelection,
    config: RenderConfig | None = None,
) -> Path | None:
    """Render selected ROIs as a colorized overlay on an MNI152 template brain.

    Returns the saved output path, or None if displayed interactively.
    """
    return _plot_roi_overlay(atlas, selection, config)


class RoiWorkflow:
    """Class-based workflow wrapper for repeated atlas operations."""

    def __init__(self, atlas: AtlasSpec):
        self.atlas = atlas

    def resolve_selection(self, selection: SelectionInput, *, strict: bool = True) -> RoiSelection:
        return resolve_roi_selection(self.atlas, selection, strict=strict)

    def build_mask(self, selection: RoiSelection) -> nib.Nifti1Image:
        return build_roi_mask(self.atlas, selection)

    def plot_overlay(
        self,
        selection: RoiSelection,
        config: RenderConfig | None = None,
    ) -> Path | None:
        return plot_roi_overlay(self.atlas, selection, config)
