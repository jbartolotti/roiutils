"""ROI selection and mask creation logic."""

from __future__ import annotations

from collections.abc import Iterable

import nibabel as nib
import numpy as np

from .errors import RoiSelectionError
from .models import AtlasSpec, RoiSelection, SelectionConfig, SelectionInput


def resolve_roi_selection(
    atlas: AtlasSpec,
    selection: SelectionInput,
    *,
    config: SelectionConfig | None = None,
) -> RoiSelection:
    """Resolve mixed ROI identifiers (IDs or labels) into atlas IDs."""
    config = config or SelectionConfig()
    label_to_id = {label.lower(): roi_id for roi_id, label in atlas.labels_by_id.items()}

    resolved: list[int] = []
    missing: list[str] = []

    for item in selection:
        roi_id: int | None
        if isinstance(item, int):
            roi_id = item if item in atlas.labels_by_id else None
            missing_token = str(item)
        elif isinstance(item, str):
            normalized = item.strip().lower()
            roi_id = label_to_id.get(normalized)
            missing_token = item
        else:
            raise RoiSelectionError(f"Unsupported ROI selector type: {type(item)!r}")

        if roi_id is None:
            missing.append(missing_token)
            continue

        resolved.append(roi_id)

    unique_ids = tuple(sorted(set(resolved)))
    if missing and config.strict:
        missing_text = ", ".join(missing)
        raise RoiSelectionError(f"Could not resolve ROI selector(s): {missing_text}")
    if not unique_ids:
        raise RoiSelectionError("No ROI IDs were resolved from selection input.")

    selected_labels = {roi_id: atlas.labels_by_id[roi_id] for roi_id in unique_ids}
    return RoiSelection(ids=unique_ids, labels_by_id=selected_labels)


def build_roi_mask(atlas: AtlasSpec, selection: RoiSelection) -> nib.Nifti1Image:
    """Create a binary mask image for selected ROI IDs."""
    atlas_data = np.rint(atlas.image.get_fdata()).astype(np.int32)
    selected = np.isin(atlas_data, np.array(selection.ids, dtype=np.int32))

    if int(selected.sum()) == 0:
        raise RoiSelectionError("Selected ROIs are empty in the provided atlas image.")

    mask = selected.astype(np.uint8)
    return nib.Nifti1Image(mask, affine=atlas.image.affine, header=atlas.image.header)
