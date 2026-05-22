"""ROI selection and mask creation logic."""

from __future__ import annotations

import warnings

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
    """Resolve mixed ROI identifiers (IDs or labels) into atlas IDs.

    Numeric ROI IDs are treated as direct atlas indices and do not require a
    label mapping entry. Label-name selectors must exist in atlas.labels_by_id.
    """
    config = config or SelectionConfig()
    label_to_id = {label.lower(): roi_id for roi_id, label in atlas.labels_by_id.items()}

    resolved: list[int] = []
    missing_labels: list[str] = []

    for item in selection:
        if isinstance(item, int):
            resolved.append(item)
            continue
        elif isinstance(item, str):
            normalized = item.strip().lower()
            roi_id = label_to_id.get(normalized)
            if roi_id is None:
                missing_labels.append(item)
                continue
            resolved.append(roi_id)
        else:
            raise RoiSelectionError(f"Unsupported ROI selector type: {type(item)!r}")

    unique_ids = tuple(sorted(set(resolved)))
    if missing_labels:
        missing_text = ", ".join(missing_labels)
        raise RoiSelectionError(
            f"Could not resolve ROI label selector(s): {missing_text}"
        )
    if not unique_ids:
        raise RoiSelectionError("No ROI IDs were resolved from selection input.")

    unlabeled_ids = [roi_id for roi_id in unique_ids if roi_id not in atlas.labels_by_id]
    if unlabeled_ids:
        warnings.warn(
            "Selected ROI IDs missing label entries: "
            + ", ".join(str(x) for x in unlabeled_ids)
            + ". Using numeric IDs directly.",
            stacklevel=2,
        )

    selected_labels = {
        roi_id: atlas.labels_by_id.get(roi_id, f"ROI {roi_id}") for roi_id in unique_ids
    }
    return RoiSelection(ids=unique_ids, labels_by_id=selected_labels)


def build_roi_mask(atlas: AtlasSpec, selection: RoiSelection) -> nib.Nifti1Image:
    """Create a binary mask image for selected ROI IDs."""
    atlas_data = np.rint(atlas.image.get_fdata()).astype(np.int32)
    selected = np.isin(atlas_data, np.array(selection.ids, dtype=np.int32))

    if int(selected.sum()) == 0:
        raise RoiSelectionError("Selected ROIs are empty in the provided atlas image.")

    mask = selected.astype(np.uint8)
    return nib.Nifti1Image(mask, affine=atlas.image.affine, header=atlas.image.header)
