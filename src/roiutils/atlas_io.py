"""Atlas loading and validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import nibabel as nib
import numpy as np

from .errors import AtlasValidationError
from .models import AtlasSpec


def load_atlas(
    atlas_path: str | Path,
    labels_by_id: Mapping[int, str],
    *,
    template_name: str = "MNI152",
) -> AtlasSpec:
    """Load an atlas image and validate IDs against provided labels."""
    atlas_path = Path(atlas_path)
    image = nib.load(str(atlas_path))
    atlas = AtlasSpec(
        image=image,
        labels_by_id=dict(labels_by_id),
        source_path=atlas_path,
        template_name=template_name,
    )
    validate_atlas(atlas)
    return atlas


def validate_atlas(atlas: AtlasSpec) -> None:
    """Validate atlas voxel values and labels mapping consistency."""
    data = atlas.image.get_fdata()
    if not np.all(np.isfinite(data)):
        raise AtlasValidationError("Atlas image contains non-finite values.")

    rounded = np.rint(data)
    if not np.allclose(data, rounded):
        raise AtlasValidationError("Atlas image must contain integer label values.")

    ids_in_image = _non_background_ids(rounded)
    missing = [roi_id for roi_id in ids_in_image if roi_id not in atlas.labels_by_id]
    if missing:
        missing_text = ", ".join(str(x) for x in missing[:10])
        raise AtlasValidationError(
            f"Atlas labels are missing IDs present in image: {missing_text}"
        )


def load_labels_tsv(path: str | Path) -> dict[int, str]:
    """Load atlas label metadata from a TSV with columns: id and label."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AtlasValidationError("Labels TSV is empty.")

    header = [part.strip().lower() for part in lines[0].split("\t")]
    if "id" not in header or "label" not in header:
        raise AtlasValidationError("Labels TSV must contain 'id' and 'label' columns.")

    id_col = header.index("id")
    label_col = header.index("label")

    labels: dict[int, str] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) <= max(id_col, label_col):
            raise AtlasValidationError(f"Malformed TSV row: {line}")
        roi_id = int(parts[id_col])
        label = parts[label_col].strip()
        labels[roi_id] = label

    if not labels:
        raise AtlasValidationError("Labels TSV had no usable rows.")

    return labels


def _non_background_ids(data: np.ndarray) -> list[int]:
    unique_values = np.unique(data.astype(np.int64))
    ids = [int(value) for value in unique_values if int(value) != 0]
    return ids
