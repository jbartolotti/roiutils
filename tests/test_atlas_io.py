from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from roiutils.atlas_io import load_atlas
from roiutils.errors import AtlasValidationError


def _write_atlas(tmp_path: Path, data: np.ndarray) -> Path:
    image = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmp_path / "atlas.nii.gz"
    nib.save(image, str(path))
    return path


def test_load_atlas_success(tmp_path: Path) -> None:
    data = np.array([[[0, 1], [2, 2]]], dtype=np.int16)
    atlas_path = _write_atlas(tmp_path, data)

    atlas = load_atlas(atlas_path, {1: "ROI One", 2: "ROI Two"})

    assert atlas.template_name == "MNI152"
    assert atlas.labels_by_id[1] == "ROI One"


def test_load_atlas_missing_labels_allowed(tmp_path: Path) -> None:
    data = np.array([[[0, 1], [2, 2]]], dtype=np.int16)
    atlas_path = _write_atlas(tmp_path, data)

    atlas = load_atlas(atlas_path, {2: "ROI Two"})
    assert atlas.labels_by_id[2] == "ROI Two"


def test_load_atlas_non_integer_fails(tmp_path: Path) -> None:
    data = np.array([[[0.0, 1.2]]], dtype=np.float32)
    atlas_path = _write_atlas(tmp_path, data)

    with pytest.raises(AtlasValidationError):
        load_atlas(atlas_path, {1: "ROI One"})
