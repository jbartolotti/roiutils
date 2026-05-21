import nibabel as nib
import numpy as np
import pytest

from roiutils.errors import RoiSelectionError
from roiutils.models import AtlasSpec, SelectionConfig
from roiutils.roi_select import build_roi_mask, resolve_roi_selection


@pytest.fixture
def atlas() -> AtlasSpec:
    data = np.array([[[0, 2], [4, 4]]], dtype=np.int16)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    return AtlasSpec(image=image, labels_by_id={2: "Left", 4: "Right"})


def test_selection_by_id_and_label(atlas: AtlasSpec) -> None:
    selection = resolve_roi_selection(atlas, [2, "Right"])
    assert selection.ids == (2, 4)


def test_selection_missing_strict_fails(atlas: AtlasSpec) -> None:
    with pytest.raises(RoiSelectionError):
        resolve_roi_selection(atlas, ["Missing"], config=None)


def test_selection_missing_permissive(atlas: AtlasSpec) -> None:
    selection = resolve_roi_selection(
        atlas,
        ["Missing", "Right"],
        config=SelectionConfig(strict=False),
    )
    assert selection.ids == (4,)


def test_build_mask(atlas: AtlasSpec) -> None:
    selection = resolve_roi_selection(atlas, [4])
    mask = build_roi_mask(atlas, selection)
    mask_data = mask.get_fdata()
    assert int(mask_data.sum()) == 2
    assert mask_data.shape == atlas.image.shape
