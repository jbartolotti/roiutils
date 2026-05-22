import nibabel as nib
import numpy as np
import pytest

from roiutils.errors import RoiSelectionError
from roiutils.models import AtlasSpec
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


def test_selection_missing_label_always_fails(atlas: AtlasSpec) -> None:
    with pytest.raises(RoiSelectionError):
        resolve_roi_selection(atlas, ["Missing", "Right"], config=None)


def test_selection_numeric_without_label_warns() -> None:
    data = np.array([[[0, 2], [9, 9]]], dtype=np.int16)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    atlas = AtlasSpec(image=image, labels_by_id={2: "Left"})

    with pytest.warns(UserWarning, match="missing label entries"):
        selection = resolve_roi_selection(atlas, [9])

    assert selection.ids == (9,)
    assert selection.labels_by_id[9] == "ROI 9"


def test_build_mask(atlas: AtlasSpec) -> None:
    selection = resolve_roi_selection(atlas, [4])
    mask = build_roi_mask(atlas, selection)
    mask_data = mask.get_fdata()
    assert int(mask_data.sum()) == 2
    assert mask_data.shape == atlas.image.shape
