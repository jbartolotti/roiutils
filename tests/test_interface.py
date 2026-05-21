import nibabel as nib
import numpy as np

from roiutils.interface import RoiWorkflow, build_roi_mask, resolve_roi_selection
from roiutils.models import AtlasSpec


def test_interface_functions_and_workflow() -> None:
    data = np.array([[[0, 1], [1, 2]]], dtype=np.int16)
    atlas = AtlasSpec(
        image=nib.Nifti1Image(data, affine=np.eye(4)),
        labels_by_id={1: "A", 2: "B"},
    )

    selection = resolve_roi_selection(atlas, ["A"])
    mask = build_roi_mask(atlas, selection)
    assert int(mask.get_fdata().sum()) == 2

    workflow = RoiWorkflow(atlas)
    selection2 = workflow.resolve_selection([2])
    mask2 = workflow.build_mask(selection2)
    assert int(mask2.get_fdata().sum()) == 1
