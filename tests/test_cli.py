from pathlib import Path

import nibabel as nib
import numpy as np

from roiutils.cli import main


def test_cli_creates_mask_and_metadata(tmp_path: Path) -> None:
    atlas_data = np.array([[[0, 1], [2, 2]]], dtype=np.int16)
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(atlas_data, affine=np.eye(4)), str(atlas_path))

    labels_path = tmp_path / "labels.tsv"
    labels_path.write_text("id\tlabel\n1\tROI_A\n2\tROI_B\n", encoding="utf-8")

    output_path = tmp_path / "mask.nii.gz"
    metadata_path = tmp_path / "meta.json"

    exit_code = main(
        [
            "--atlas",
            str(atlas_path),
            "--labels",
            str(labels_path),
            "--select",
            "ROI_B",
            "--output",
            str(output_path),
            "--metadata",
            str(metadata_path),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert metadata_path.exists()

    mask = nib.load(str(output_path)).get_fdata()
    assert int(mask.sum()) == 2
