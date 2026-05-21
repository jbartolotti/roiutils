"""Core domain models for roiutils."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib


@dataclass(frozen=True)
class AtlasSpec:
    """Atlas data and label metadata used for ROI selection."""

    image: nib.spatialimages.SpatialImage
    labels_by_id: Mapping[int, str]
    source_path: Path | None = None
    template_name: str = "MNI152"


@dataclass(frozen=True)
class RoiSelection:
    """Normalized ROI selection resolved to atlas IDs."""

    ids: tuple[int, ...]
    labels_by_id: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for resolving ROI selections."""

    strict: bool = True


SelectionInput = Sequence[int | str]
