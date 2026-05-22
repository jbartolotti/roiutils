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


@dataclass
class RenderConfig:
    """Configuration for ROI overlay rendering."""

    display_mode: str = "z"
    """Slice plane(s): 'ortho', 'x', 'y', 'z', 'tiled'."""

    cut_coords: int | tuple | None = 7
    """Number of evenly-spaced cuts (int), specific MNI coords (tuple), or None for auto."""

    cmap: str | None = None
    """Colormap name. Defaults to 'tab10' for <=10 ROIs, 'tab20' for more."""

    alpha: float = 0.7
    """Overlay opacity (0=transparent, 1=opaque)."""

    dpi: int = 150
    """Output image resolution."""

    output_path: str | Path | None = None
    """Destination PNG path. If None the figure is shown interactively."""

    title: str | None = None
    """Optional figure title."""


SelectionInput = Sequence[int | str]
