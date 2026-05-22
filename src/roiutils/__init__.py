"""Public package surface for roiutils."""

from .interface import (
    build_roi_mask,
    load_atlas,
    plot_roi_overlay,
    resolve_roi_selection,
)

__all__ = ["build_roi_mask", "load_atlas", "plot_roi_overlay", "resolve_roi_selection"]
__version__ = "0.1.0"
