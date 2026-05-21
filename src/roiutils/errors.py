"""Custom exceptions for roiutils."""


class RoiUtilsError(Exception):
    """Base exception for roiutils errors."""


class AtlasValidationError(RoiUtilsError):
    """Raised when an atlas or atlas metadata is invalid."""


class RoiSelectionError(RoiUtilsError):
    """Raised when ROI selection cannot be resolved."""
