"""Project-specific exception classes."""

from __future__ import annotations


class ReadingBetweenLayersError(Exception):
    """Base exception for this project."""


class UnsupportedModelTaskCombo(ReadingBetweenLayersError):
    """Raised when a (model, task) pair is not allowed by compatibility."""


class CacheMiss(ReadingBetweenLayersError):
    """Raised when a required cache entry is absent and no compute fn is given."""


class GeometryComputationError(ReadingBetweenLayersError):
    """Raised when a geometry metric fails for any reason."""


class TierSkipError(ReadingBetweenLayersError):
    """Raised when a tier-C model is requested without an explicit override."""


class InvalidConfigError(ReadingBetweenLayersError):
    """Raised when a config is internally inconsistent."""
