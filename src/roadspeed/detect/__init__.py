"""Detection backends. Background subtraction is the default and needs no downloads."""

from .base import Detection, make_detector

__all__ = ["Detection", "make_detector"]
