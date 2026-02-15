"""InSAR processing modules."""

try:
    from .insar_reader import InSARReader
except ImportError:
    InSARReader = None

__all__ = ['InSARReader']
