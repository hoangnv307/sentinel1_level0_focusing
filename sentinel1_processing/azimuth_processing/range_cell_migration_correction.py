"""Range Cell Migration Correction from DAD Section 6.3.2."""

from ..rcmc import build_interpolation_table, correct as apply

__all__ = ["build_interpolation_table", "apply"]
