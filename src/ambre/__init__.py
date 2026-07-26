"""AMBRE: aligned relation-sequence and non-backtracking encoding for KGC.

This package contains the anonymous implementation used for multi-knowledge-
graph completion experiments. It depends on PyKEEN for datasets and triples
factory utilities.
"""

from .multi_factory import DatasetFactories, MultiTriplesFactory
from .model import MUGKGC

__all__ = [
    "DatasetFactories",
    "MultiTriplesFactory",
    "MUGKGC",
]
