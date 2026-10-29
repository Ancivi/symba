"""SYMBA: symbolic motif significance analysis.

Supplementary code package for the SYMBA paper. Re-exports every public
function from its submodules so that both ``import symba`` / ``symba.func(...)``
and ``from symba import *`` / ``func(...)`` call styles work.
"""
from .sequences import *
from .shuffling import *
from .probability import *
from .bayesian import *
from .noise import *
from .metrics import *
from .datasets import *
from .plotting import *
from .utils import *

from . import sequences, shuffling, probability, bayesian, noise, metrics, datasets, plotting, utils

__all__ = (
    utils.__all__
    + sequences.__all__
    + shuffling.__all__
    + probability.__all__
    + bayesian.__all__
    + noise.__all__
    + metrics.__all__
    + datasets.__all__
    + plotting.__all__
)
