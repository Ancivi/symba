"""Reproducibility helpers."""
import random

import numpy as np

__all__ = ["set_seed"]


def set_seed(seed):
    """Seed every random number generator used by symba.

    All random steps in the package (planted motifs, background noise, symbol
    noise, shuffles, ordering of simultaneous events) draw from Python's
    ``random`` module or NumPy's global random state, so seeding both before an
    experiment makes it reproducible.

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
