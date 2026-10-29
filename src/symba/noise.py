"""Noise-injection models for synthetic symbolic sequences."""
import random
import numpy as np

__all__ = [
    "apply_noise_to_sentence",
    "add_background_noise",
    "replace_symbol_noise",
    "delete_symbol_noise",
]


def apply_noise_to_sentence(sentence, alph_length=21, p_sub=0.05, p_remove=0.05):
    """
    Applies symbol substitution and deletion noise to a sequence of motifs (tuples).

    Each symbol is independently replaced by a uniformly drawn symbol of the
    alphabet with probability ``p_sub`` (possibly by itself), then deleted with
    probability ``p_remove``. Motifs that lose all their symbols are dropped.
    """
    noisy_sentence = []
    alphabet = list(range(alph_length))

    for word in sentence:
        current_word = list(word)

        # 1. Substitution (allows substituting a symbol with itself)
        if p_sub > 0:
            for i in range(len(current_word)):
                if random.random() < p_sub:
                    current_word[i] = random.choice(alphabet)

        # 2. Deletion
        if p_remove > 0:
            # Iterate backwards so popping elements doesn't mess up our loop indices
            for i in range(len(current_word) - 1, -1, -1):
                if random.random() < p_remove:
                    current_word.pop(i)

        # 3. Only keep the word if deletion hasn't erased it completely
        if current_word:
            noisy_sentence.append(tuple(current_word))

    return noisy_sentence


def add_background_noise(
    random_sentence,
    nsr=1,
    spacebar=1,
    space_weight=10,
    alph_length=21,
    gamma=1,
    different_alphabet=0,
    cutoff_type="none",   # "none" | "exp" | "stretched" | "logistic" | "hard"
    r_cut=30,                  # cutoff rank scale
    beta=2.0,                  # sharpness for "stretched": exp(-(r/rc)^beta)
    k=2.0,                     # steepness for "logistic"
    eps=0.0                    # residual past r_cut for "hard" (0 -> true zero)
):
    """
    Inserts background noise symbols among the motifs of a sentence.

    Adds ``round(nsr * total motif symbols)`` single noise symbols whose
    rank-frequency distribution follows a power law r^(-gamma) (Zipf; uniform
    if gamma == 0), optionally multiplied by a cutoff kernel. If ``spacebar``
    is 1, the space token -1 is included with extra weight ``space_weight``.
    The motifs and noise symbols are then shuffled together.

    Returns the noisy sentence: a list mixing motif tuples and single symbols.
    """
    # Build alphabet
    numbers = list(range(alph_length, 2*alph_length)) if different_alphabet == 1 else list(range(alph_length))
    if spacebar == 1:
        numbers = numbers + [-1]  # space token is last

    R = np.arange(1, len(numbers) + 1, dtype=float)

    # Base power-law r^{-gamma} (uniform if gamma==0)
    base = R ** (-gamma) if gamma != 0 else np.ones_like(R)

    # Cutoff kernel
    if cutoff_type == "none":
        kernel = np.ones_like(R)
        print("using no cutoff in the noise addition")
    elif cutoff_type == "exp":
        kernel = np.exp(-R / float(r_cut))
    elif cutoff_type == "stretched":
        kernel = np.exp(- (R / float(r_cut)) ** float(beta))
    elif cutoff_type == "logistic":
        # divide by (1 + exp(k(r-r_c))) ≈ sharp step at r_c; larger k = sharper
        kernel = 1.0 / (1.0 + np.exp(float(k) * (R - float(r_cut))))
    elif cutoff_type == "hard":
        kernel = (R <= float(r_cut)).astype(float)
        if eps > 0:
            kernel = np.where(R <= float(r_cut), 1.0, float(eps))
    else:
        raise ValueError("cutoff_type must be one of: 'none', 'exp', 'stretched', 'logistic', 'hard'")

    weights = base * kernel

    # Optional space weighting (space is last rank)
    if spacebar == 1 and space_weight is not None and space_weight != 1:
        weights[-1] *= float(space_weight)

    # Guard against all-zero weights
    if not np.any(weights > 0):
        weights = np.ones_like(weights)

    # Number of noise symbols to add
    len_symbols = sum(len(word) for word in random_sentence)
    num_symbols_to_add = int(np.round(len_symbols * nsr))

    noisy_symbols = random.choices(numbers, weights=weights.tolist(), k=num_symbols_to_add)
    noisy_sentence = random_sentence + noisy_symbols
    random.shuffle(noisy_sentence)
    return noisy_sentence


def replace_symbol_noise(sequence, p_s, alph_length, rng=None):
    """
    Substitute each symbol in sequence with probability p_s
    with a randomly chosen symbol from [0, alph_length).

    Parameters:
    - sequence:    list or array of integer symbols
    - p_s:         substitution probability per symbol
    - alph_length: alphabet size (symbols are in [0, alph_length))
    - rng:         optional np.random.Generator; if omitted, one is derived from
                   NumPy's global random state, so symba.set_seed makes it reproducible

    Returns:
    - noisy sequence as a list
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(2**31 - 1))

    sequence = np.array(sequence)
    n = len(sequence)

    # draw a boolean mask: True where substitution happens
    mask = rng.random(n) < p_s

    # draw random replacement symbols for every position
    # (only those where mask=True will be used)
    replacements = rng.integers(0, alph_length, size=n)

    noisy = sequence.copy()
    noisy[mask] = replacements[mask]

    return noisy.tolist()


def delete_symbol_noise(sequence, p_s, rng=None):
    """
    Remove each symbol in sequence with probability p_s.

    Parameters:
    - sequence:    list or array of integer symbols
    - p_s:         deletion probability per symbol
    - rng:         optional np.random.Generator; if omitted, one is derived from
                   NumPy's global random state, so symba.set_seed makes it reproducible

    Returns:
    - noisy sequence as a list (shorter than input by ~p_s * n symbols)
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(2**31 - 1))

    sequence = np.array(sequence)
    n = len(sequence)

    # draw a boolean mask: True where symbol is KEPT
    mask = rng.random(n) >= p_s

    return sequence[mask].tolist()
