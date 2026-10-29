"""Symbolic sequence generation and manipulation utilities."""
import random
import itertools
import numpy as np
from itertools import permutations
from collections import defaultdict

from .noise import apply_noise_to_sentence, add_background_noise

__all__ = [
    "generate_random_motifs",
    "flatten",
    "tuple_permutations",
    "converter_directed_to_undirected",
    "events_to_symbolic_sequences",
    "get_surviving_dictionary",
    "generate_synthetic_sequence",
    "HigherOrderMarkov",
]


def generate_random_motifs(n2 = 5, n3 = 5, n4 = 5, alph_length = 21):
    """
    Draws random motifs (tuples of symbols in [0, alph_length)).

    Returns a list with n2 motifs of length 2, n3 of length 3 and n4 of length 4.
    Motifs are drawn independently, so duplicates are possible.
    """
    words_list = []

    for _ in range(n2):
        words_list.append(tuple(random.randint(0, alph_length - 1) for _ in range(2)))

    for _ in range(n3):
        words_list.append(tuple(random.randint(0, alph_length - 1) for _ in range(3)))

    for _ in range(n4):
        words_list.append(tuple(random.randint(0, alph_length - 1) for _ in range(4)))

    return words_list


def flatten(lst):
    """Flattens a nested list or tuple into a single flat list."""
    flat_list = []
    for item in lst:
        if isinstance(item, (list, tuple)):  # Check for nested structures
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def tuple_permutations(t):
    l = len(t)
    if l > 1:
        return list(permutations(t, l))
    else:
        return [t]  # just itself if length 1 or 0


def converter_directed_to_undirected(directed_dict):
    undirected_dict = defaultdict(int)

    for element, value in directed_dict.items():
        ele_sort = tuple(sorted(element))
        undirected_dict[ele_sort] += value

    return undirected_dict


def events_to_symbolic_sequences(time_bin, selected_times, spacebar=0, random_sametime=1):
    """
    Converts a table of timestamped events into a symbolic sequence.

    Parameters
    ----------
    time_bin : number or pd.Timedelta
        Gap threshold for the space token: if ``spacebar`` is 1, a -1 is inserted
        whenever two consecutive event times are more than ``time_bin`` apart.
    selected_times : pd.DataFrame
        One row per event, indexed by event time, with a ``unit_id`` column
        holding the symbol (e.g. neuron, stock ticker, email sender).
        The index must be sortable in chronological order (numbers or datetimes).
    spacebar : {0, 1}
    random_sametime : {0, 1}
        If 1, events sharing the same timestamp are put in random order
        (NumPy's global random state; use symba.set_seed for reproducibility).

    Returns
    -------
    hyper_symbolic_sequence : list
        Symbols in time order (-1 for spaces).
    spike_symbolic_sequence : list
        The corresponding event times (-1 for spaces).
    """

    count_same_time = 0
    df_grouped = selected_times.groupby(selected_times.index)

    # Use iterators to avoid storing all groups at once
    spike_symbolic_sequence = []
    hyper_symbolic_sequence = []

    last_label = None  # Track the last timestamp

    # Iterate directly over grouped data
    for label, group in df_grouped:
        hyperedge = list(group['unit_id'])  # Convert to list once

        if len(hyperedge) > 1 and random_sametime:
            count_same_time += 1
            np.random.shuffle(hyperedge)

        # Handle spacebar condition efficiently
        if spacebar and last_label is not None and (label > last_label + time_bin):
            spike_symbolic_sequence.append(-1)
            hyper_symbolic_sequence.append(-1)

        spike_symbolic_sequence.extend([label] * len(hyperedge))
        hyper_symbolic_sequence.extend(hyperedge)

        last_label = label  # Update last seen timestamp

    print(f"{count_same_time} events occurred at the same time")

    return hyper_symbolic_sequence, spike_symbolic_sequence


def get_surviving_dictionary(noisy_sentence, original_dictionary, ordered_words):
    """
    Returns only the words from the original dictionary that still exist
    in the noisy sentence at least once.
    """
    surviving_words = []

    if ordered_words == 0:
        # For unordered, we sort the tuples (so that e.g. (1,2) and (2,1) are considered the same)
        noisy_set = set(tuple(sorted(word)) for word in noisy_sentence)
        for word in original_dictionary:
            if tuple(sorted(word)) in noisy_set:
                surviving_words.append(word)
    else:
        # For ordered, we just check exact matches
        noisy_set = set(noisy_sentence)
        for word in original_dictionary:
            if word in noisy_set:
                surviving_words.append(word)

    return surviving_words


def generate_synthetic_sequence(alph_length= 20, num_unique_2words = 10, num_unique_3words = 10, num_unique_4words = 10, rep_2words = 10, rep_3words = 10, rep_4words = 10, nsr = 10,
                              gamma = 1, spacebar = 1, space_weight = 10, ordered_words = 0,  time_bin = 2, different_alphabet = 0,
                              p_sub=0,
                              p_remove=0):
    """
    Generates an artificial symbolic sequence with planted motifs, used as ground truth.

    Random motifs of length 2, 3 and 4 are drawn (duplicates are merged, so the
    number of distinct motifs can be smaller than requested), each repeated
    ``rep_kwords`` times, optionally corrupted by symbol substitution/deletion
    noise, mixed with Zipf-distributed background noise symbols, shuffled and
    flattened into a sequence.

    Parameters
    ----------
    alph_length : int
        Alphabet size.
    num_unique_2words, num_unique_3words, num_unique_4words : int
        Number of distinct motifs to plant for each length.
    rep_2words, rep_3words, rep_4words : int
        Repetitions of each motif (0 disables that length).
    nsr : float
        Noise-to-signal ratio: noise symbols per motif symbol.
    gamma : float
        Exponent of the Zipf rank-frequency distribution of the noise symbols.
    spacebar, space_weight : int, float
        Whether (and how often) to include the space token -1 among noise symbols.
    ordered_words : {0, 1}
        0: motifs are unordered (their symbols may appear in any order);
        1: motifs are ordered.
    time_bin : int
        Extra time gap assigned to spaces in the returned event times.
    different_alphabet : {0, 1}
        If 1, noise symbols come from a separate alphabet [alph_length, 2*alph_length).
    p_sub, p_remove : float
        Per-symbol substitution and deletion probabilities applied to the motifs.

    Returns
    -------
    hyper_symbolic_sequence : list
    spike_symbolic_sequence : list
        Fake event times, one per symbol (-1 for spaces).
    surviving_dict_per_size : dict[int, list[tuple]]
        Ground-truth motifs per length that are still intact after symbol noise.
    surviving_words : list[tuple]
    """
    words_in_sentence_per_size_dict = {2:[], 3:[], 4:[]}

    if rep_2words == 0:
        num_unique_2words = 0
    if rep_3words == 0:
        num_unique_3words = 0
    if rep_4words == 0:
        num_unique_4words = 0

    ## Step 1: Generate random words "dictionary": it is not a python dictionary, its a list of tuple
    words_list = generate_random_motifs(num_unique_2words, num_unique_3words, num_unique_4words, alph_length)

    print(len(words_list))
    if ordered_words == 0:
        ### for unordered case, since words with different order are the same
        ### we have first to order each word, then take the set of words:
        words_list = [tuple(sorted(word)) for word in words_list]
        words_list_unique = list(set(words_list))
        words_list = words_list_unique
    else:
        ### for the ordered case this is enough:
        words_list_unique = list(set(words_list))
        words_list = words_list_unique
    print(len(words_list))

    words2_list = [word for word in words_list if len(word) == 2]
    words3_list = [word for word in words_list if len(word) == 3]
    words4_list = [word for word in words_list if len(word) == 4]

    words_list = sorted(words_list, key=lambda x: len(x))

    ### Create the sentence
    random_sentence = (
        [tup for tup in words2_list for _ in range(rep_2words)] +
        [tup for tup in words3_list for _ in range(rep_3words)] +
        [tup for tup in words4_list for _ in range(rep_4words)]
    )


    # Shuffle the list
    random.shuffle(random_sentence)

    # If ordered_words is 0, shuffle each tuple's contents
    if ordered_words == 0:
        random_sentence = [tuple(random.sample(tup, len(tup))) for tup in random_sentence]

    ### Create a dictionary to store the words in the sentence
    words_in_sentence = words_list

    words_in_sentence_per_size_dict[2] = words2_list
    words_in_sentence_per_size_dict[3] = words3_list
    words_in_sentence_per_size_dict[4] = words4_list


    print("num unique 2words", len(words2_list), "num unique 3words", len(words3_list), "num unique 4words", len(words4_list))

    ### Step 2.0: add different types of noise to the signal: symbol depletion, symbol substitution, (timing jitter?)
    ### symbol depletion: remove a fraction of symbols from the words in the sentence

    # 1. Apply Noise
    noisy_sentence = random_sentence  # default: no noise applied
    surviving_words = words_in_sentence
    surviving_dict_per_size = words_in_sentence_per_size_dict
    if p_sub > 0 or p_remove > 0:
        print(f"Applying noise: p_sub={p_sub}, p_remove={p_remove}")
        noisy_sentence = apply_noise_to_sentence(
            random_sentence,
            alph_length,
            p_sub=p_sub,
            p_remove=p_remove
        )

        # 2. Filter the dictionary to establish the true ground truth
        surviving_words = get_surviving_dictionary(
            noisy_sentence,
            words_in_sentence,
            ordered_words
        )

        # Optional: Update your size dictionary based on survivors
        surviving_dict_per_size = {
            2: [w for w in surviving_words if len(w) == 2],
            3: [w for w in surviving_words if len(w) == 3],
            4: [w for w in surviving_words if len(w) == 4]
        }


    ### Step 3: add background noise to the sentence. It also add spaces, associated with -1 by default
    noisy_sentence = add_background_noise(noisy_sentence, nsr, spacebar, space_weight, alph_length, gamma,
                                               different_alphabet = different_alphabet,
                                               cutoff_type="none")

    ## Step 4: Flat the noisy sentence to create the artifical symbolic sequence

    flattened_list = flatten(noisy_sentence)
    fake_spikes = [i for i in range(len(flattened_list))]
    time_gap = 0
    for el_id, element in enumerate(flattened_list):
        if element == -1:
            fake_spikes[el_id] = -1
            time_gap += time_bin + 1
        else:
            fake_spikes[el_id] = time_gap + el_id

    hyper_symbolic_sequence = flattened_list
    spike_symbolic_sequence = fake_spikes

    ### lenght of the symbolic sequence compared with the signal (number of symbols in the sentence without noise)
    len_symbols = 0
    for word in random_sentence:
        len_symbols += len(word)
    print("non noisy symbols number (signal length): ", len_symbols)
    print("symbolic sequence total length: ", len(hyper_symbolic_sequence))

    return hyper_symbolic_sequence, spike_symbolic_sequence, surviving_dict_per_size, surviving_words


class HigherOrderMarkov:
    """
    Random Markov chain of a given order, used to generate sequences whose
    k-tuple statistics are exactly those expected by the order-(k-2) null model.

    The initial distribution and every transition distribution are drawn from a
    symmetric Dirichlet(alpha) distribution.
    """

    def __init__(self, alphabet_size, order, alpha=1.0, seed=None):
        """
        alphabet_size : number of symbols (0,...,alphabet_size-1)
        order         : Markov order
                        0 = IID
                        1 = first-order
                        2 = second-order
                        ...
        alpha         : Dirichlet concentration parameter
        seed          : seed of the chain's own random generator (used both to
                        draw the transition probabilities and to sample)
        """

        self.A = alphabet_size
        self.order = order
        self.rng = np.random.default_rng(seed)

        # ---------- IID case ----------
        if order == 0:
            self.initial = self.rng.dirichlet(np.full(self.A, alpha))
            return

        # ---------- initial context ----------
        self.initial = self.rng.dirichlet(np.full(self.A, alpha))

        # ---------- transition probabilities ----------
        self.transitions = {}

        contexts = itertools.product(range(self.A), repeat=order)

        for c in contexts:
            self.transitions[c] = self.rng.dirichlet(
                np.full(self.A, alpha)
            )

    def sample(self, length):

        seq = np.zeros(length, dtype=int)

        # ---------- IID ----------
        if self.order == 0:
            return self.rng.choice(
                self.A,
                size=length,
                p=self.initial
            )

        # ---------- initial symbols ----------
        seq[:self.order] = self.rng.choice(
            self.A,
            size=self.order,
            p=self.initial
        )

        # ---------- recursive generation ----------
        for t in range(self.order, length):

            context = tuple(seq[t-self.order:t])

            seq[t] = self.rng.choice(
                self.A,
                p=self.transitions[context]
            )

        return seq

    def sample_dataset(self, n_sequences, length):

        return np.array(
            [self.sample(length) for _ in range(n_sequences)]
        )
