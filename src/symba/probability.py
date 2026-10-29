"""Empirical counting and analytical expected-probability computations for motifs/hyperedges."""
import numpy as np
from collections import defaultdict, Counter

from .sequences import tuple_permutations, converter_directed_to_undirected
from .shuffling import shuffle_preserve, kandel_uniform_k_shuffle

__all__ = [
    "count_ktuples",
    "empirical_ktuples",
    "find_zscores",
    "positive_or_negative_scores_persize",
    "observed_prob_and_count_for_batch",
    "expected_undirected_probability_all_batches_stats",
    "expected_undirected_probability_all_batches",
    "expected_undirected_probability_single_batch",
    "batch_statistics_undirected",
    "expected_directed_probability_all_batches",
    "expected_directed_probability_single_batch",
    "expected_directed_probability_all_batches_stats",
    "batch_statistics_directed",
    "batch_zscores_per_size",
    "observed_counts_and_expected_probs",
    "overlapping_hyperedges_spikes_of_size",
    "observed_prob_and_count_directed_from_sequence",
    "observed_probability_and_count_from_hyperedges",
    "expected_undirected_probability",
    "expected_directed_probability",
]


def count_ktuples(seq, k, ordered_words=1):
    """
    Count k-tuples in seq.
    If ordered_words==1: (x1,x2,...,xk) and (x2,x1,...,xk) are different keys.
    If ordered_words==0: tuples are keyed by their sorted multiset, so order is ignored.
    """
    c = Counter()
    for i in range(len(seq) - k + 1):
        window = seq[i:i+k]
        kt = tuple(window) if ordered_words == 1 else tuple(sorted(window))
        c[kt] += 1
    return c

def empirical_ktuples(orig_seq, k, k_preserved, B=100, ordered_words=1, shuffle_method = 'non_biased'):
    """
    Compute empirical null statistics for the k-tuples of a sequence from B shuffles.

    The null model preserves the counts of all k_preserved-tuples. With
    shuffle_method='non_biased' (default) shuffles are drawn uniformly with the
    Kandel et al. algorithm (kandel_uniform_k_shuffle); with 'biased' a faster
    random Eulerian path (shuffle_preserve) is used, which is not uniform.
    'ordered_words' controls whether tuples are treated as ordered (1) or unordered (0)
    consistently in both the original and randomized sequences.
    Only tuples present in the original sequence are evaluated.

    Returns
    -------
    dict[tuple, dict]
        For each tuple: observed count 'obs', null mean 'mu_null' and standard
        deviation 'sd_null', z-score 'z', and empirical p-values 'p_enrich',
        'p_deplete', 'p_two' (with the +1 correction).
    """
    k_node = k_preserved - 1

    # Count in the original sequence under the chosen tuple semantics
    orig_counts = count_ktuples(orig_seq, k, ordered_words=ordered_words)

    # Collect counts ONLY for tuples present in the original (keeps current behavior)
    all_counts = {kt: [] for kt in orig_counts}

    for _ in range(B):

        if shuffle_method == 'biased':
            r = shuffle_preserve(orig_seq, k_node)
        else:
            r = kandel_uniform_k_shuffle(orig_seq, k_preserved)

        cr = count_ktuples(r, k, ordered_words=ordered_words)
        for kt in orig_counts:
            all_counts[kt].append(cr.get(kt, 0))

    # Compute z-scores and empirical p-values
    results = {}
    for kt, null_list in all_counts.items():
        obs = orig_counts[kt]
        mu = np.mean(null_list)
        sigma = np.std(null_list, ddof=1) if len(null_list) > 1 else 0.0
        z = (obs - mu) / sigma if sigma > 0 else 0.0

        ge = sum(1 for x in null_list if x >= obs)
        le = sum(1 for x in null_list if x <= obs)
        p_enrich = (1 + ge) / (B + 1)
        p_deplete = (1 + le) / (B + 1)
        p_two = min(1.0, 2 * min(p_enrich, p_deplete))

        results[kt] = {
            "obs": obs,
            "mu_null": mu,
            "sd_null": sigma,
            "z": z,
            "p_enrich": p_enrich,
            "p_deplete": p_deplete,
            "p_two": p_two
        }
    return results


def find_zscores(mean_observed, observed_counts, expected_probs, std_expected_dict,  observation_threshold = 7, with_space = 0):
    zscores_dict = {}
    for hyper in mean_observed.keys():
        if with_space == 0:
            if -1 in hyper:
                continue
        if len(hyper) == 1:
            continue
        if observed_counts[hyper] < observation_threshold:
            z_score = 0
        else:
            z_score = (mean_observed[hyper] - expected_probs[hyper])/std_expected_dict[hyper]
        zscores_dict[hyper] = z_score
    return zscores_dict


def positive_or_negative_scores_persize(hyper_dict, positive_score = 1, max_size = 4):
    per_size_dict = {size: {} for size in range(2,max_size+1)}
    for hyper, score in hyper_dict.items():
        if len(hyper) == 1:
                continue
        if positive_score == 1:
            if score >= 0:
                per_size_dict[len(hyper)][hyper] = score
        if positive_score == 0:
            if score < 0:
                per_size_dict[len(hyper)][hyper] = -score
    return per_size_dict


def observed_prob_and_count_for_batch(hyper_symbolic_sequence, spike_symbolic_sequence, max_size = 4, batch_num = 10):

    observed_prob_dict_for_batch = {}
    count_dict_for_batch = {}
    for batch_id in range(batch_num):
        total_len = len(hyper_symbolic_sequence)
        batch_length = int(total_len / batch_num)
        start_id = batch_id*batch_length
        stop_id = (batch_id+1)*batch_length
        if batch_length*batch_num < len(hyper_symbolic_sequence):
            if batch_id == batch_num - 1:
                stop_id = (batch_id+1)*batch_length + (len(hyper_symbolic_sequence) - (batch_length*batch_num))
        batch_hyper_sequence = hyper_symbolic_sequence[start_id:stop_id]
        batch_spike_sequence = spike_symbolic_sequence[start_id:stop_id]

        #### this is the observed probability:
        observed_prob_dict_for_batch[batch_id], count_dict_for_batch[batch_id] = observed_prob_and_count_directed_from_sequence(batch_hyper_sequence,
                                                                                                                                batch_spike_sequence,
                                                                                                                                max_size)
    return observed_prob_dict_for_batch, count_dict_for_batch


def expected_undirected_probability_all_batches_stats(observed_prob_dict_for_batch):

    batch_num = len(observed_prob_dict_for_batch.keys())

    expected_prob_for_batch_und = expected_undirected_probability_all_batches(observed_prob_dict_for_batch)

    std_expected_und_dict = {}
    for hyper, prob_batches in expected_prob_for_batch_und.items():
        ave = 0
        ave2 = 0
        for expected_freq in prob_batches.values():
            ave += expected_freq
            ave2 += expected_freq*expected_freq
        ave  /= (batch_num)
        ave2 /= (batch_num)
        if batch_num > 1:
            std_expected_und_dict[hyper] = np.sqrt((ave2 - (ave*ave))/(batch_num-1))
        else:
            std_expected_und_dict[hyper] = 0

    return expected_prob_for_batch_und, std_expected_und_dict


def expected_undirected_probability_all_batches(observed_prob_dict_for_batch):

    unique_hyper =  set(tuple(sorted(hyper)) for batch in observed_prob_dict_for_batch.values() for hyper in batch)
    unique_hyper_directed = set(hyper for batch in observed_prob_dict_for_batch.values() for hyper in batch)

    expected_prob_for_batch_und = {}
    for hyper in unique_hyper:
        expected_prob_for_batch_und[hyper] = {}

    for batch_id, batch in observed_prob_dict_for_batch.items():
        temp_expected_prob_dict_for_batch = expected_undirected_probability_single_batch(batch, unique_hyper_directed)
        for hyper, prob in temp_expected_prob_dict_for_batch.items():
            expected_prob_for_batch_und[hyper][batch_id] = prob
    return expected_prob_for_batch_und


def expected_undirected_probability_single_batch(prob_dict, full_hyper_list_dir):

    expected_und_dict = {}
    undirected_hyper = set(tuple(sorted(key)) for key in full_hyper_list_dir)

    for hyper in undirected_hyper:
        expected_und_dict[hyper] = 0
        permutations_list = tuple_permutations(hyper)
        permutations_list = set(permutations_list)

        for element in permutations_list:

            if len(element) == 1:
                expected_freq = prob_dict.get(element, 0)

            elif len(element) == 2:
                sub_event1 = (element[0],)
                sub_event2 = (element[1],)
                # Safely get the probabilities, defaulting to 0 to prevent KeyError
                expected_freq = prob_dict.get(sub_event1, 0) * prob_dict.get(sub_event2, 0)

            else:
                sub_event1 = tuple(element[0:-1])
                sub_event2 = tuple(element[1:])
                sub_event3 = tuple(element[1:-1])

                if prob_dict.get(sub_event3, 0) == 0:
                    expected_freq = 0
                else:
                    expected_freq = (
                        prob_dict.get(sub_event1, 0) *
                        prob_dict.get(sub_event2, 0) /
                        prob_dict[sub_event3]
                    )

            expected_und_dict[hyper] += expected_freq

    return expected_und_dict

def batch_statistics_undirected(batch_num, max_size, hyper_symbolic_sequence, spike_symbolic_sequence):

    ### note: observed_prob_dict_for_batch and expected_prob_dict_for_batch_und are structured in different way
    ### observed_prob_dict_for_batch is batch: hyper: observed_prob
    ### expected_prob_dict_for_batch_und is an alternative more convenient for fast computation version: hyper: batch: prob

    ##### the observed probability for directed hyperedges for the whole sequence
    observed_prob_dict, _ = observed_prob_and_count_directed_from_sequence(hyper_symbolic_sequence,
                                                                                    spike_symbolic_sequence,
                                                                                    max_size)
    ##### the expected probability for the udirected hyperedges computed on the full sequence:
    expected_full_dict_und = expected_undirected_probability(observed_prob_dict)

    ##### the observed probability for directed hyperedges for each batch
    observed_prob_dict_for_batch, count_dict_for_batch = observed_prob_and_count_for_batch(hyper_symbolic_sequence, spike_symbolic_sequence,
                                                                                                  max_size, batch_num)

    expected_prob_dict_for_batch_und, std_expected_und_dict = expected_undirected_probability_all_batches_stats(observed_prob_dict_for_batch)

    ### compute the total number of times each motif has been observed in each batch, and start computing the mean of the observed
    ### probability in each batch

    count_und_dict_for_hyper = {}
    mean_observed = {}
    for batch_id, batch in count_dict_for_batch.items():
        for hyper, count in batch.items():
            hyp_sort = tuple(sorted(hyper))
            try:
                count_und_dict_for_hyper[hyp_sort] += count
                mean_observed[hyp_sort] += observed_prob_dict_for_batch[batch_id][hyper]
            except:
                count_und_dict_for_hyper[hyp_sort] = count
                mean_observed[hyp_sort] = observed_prob_dict_for_batch[batch_id][hyper]

    ### divide the sum of the observed probabilities by the number of batches to obtain the mean
    for hyper in mean_observed.keys():
        mean_observed[hyper] /= batch_num


    mean_expected = {}
    for hyper, batches_prob in expected_prob_dict_for_batch_und.items():
            mean_expected[hyper] = np.sum([prob for prob in batches_prob.values()])/ batch_num

    return expected_full_dict_und, mean_expected, std_expected_und_dict, mean_observed, count_und_dict_for_hyper

def expected_directed_probability_all_batches(observed_prob_dict_for_batch):

    unique_hyper = {hyper for batch in observed_prob_dict_for_batch.values() for hyper in batch.keys()}

    expected_prob_for_batch_dir = {}
    for hyper in unique_hyper:
        expected_prob_for_batch_dir[hyper] = {}

    for batch_id, batch in observed_prob_dict_for_batch.items():
        temp_expected_prob_dict_for_batch = expected_directed_probability_single_batch(batch, unique_hyper)
        #temp_expected_prob_dict_for_batch = expected_directed_probability(batch)
        for hyper, prob in temp_expected_prob_dict_for_batch.items():
            expected_prob_for_batch_dir[hyper][batch_id] = prob
    return expected_prob_for_batch_dir

def expected_directed_probability_single_batch(prob_dict, full_hyper_list_dir):

    expected_dir_dict = {}
    directed_hyper = set(full_hyper_list_dir)

    for hyper in directed_hyper:
        expected_dir_dict[hyper] = 0

        if len(hyper) == 1:
            expected_freq = prob_dict.get(hyper, 0)

        elif len(hyper) == 2:
            sub_event1 = (hyper[0],)
            sub_event2 = (hyper[1],)
            # Safely get the probabilities, defaulting to 0 to prevent KeyError
            expected_freq = prob_dict.get(sub_event1, 0) * prob_dict.get(sub_event2, 0)

        else:
            sub_event1 = tuple(hyper[0:-1])
            sub_event2 = tuple(hyper[1:])
            sub_event3 = tuple(hyper[1:-1])

            if prob_dict.get(sub_event3, 0) == 0:
                expected_freq = 0
            else:
                expected_freq = (
                    prob_dict.get(sub_event1, 0) *
                    prob_dict.get(sub_event2, 0) /
                    prob_dict[sub_event3]
                )

        expected_dir_dict[hyper] = expected_freq

    return expected_dir_dict

def expected_directed_probability_all_batches_stats(observed_prob_dict_for_batch):

    batch_num = len(observed_prob_dict_for_batch.keys())

    expected_prob_for_batch_dir = expected_directed_probability_all_batches(observed_prob_dict_for_batch)

    ### for directed hyperedges:
    std_expected_dir_dict = {}
    for hyper, prob_batches in expected_prob_for_batch_dir.items():
        ave = 0
        ave2 = 0
        for expected_freq in prob_batches.values():
            ave += expected_freq
            ave2 += expected_freq*expected_freq
        ave  /= batch_num
        ave2 /= batch_num
        if batch_num > 1:
            std_expected_dir_dict[hyper] = np.sqrt((ave2 - (ave*ave))/(batch_num-1))
        else:
            std_expected_dir_dict[hyper] = 0

    return expected_prob_for_batch_dir, std_expected_dir_dict

def batch_statistics_directed(batch_num, max_size, hyper_symbolic_sequence, spike_symbolic_sequence):

    ### note: observed_prob_dict_for_batch and expected_prob_dict_for_batch_und are structured in different way
    ### observed_prob_dict_for_batch is batch: hyper: observed_prob
    ### expected_prob_dict_for_batch_und is an alternative more convenient for fast computation version: hyper: batch: prob

    ##### the observed probability for directed hyperedges for the whole sequence
    observed_prob_dict, _ = observed_prob_and_count_directed_from_sequence(hyper_symbolic_sequence,
                                                                                    spike_symbolic_sequence,
                                                                                    max_size)
    ##### the expected probability for the directed hyperedges computed on the full sequence:
    expected_full_dict_dir = expected_directed_probability(observed_prob_dict)

    ##### the observed probability for directed hyperedges for each batch
    observed_prob_dict_for_batch, count_dict_for_batch = observed_prob_and_count_for_batch(hyper_symbolic_sequence, spike_symbolic_sequence,
                                                                                                  max_size, batch_num)

    expected_prob_dict_for_batch_dir, std_expected_dir_dict = expected_directed_probability_all_batches_stats(observed_prob_dict_for_batch)

    ### compute the total number of times each motif has been observed in each batch, and start computing the mean of the observed
    ### probability in each batch

    count_dir_dict_for_hyper = {}
    mean_observed = {}
    for batch_id, batch in count_dict_for_batch.items():
        for hyper, count in batch.items():
            try:
                count_dir_dict_for_hyper[hyper] += count
                mean_observed[hyper] += observed_prob_dict_for_batch[batch_id][hyper]
            except:
                count_dir_dict_for_hyper[hyper] = count
                mean_observed[hyper] = observed_prob_dict_for_batch[batch_id][hyper]


    ### divide the sum of the observed probabilities by the number of batches to obtain the mean
    for hyper in mean_observed.keys():
        mean_observed[hyper] /= batch_num


    mean_expected = {}
    for hyper, batches_prob in expected_prob_dict_for_batch_dir.items():
            mean_expected[hyper] = np.sum([prob for prob in batches_prob.values()])/ batch_num

    return expected_full_dict_dir, mean_expected, std_expected_dir_dict, mean_observed, count_dir_dict_for_hyper



def batch_zscores_per_size(hyper_symbolic_sequence, spike_symbolic_sequence, max_size = 3, batch_num = 20,  positive_js = 1, observation_threshold = 0, with_space = 0, ordered_words = 0):
    """
    Batch z-score baseline: scores each tuple by comparing its observed probability
    with its expected probability across ``batch_num`` consecutive batches of the sequence.

    z = (mean observed probability - mean expected probability) / standard error
    of the expected probability across batches. Tuples observed fewer than
    ``observation_threshold`` times get z = 0.

    Parameters
    ----------
    hyper_symbolic_sequence, spike_symbolic_sequence : list
    max_size : int
        Largest tuple length.
    batch_num : int
    positive_js : {0, 1}
        1: keep over-represented tuples (z >= 0); 0: keep under-represented ones (as |z|).
    observation_threshold : int
    with_space : {0, 1}
        0: ignore tuples containing the space token -1.
    ordered_words : {0, 1}

    Returns
    -------
    dict[int, dict[tuple, float]]
        z-scores per tuple length (2..max_size).
    """
    if ordered_words == 0:
        _, expected_probs_batch, std_expected_dict_batch, mean_observed_batch, observed_counts_batch = batch_statistics_undirected(batch_num, max_size,
                                                                                                  hyper_symbolic_sequence, spike_symbolic_sequence)
    if ordered_words == 1:
        _, expected_probs_batch, std_expected_dict_batch, mean_observed_batch, observed_counts_batch = batch_statistics_directed(batch_num, max_size,
                                                                                                  hyper_symbolic_sequence, spike_symbolic_sequence)
    zscores_dict = find_zscores(mean_observed_batch, observed_counts_batch,
                                    expected_probs_batch, std_expected_dict_batch,
                                    observation_threshold = observation_threshold, with_space = with_space)
    zscores_per_size_dict = positive_or_negative_scores_persize(zscores_dict, positive_js, max_size)
    return zscores_per_size_dict


def observed_counts_and_expected_probs(hyper_symbolic_sequence, spike_symbolic_sequence, max_size=3, ordered_words = 0):
    """
    Counts every tuple (length 1..max_size) of consecutive symbols and computes
    its expected probability under the null model (first step of SYMBA).

    The expected probability of a k-tuple uses the observed probabilities of its
    sub-tuples: P(a)P(b) for pairs, and P(a..y)P(b..z)/P(b..y) for longer tuples.
    For unordered tuples the expectations of all distinct orderings are summed.

    Parameters
    ----------
    hyper_symbolic_sequence : list
        The symbolic sequence.
    spike_symbolic_sequence : list
        Event time of each symbol (only used to build the windows; same length).
    max_size : int
        Largest tuple length.
    ordered_words : {0, 1}
        0: tuples are unordered (keyed by their sorted symbols); 1: ordered.

    Returns
    -------
    observed_counts : dict[tuple, int]
    expected_probs : dict[tuple, float]
    """
    observed_directed_prob, observed_directed_counts = observed_prob_and_count_directed_from_sequence(hyper_symbolic_sequence,
                                                                                              spike_symbolic_sequence, max_size)
    if ordered_words == 0:
        observed_undirected_counts = converter_directed_to_undirected(observed_directed_counts)
        expected_undirected_prob = expected_undirected_probability(observed_directed_prob)
        return(observed_undirected_counts, expected_undirected_prob)
    if ordered_words == 1:
        expected_directed_prob = expected_directed_probability(observed_directed_prob)
        return(observed_directed_counts, expected_directed_prob)


def overlapping_hyperedges_spikes_of_size(hyper_symbolic_sequence, spike_symbolic_sequence, max_size=4):
    """
    Create dictionaries of overlapping hyperedges and their corresponding spike times for different sizes.

    Parameters:
    - hyper_symbolic_sequence: List of hyperedges
    - spike_symbolic_sequence: List of spike times
    - max_size: Maximum size of hyperedges

    Returns:
    - overlapping_hyperedges_of_size: Dictionary of overlapping hyperedges by size
    - spikes_hyperedges_of_size: Dictionary of spike times for hyperedges by size
    """
    overlapping_hyperedges_of_size = defaultdict(dict)
    spikes_hyperedges_of_size = defaultdict(dict)

    sequence_length = len(hyper_symbolic_sequence)
    if max_size > sequence_length:
        print(f"The maximum size possible is {sequence_length}. Changing max_size to {sequence_length}.")
        max_size = sequence_length

    for len_hyper in range(1, max_size + 1):
        for i in range(sequence_length - len_hyper + 1):
            hyper = tuple(hyper_symbolic_sequence[i:i + len_hyper])
            spikes = tuple(spike_symbolic_sequence[i:i + len_hyper])
            overlapping_hyperedges_of_size[len_hyper][i] = hyper
            spikes_hyperedges_of_size[len_hyper][i] = spikes

    return overlapping_hyperedges_of_size, spikes_hyperedges_of_size


def observed_prob_and_count_directed_from_sequence(hyper_symbolic_sequence, spike_symbolic_sequence, max_size):


    overlapping_hyperedges_of_size, _ = overlapping_hyperedges_spikes_of_size(hyper_symbolic_sequence,
                                                                                              spike_symbolic_sequence, max_size)

    observed_prob_dir, observed_counts_dir = observed_probability_and_count_from_hyperedges(overlapping_hyperedges_of_size)

    return observed_prob_dir, observed_counts_dir

def observed_probability_and_count_from_hyperedges(overlapping_hyperedges_of_size):
    """
    Computes observed probabilities and counts for hyperedges.

    Parameters:
    - overlapping_hyperedges_of_size: Dictionary of overlapping hyperedges by size

    Returns:
    - prob_dict: Dictionary of observed probabilities for each hyperedge
    - count_dict: Dictionary of counts for each hyperedge
    """

    # Initialize dictionaries for counts and probabilities
    count_dict = defaultdict(int)
    prob_dict = {}

    # Count occurrences of each hyperedge
    for dictionary in overlapping_hyperedges_of_size.values():
        for element in dictionary.values():
            count_dict[element] += 1

    # Calculate probabilities for each hyperedge
    for key in count_dict.keys():
        prob_dict[key] = count_dict[key] / len(overlapping_hyperedges_of_size[len(key)])

    return prob_dict, count_dict


def expected_undirected_probability(prob_dict):

    expected_und_dict = {}
    undirected_hyper = set(tuple(sorted(key)) for key in prob_dict.keys())


    for hyper in undirected_hyper:
        expected_und_dict[hyper] = 0
        permutations_list = tuple_permutations(hyper)
        permutations_list = set(permutations_list)

        for element in permutations_list:

            if len(element) == 1:
                expected_freq = prob_dict.get(element, 0)

            elif len(element) == 2:
                # safe: (a,) and (b,) must exist if (a,b) exists
                sub_event1 = (element[0],)
                sub_event2 = (element[1],)
                expected_freq = prob_dict[sub_event1] * prob_dict[sub_event2]

            else:
                sub_event1 = tuple(element[0:-1])
                sub_event2 = tuple(element[1:])
                sub_event3 = tuple(element[1:-1])

                if prob_dict.get(sub_event3, 0) == 0:
                    expected_freq = 0
                else:
                    expected_freq = (
                        prob_dict.get(sub_event1, 0) *
                        prob_dict.get(sub_event2, 0) /
                        prob_dict[sub_event3]
                    )

            expected_und_dict[hyper] += expected_freq


    return expected_und_dict


def expected_directed_probability(prob_dict):

    expected_dir_dict = {}
    for element in prob_dict.keys():
        if len(element) == 1:
            expected_freq = prob_dict[element]
        else :
            sub_event1 = tuple(element[0:-1])
            sub_event2 = tuple(element[1:])
            if len(element) > 2:
                sub_event3 = tuple(element[1:-1])
                if prob_dict[sub_event3] == 0:
                    expected_freq = 0
                else:
                    expected_freq = prob_dict[sub_event1]*prob_dict[sub_event2]/prob_dict[sub_event3]
            elif len(element) == 2:
                expected_freq = prob_dict[sub_event1]*prob_dict[sub_event2]
            else:
                print("something is wrong with this hyperedge ", element)
        expected_dir_dict[element] = expected_freq
    return expected_dir_dict
