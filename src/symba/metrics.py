"""Scoring, classification-performance (PR/ROC) and cross-method comparison metrics."""
import io
import math
import contextlib
import numpy as np
import scipy.stats as stats
from collections import defaultdict
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from .probability import empirical_ktuples, observed_counts_and_expected_probs
from .bayesian import js_score_per_size
from .shuffling import kandel_uniform_k_shuffle

__all__ = [
    "compute_qvalues",
    "compute_scores",
    "unordered_count",
    "get_subtuple_exceptions",
    "build_y_true_y_scores",
    "compute_pr_and_roc",
    "compute_pr_and_roc_subtuple_aware",
    "compare_symba_vs_shuffle",
    "symba_and_shuffle_scores",
    "shuffled_js_scores",
    "bjs_threshold_percentile",
    "bjs_threshold_fdr",
]


def compute_qvalues(results, p_label="p_enrich"):
    """
    Add Benjamini-Hochberg q-values to the results dictionary.
    """
    tuples = list(results.keys())
    pvals = np.array([results[kt][p_label] for kt in tuples])
    M = len(pvals)

    # Sort p-values
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]

    # Compute BH q-values
    qvals = np.empty(M)
    for i, p in enumerate(sorted_pvals):
        qvals[i] = p * M / (i + 1)

    # Enforce monotonicity (non-increasing when traversing from largest rank to smallest)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    qvals = np.clip(qvals, 0, 1)

    # Assign q-values back
    for idx, kt_idx in enumerate(sorted_indices):
        results[tuples[kt_idx]]["q_value"] = qvals[idx]

    return results


def compute_scores(
    orig_seq,
    size_max=3,
    B=100,
    score_type="z",
    with_space=0,
    positive_score=1,
    ordered_words=1,
    shuffle_method = 'non_biased'
):
    """
    Shuffle-based baseline: significance scores of all tuples of length 2..size_max,
    from B shuffles preserving the (k-1)-tuple counts (see empirical_ktuples).

    Parameters
    ----------
    orig_seq : list[int]
        Original input sequence (integers; can include -1).
    size_max : int
        Maximum tuple length (k goes from 2 to size_max).
    B : int, default=100
        Number of shuffled sequences for null distribution.
    score_type : {"z", "p_value", "q_value"}, default="z"
    with_space : int, default=0
        If 0, exclude tuples containing "-1"; if 1, include them.
    positive_score : int, default=1
        If 1: enrichment evidence (z >= 0, p_enrich, q from p_enrich).
        If 0: depletion evidence (z <= 0, |z|, p_deplete, q from p_deplete).
        If 2: two-sided (z raw, p_two, q from p_two).
    ordered_words : int, default=1
        If 1: tuples are ordered.
        If 0: tuples are keyed by their sorted multiset (unordered / order ignored).

    Returns
    -------
    dict_score : dict
        dict_score[k][tuple] = score
    results_all_k : dict
        Flat dict of all stats across k.
    """
    dict_score = {}
    results_all_k = {}

    for k in range(2, size_max + 1):
        k_preserved = k - 1
        # Use the (original) empirical_ktuples which only tracks tuples present in the original
        results = empirical_ktuples(
            orig_seq, k, k_preserved, B=B, ordered_words=ordered_words, shuffle_method = shuffle_method
        )

        # --- q-values if requested ---
        if score_type == "q_value":
            p_label = "p_enrich" if positive_score == 1 else (
                "p_deplete" if positive_score == 0 else "p_two"
            )
            results = compute_qvalues(results, p_label=p_label)

        dict_score[k] = {}

        for kt, stats_ in results.items():
            results_all_k[kt] = stats_
            # Skip tuples with "-1" if with_space=0
            if with_space == 0 and -1 in kt:
                continue

            if score_type == "z":
                if positive_score == 1 and stats_["z"] >= 0:
                    dict_score[k][kt] = stats_["z"]
                elif positive_score == 0 and stats_["z"] <= 0:
                    dict_score[k][kt] = abs(stats_["z"])
                elif positive_score == 2:
                    dict_score[k][kt] = stats_["z"]

            elif score_type == "p_value":
                if positive_score == 1:
                    dict_score[k][kt] = -np.log10(stats_["p_enrich"])
                elif positive_score == 0:
                    dict_score[k][kt] = -np.log10(stats_["p_deplete"])
                elif positive_score == 2:
                    dict_score[k][kt] = -np.log10(stats_["p_two"])

            elif score_type == "q_value":
                dict_score[k][kt] = -np.log10(stats_["q_value"])

    return dict_score, results_all_k


def unordered_count(N, k):
    """Number of unordered k-combinations with repetition from N items."""
    return math.comb(N + k - 1, k)


def get_subtuple_exceptions(words_in_sentence_per_size_dict, ordered_words=0):
    """
    For every motif in the ground truth, generate all sub-tuples of shorter length.
    These should not count as false positives even if not in the original dictionary.

    Returns
    -------
    exceptions : dict[length] = set of sub-tuples that are consequences of longer motifs
    """
    exceptions = defaultdict(set)

    for length, motifs in words_in_sentence_per_size_dict.items():
        for motif in motifs:
            # Generate all contiguous sub-tuples of all shorter lengths
            for sub_len in range(2, length):
                for start in range(len(motif) - sub_len + 1):
                    sub = tuple(motif[start:start + sub_len])
                    if ordered_words == 0:
                        sub = tuple(sorted(sub))
                    exceptions[sub_len].add(sub)

    return exceptions


def build_y_true_y_scores(score_dict, positive_motifs, motif_length, count2, count3, count4):
    """
    Builds the y_true / y_scores vectors used by both PR and ROC computations.

    Every method is evaluated on the same set of tuples, of size
    count2/count3/count4 for lengths 2/3/4 (normally: all distinct tuples of
    that length observed in the sequence). Tuples a method did not score
    (e.g. because it only reports over-represented tuples) are ranked below all
    tuples it did score; ground-truth motifs it did not score count as missed.
    """
    scored = score_dict.get(motif_length, {})
    positives = set(positive_motifs[motif_length])

    y_true   = [1 if motif in positives else 0 for motif in scored]
    y_scores = list(scored.values())

    # unscored tuples get a score below every scored tuple
    finite_scores = [s for s in y_scores if np.isfinite(s)]
    fill = min(finite_scores) - 1.0 if finite_scores else -1.0

    # missing positives
    for motif in positive_motifs[motif_length]:
        if motif not in scored:
            y_true.append(1)
            y_scores.append(fill)

    # pad negatives to reach total count
    count_map = {2: count2, 3: count3, 4: count4}
    total = count_map.get(motif_length, 0)
    for _ in range(total - len(y_scores)):
        y_true.append(0)
        y_scores.append(fill)

    return np.array(y_true), np.array(y_scores)


def compute_pr_and_roc(score_dict, positive_motifs, motif_length, count2, count3, count4):
    """
    Computes PR and ROC curves and all derived metrics from a single y_true/y_scores pair.

    Parameters
    ----------
    score_dict : dict[int, dict[tuple, float]]
        Scores per tuple length (higher = more significant).
    positive_motifs : dict[int, list[tuple]]
        Ground-truth motifs per length.
    motif_length : int
        Length to evaluate.
    count2, count3, count4 : int
        Size of the evaluation set for each length (see build_y_true_y_scores).

    Returns
    -------
    dict with keys:
        y_true, y_scores,
        precision_curve, recall_curve, pr_thresholds,
        auc_pr, best_f1, best_precision, best_recall,
        fpr_curve, tpr_curve, roc_thresholds,
        auc_roc
    """
    y_true, y_scores = build_y_true_y_scores(
        score_dict, positive_motifs, motif_length, count2, count3, count4
    )

    result = {
        'y_true':   y_true,
        'y_scores': y_scores,
    }

    # guard
    if len(y_true) == 0 or sum(y_true) == 0:
        result.update({
            'precision_curve': None, 'recall_curve': None, 'pr_thresholds': None,
            'auc_pr': 0.0, 'best_f1': 0.0, 'best_precision': 0.0, 'best_recall': 0.0,
            'fpr_curve': None, 'tpr_curve': None, 'roc_thresholds': None,
            'auc_roc': 0.0,
        })
        return result

    # PR
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    if len(f1_scores) > 0:
        best_idx      = np.argmax(f1_scores)
        best_f1       = f1_scores[best_idx]
        best_precision = precision[best_idx]
        best_recall    = recall[best_idx]
    else:
        best_f1 = best_precision = best_recall = 0.0

    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    result.update({
        'precision_curve': precision,
        'recall_curve':    recall,
        'pr_thresholds':   pr_thresholds,
        'auc_pr':          auc(recall, precision),
        'best_f1':         best_f1,
        'best_precision':  best_precision,
        'best_recall':     best_recall,
        'fpr_curve':       fpr,
        'tpr_curve':       tpr,
        'roc_thresholds':  roc_thresholds,
        'auc_roc':         auc(fpr, tpr),
    })

    return result


def compute_pr_and_roc_subtuple_aware(
    score_dict, words_in_sentence_per_size_dict, length,
    count2, count3, count4,
    ordered_words=0, exclude_subtuple_fp=False, observed_tuples=None
):
    """
    compute_pr_and_roc with optional exclusion of "ambiguous" tuples.

    If a longer motif such as (3, 7, 12) is planted, its sub-tuples (3, 7) and
    (7, 12) are over-represented too, so reporting them is not really a false
    positive. With exclude_subtuple_fp=True, these sub-tuples (unless they are
    themselves ground-truth motifs) are removed from the evaluation, and the
    evaluation-set size is reduced by the same amount for every method.

    Parameters
    ----------
    score_dict, words_in_sentence_per_size_dict, length, count2, count3, count4
        As in compute_pr_and_roc.
    ordered_words : {0, 1}
    exclude_subtuple_fp : bool
    observed_tuples : iterable of tuples, optional
        All tuples observed in the sequence (e.g. the keys of the observed
        counts). Only observed ambiguous sub-tuples are removed from the
        evaluation set; if omitted, all of them are assumed to be observed.
    """
    if not exclude_subtuple_fp:
        return compute_pr_and_roc(
            score_dict, words_in_sentence_per_size_dict,
            length, count2, count3, count4
        )

    exceptions = get_subtuple_exceptions(words_in_sentence_per_size_dict, ordered_words)
    exceptions_at_length = exceptions.get(length, set())

    if not exceptions_at_length:
        return compute_pr_and_roc(
            score_dict, words_in_sentence_per_size_dict,
            length, count2, count3, count4
        )

    ground_truth = set(words_in_sentence_per_size_dict.get(length, []))
    ambiguous = exceptions_at_length - ground_truth
    if observed_tuples is not None:
        observed_at_length = {
            tuple(sorted(t)) if ordered_words == 0 else tuple(t)
            for t in observed_tuples if len(t) == length
        }
        ambiguous &= observed_at_length

    filtered_score_dict = defaultdict(dict)
    for l, motif_scores in score_dict.items():
        if l != length:
            filtered_score_dict[l] = motif_scores
        else:
            for motif, score in motif_scores.items():
                motif_key = tuple(sorted(motif)) if ordered_words == 0 else motif
                if motif_key not in ambiguous:
                    filtered_score_dict[l][motif] = score

    counts = {2: count2, 3: count3, 4: count4}
    counts[length] -= len(ambiguous)

    return compute_pr_and_roc(
        filtered_score_dict, words_in_sentence_per_size_dict,
        length, counts[2], counts[3], counts[4]
    )


def compare_symba_vs_shuffle(js_dict, p_dict_per_size, z_dict_per_size, lengths=[2, 3, 4],
                             verbose=True):
    """
    Spearman correlation between SYMBA scores and shuffle scores, length by length.

    Parameters
    ----------
    js_dict          : dict[length][tuple] = symba score
    p_dict_per_size  : dict[length][tuple] = -log10(p_enrich) score
    z_dict_per_size  : dict[length][tuple] = z score
    lengths          : list of tuple lengths to compare
    verbose          : if True, print the results of each length

    Returns
    -------
    results : dict[length] = {"n_common_tuples",
                              "spearman_symba_p", "pval_spearman_symba_p",
                              "spearman_symba_z", "pval_spearman_symba_z"}
        computed on the tuples scored by all three methods
    """
    results = {}

    for length in lengths:
        # Get common tuples across all three dicts
        common_tuples = list(
            set(js_dict.get(length, {}).keys()) &
            set(p_dict_per_size.get(length, {}).keys()) &
            set(z_dict_per_size.get(length, {}).keys())
        )

        if len(common_tuples) < 2:
            print(f"Length {length}: not enough common tuples, skipping.")
            continue

        symba_scores = np.array([js_dict[length][t]         for t in common_tuples])
        p_scores     = np.array([p_dict_per_size[length][t] for t in common_tuples])
        z_scores     = np.array([z_dict_per_size[length][t] for t in common_tuples])

        spearman_p, pval_spearman_p = stats.spearmanr(symba_scores, p_scores)
        spearman_z, pval_spearman_z = stats.spearmanr(symba_scores, z_scores)

        results[length] = {
            "n_common_tuples": len(common_tuples),
            "spearman_symba_p":  spearman_p, "pval_spearman_symba_p":  pval_spearman_p,
            "spearman_symba_z":  spearman_z, "pval_spearman_symba_z":  pval_spearman_z,
        }

        if verbose:
            print(f"\nLength {length} ({len(common_tuples)} tuples)")
            for k, v in results[length].items():
                if k != "n_common_tuples":
                    print(f"  {k}: {v:.2e}" if "pval" in k else f"  {k}: {v:.3f}")

    return results


def symba_and_shuffle_scores(hyper_symbolic_sequence, spike_symbolic_sequence, max_size=3,
                             number_random_shuffles=1000, smooth_prior=1., with_space=0,
                             quantiles_list=(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)):
    """
    SYMBA scores and shuffle scores (unordered tuples of length 2 to max_size),
    with the settings used for the real-data examples of the paper.

    Returns
    -------
    js_scores : dict[length][tuple] = SYMBA score (d_JS)
    p_scores  : dict[length][tuple] = -log10(p_enrich) of the shuffle test (higher = more enriched)
    z_scores  : dict[length][tuple] = z-score of the shuffle test
    """
    observed_counts, expected_probs = observed_counts_and_expected_probs(
        hyper_symbolic_sequence, spike_symbolic_sequence, max_size=max_size, ordered_words=0)
    js_scores = js_score_per_size(
        hyper_symbolic_sequence, observed_counts, expected_probs, max_size=max_size, positive_js=1,
        smooth_prior=smooth_prior, number_of_points=0, quantiles_list=list(quantiles_list),
        with_space=with_space, measure='js_distance')
    _, results_all_k = compute_scores(
        hyper_symbolic_sequence, size_max=max_size, B=number_random_shuffles,
        with_space=with_space, ordered_words=0, shuffle_method='non_biased')
    p_scores, z_scores = defaultdict(dict), defaultdict(dict)
    for motif, res in results_all_k.items():
        if with_space == 0 and -1 in motif:
            continue
        p_scores[len(motif)][motif] = -np.log10(res["p_enrich"])
        z_scores[len(motif)][motif] = res["z"]
    return js_scores, p_scores, z_scores


def shuffled_js_scores(hyper_symbolic_sequence, max_size=3, n_shuffles=10, ordered_words=0,
                       smooth_prior=1., with_space=0,
                       quantiles_list=(0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)):
    """
    SYMBA scores of the k-tuples in shuffled copies of the sequence, for k = 2..max_size.

    For each k the sequence is shuffled preserving the counts of all (k-1)-tuples
    (as in the shuffle method, see compute_scores). The shuffled sequences satisfy
    the null model of SYMBA for the k-tuples, so any high score in them is a
    false positive. Used by bjs_threshold_percentile and bjs_threshold_fdr.

    Parameters
    ----------
    hyper_symbolic_sequence : list
        The symbolic sequence.
    max_size : int
        Largest tuple length.
    n_shuffles : int
        Number of shuffled sequences per tuple length (about 10 is enough).
    ordered_words : {0, 1}
        As in observed_counts_and_expected_probs.
    smooth_prior, with_space, quantiles_list
        As in js_score_per_size.

    Returns
    -------
    dict[int, list[np.ndarray]]
        For each length, one array of scores (over-represented tuples) per shuffled sequence.
    """
    shuffled_scores = {}
    for k in range(2, max_size + 1):
        shuffled_scores[k] = []
        for _ in range(n_shuffles):
            shuffled = kandel_uniform_k_shuffle(list(hyper_symbolic_sequence), k - 1)
            # js_score_per_size prints the total expected probability of each length
            with contextlib.redirect_stdout(io.StringIO()):
                observed_counts, expected_probs = observed_counts_and_expected_probs(
                    shuffled, list(range(len(shuffled))), max_size=k, ordered_words=ordered_words)
                js = js_score_per_size(
                    shuffled, observed_counts, expected_probs, max_size=k, positive_js=1,
                    smooth_prior=smooth_prior, number_of_points=0, quantiles_list=list(quantiles_list),
                    with_space=with_space, measure='js_distance')
            shuffled_scores[k].append(np.array(
                [score for tup, score in js.get(k, {}).items() if with_space or -1 not in tup]))
    return shuffled_scores


def bjs_threshold_percentile(hyper_symbolic_sequence, max_size=3, n_shuffles=10, percentile=95,
                             ordered_words=0, shuffled_scores=None, **score_kwargs):
    """
    BJS threshold of each tuple length from the maximum score in shuffled sequences.

    The threshold is the given percentile, over the shuffled sequences, of the
    maximum SYMBA score of the k-tuples in each shuffled sequence. With
    percentile = 95, the probability that at least one k-tuple exceeds the
    threshold by chance is about 5% (conservative choice).

    Parameters
    ----------
    hyper_symbolic_sequence : list
        The symbolic sequence.
    max_size, n_shuffles, ordered_words
        As in shuffled_js_scores.
    percentile : float
        Percentile (0-100) of the maximum scores.
    shuffled_scores : dict, optional
        Output of shuffled_js_scores, to reuse the same shuffles (e.g. for bjs_threshold_fdr).
    **score_kwargs
        smooth_prior, with_space, quantiles_list, passed to shuffled_js_scores.

    Returns
    -------
    dict[int, float]
        Threshold per tuple length.
    """
    if shuffled_scores is None:
        shuffled_scores = shuffled_js_scores(hyper_symbolic_sequence, max_size=max_size, n_shuffles=n_shuffles,
                                             ordered_words=ordered_words, **score_kwargs)
    return {k: float(np.percentile([s.max() if len(s) else 0. for s in scores_k], percentile))
            for k, scores_k in shuffled_scores.items()}


def bjs_threshold_fdr(js_scores, hyper_symbolic_sequence, max_size=3, n_shuffles=10, fdr=0.05,
                      ordered_words=0, shuffled_scores=None,
                      thresholds=np.round(np.arange(0.01, 0.84, 0.01), 2), **score_kwargs):
    """
    BJS threshold of each tuple length controlling the false discovery rate.

    The threshold is the smallest value t for which the mean number of k-tuples
    with score > t in the shuffled sequences is at most ``fdr`` times the number
    of k-tuples with score > t in the original sequence, i.e. the false positives
    are expected to be at most a fraction ``fdr`` of the detected tuples
    (less conservative than bjs_threshold_percentile).

    Parameters
    ----------
    js_scores : dict[int, dict[tuple, float]]
        SYMBA scores of the original sequence (output of js_score_per_size, with
        the same ordered_words and score settings).
    hyper_symbolic_sequence : list
        The symbolic sequence.
    max_size, n_shuffles, ordered_words
        As in shuffled_js_scores.
    fdr : float
        Target false discovery rate.
    shuffled_scores : dict, optional
        Output of shuffled_js_scores, to reuse the same shuffles.
    thresholds : array
        Candidate thresholds (the score is at most sqrt(ln 2) ~ 0.83).
    **score_kwargs
        smooth_prior, with_space, quantiles_list, passed to shuffled_js_scores.

    Returns
    -------
    dict[int, float]
        Threshold per tuple length (nan if no candidate reaches the target).
    """
    if shuffled_scores is None:
        shuffled_scores = shuffled_js_scores(hyper_symbolic_sequence, max_size=max_size, n_shuffles=n_shuffles,
                                             ordered_words=ordered_words, **score_kwargs)
    thresholds = np.asarray(thresholds)
    fdr_thresholds = {}
    for k, scores_k in shuffled_scores.items():
        original = np.array([score for tup, score in js_scores.get(k, {}).items()
                             if score_kwargs.get('with_space', 0) or -1 not in tup])
        n_original = (original[None, :] > thresholds[:, None]).sum(axis=1)
        n_shuffled = np.mean([(s[None, :] > thresholds[:, None]).sum(axis=1) for s in scores_k], axis=0)
        ok = np.where((n_original > 0) & (n_shuffled <= fdr * n_original))[0]
        fdr_thresholds[k] = float(thresholds[ok[0]]) if len(ok) else np.nan
    return fdr_thresholds
