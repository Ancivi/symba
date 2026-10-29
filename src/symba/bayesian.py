"""Bayesian (Beta prior/posterior) significance scoring via JS/KL divergence."""
import numpy as np
from scipy.special import rel_entr, betaln, betaincinv, digamma

from .probability import positive_or_negative_scores_persize

__all__ = [
    "bayesian_method",
    "js_score_per_size",
    "construct_beta_prior",
    "construct_beta_posterior",
    "js_continuous_fast",
    "kl_beta_exact",
    "fast_beta_pdf",
    "score_motifs",
]


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def bayesian_method(hyper_symbolic_sequence, observed_counts, expected_probs,
                    max_size = 3,
                    smooth_prior=1,
                    number_of_points = 0,
                    quantiles_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                    with_space = 0, measure='js_distance'):
    """
    SYMBA score of every tuple: builds the Beta prior (from the expected
    probability) and the Beta posterior (after observing the counts) and
    measures how far apart they are.

    Returns
    -------
    dict[tuple, float]
        Signed score per tuple (positive = over-represented), sorted in
        decreasing order. See score_motifs for the meaning of ``measure``.
    """
    len_symb_sequence = len(hyper_symbolic_sequence)

    priors = construct_beta_prior(expected_probs, len_symb_sequence, smoothing_param = smooth_prior, max_size = max_size)

    posteriors = construct_beta_posterior(priors, observed_counts, len_symb_sequence)

    scores = score_motifs(priors, posteriors,
                          with_space = with_space,
                          number_of_points = number_of_points,
                          quantiles = np.array(quantiles_list),
                          measure=measure
                          )

    return scores


def js_score_per_size(hyper_symbolic_sequence, observed_counts, expected_probs, max_size = 3, positive_js = 1,
                    smooth_prior=1,
                    number_of_points = 0,
                    quantiles_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                    with_space = 0, measure='js_distance'):
    """
    SYMBA scores grouped by tuple length (second step of SYMBA, after
    observed_counts_and_expected_probs).

    Parameters
    ----------
    hyper_symbolic_sequence : list
        The symbolic sequence (only its length is used).
    observed_counts, expected_probs : dict
        Output of observed_counts_and_expected_probs.
    max_size : int
        Largest tuple length.
    positive_js : {0, 1}
        1: keep over-represented tuples (score >= 0); 0: keep under-represented ones (as |score|).
    smooth_prior : float
        Pseudo-count added to both Beta prior parameters.
    number_of_points : int
        0: integrate on a grid made of the ``quantiles_list`` quantiles of prior
        and posterior (plus 10 points between their means); >0: use that many
        evenly spaced points per distribution.
    quantiles_list : list of float
    with_space : {0, 1}
        0: ignore tuples containing the space token -1.
    measure : {'js_distance', 'kl_divergence'}

    Returns
    -------
    dict[int, dict[tuple, float]]
        Scores per tuple length (2..max_size).
    """
    js_hyper_dict = bayesian_method(hyper_symbolic_sequence, observed_counts, expected_probs,
                                        max_size = max_size,
                                        smooth_prior = smooth_prior,
                                        number_of_points = number_of_points,
                                        quantiles_list = quantiles_list, with_space = with_space,
                                        measure=measure)


    for i in range(1,max_size+1):
        total_exp_success_prob = np.sum(list(prob for key, prob in expected_probs.items() if len(key) == i))
        print("Total expected probability for size ", i, ": ", total_exp_success_prob)

    js_per_size_dict = positive_or_negative_scores_persize(js_hyper_dict, positive_js, max_size = max_size)


    return js_per_size_dict


def construct_beta_prior(expected_probs, len_symb_sequence, smoothing_param = 1, max_size = 3):
    """
    Beta prior of each tuple's occurrence probability, centred on its expected probability.

    For a k-tuple with expected probability p_exp in a sequence of length L
    (so n = L - k + 1 windows):
        alpha = p_exp * n + smoothing_param
        beta  = (1 - p_exp) * n + smoothing_param

    Parameters
    ----------
    expected_probs : dict[tuple, float]
    len_symb_sequence : int
    smoothing_param : float
        Pseudo-count added to both parameters (must be >= 0).
    max_size : int
        Unused; kept for backward compatibility of the call signature.

    Returns
    -------
    dict[tuple, tuple[float, float]]
        (alpha, beta) per tuple.
    """
    if smoothing_param < 0:
        print("smoothing parameter in the prior is negative, are you sure?")

    priors = {}
    for key, p_expected in expected_probs.items():
        alpha = p_expected*(len_symb_sequence - len(key) + 1)
        beta = (1 - p_expected)*(len_symb_sequence - len(key) + 1)
        alpha += smoothing_param
        beta += smoothing_param

        priors[key]=(alpha, beta)

    return priors

def construct_beta_posterior(priors, observed_counts, len_symb_sequence):
    """
    Beta posterior of each tuple, given a Binomial likelihood: the tuple was
    observed ``observed_counts[key]`` times out of L - k + 1 windows.

    Returns
    -------
    dict[tuple, tuple[float, float]]
        (alpha, beta) per tuple.
    """
    posteriors = {}

    for key, (alpha_prior, beta_prior) in priors.items():
        ### Assuming Binomial likelihood
        ### using the observed
        trials = len_symb_sequence - len(key) + 1
        successes = observed_counts[key]
        failures = trials - successes
        #### If the prior is a Beta function, given the Binomial likelihood, also the posterior would be a Beta function
        # Posterior parameters
        alpha_posterior = alpha_prior + successes
        beta_posterior = beta_prior + failures
        posteriors[key] = (alpha_posterior, beta_posterior)

    return posteriors

def js_continuous_fast(alpha_p, beta_p, alpha_q, beta_q, x):
    """
    Jensen-Shannon divergence between Beta(alpha_p, beta_p) and
    Beta(alpha_q, beta_q), integrated numerically (trapezoid rule) on the grid x.
    Both densities are renormalised on the grid before integrating.
    """
    # 1. Evaluate densities
    p = fast_beta_pdf(x, alpha_p, beta_p)
    q = fast_beta_pdf(x, alpha_q, beta_q)

    # 2. Normalize
    p /= _trapz(p, x)
    q /= _trapz(q, x)

    # 3. Calculate mixture
    m = 0.5 * (p + q)

    # 4. Compute JS divergence (rel_entr handles zeros)
    kl_pm = _trapz(rel_entr(p, m), x)
    kl_qm = _trapz(rel_entr(q, m), x)

    return 0.5 * (kl_pm + kl_qm)


def kl_beta_exact(alpha_p, beta_p, alpha_q, beta_q):
    """
    Computes exact KL(P || Q) for P=Beta(alpha_p, beta_p) and Q=Beta(alpha_q, beta_q)
    No grids or numerical integration required.
    """
    kl = (betaln(alpha_q, beta_q) - betaln(alpha_p, beta_p) +
          (alpha_p - alpha_q) * digamma(alpha_p) +
          (beta_p - beta_q) * digamma(beta_p) +
          (alpha_q - alpha_p + beta_q - beta_p) * digamma(alpha_p + beta_p))
    return kl


def fast_beta_pdf(x, a, b):
    """
    Computes the Beta PDF manually to bypass scipy.stats overhead.
    Uses log-space to ensure numerical stability.
    """
    # Clip x just slightly off 0 and 1 to prevent log(0) warnings
    x_safe = np.clip(x, 1e-300, 1.0 - 1e-15)

    # PDF = exp( (a-1)*log(x) + (b-1)*log(1-x) - log(Beta(a,b)) )
    log_pdf = (a - 1) * np.log(x_safe) + (b - 1) * np.log(1 - x_safe) - betaln(a, b)

    return np.exp(log_pdf)

def score_motifs(priors, posteriors,
                 with_space=0,
                 number_of_points=0,
                 quantiles=np.array([0.005, 0.05, 0.25, 0.5, 0.75, 0.95, 0.995]),
                 measure='js_distance'):
    """
    Scores every tuple by the divergence between its Beta prior and posterior.

    All tuples are returned (not only the significant ones); apply a threshold
    to the scores to select significant tuples.

    Parameters:
    - priors:            Dictionary of prior Beta distribution parameters (alpha, beta)
    - posteriors:        Dictionary of posterior Beta distribution parameters (alpha, beta)
    - with_space:        Flag to include hyperedges with space (-1)
    - number_of_points:  Number of points for numerical integration (0 = use quantiles)
    - quantiles:         Quantiles for Beta distribution
    - measure:           'js_distance' (default) or 'kl_divergence'
                         'js_distance'  -> signed sqrt(JSD(prior, posterior))
                         'kl_divergence'-> signed KL(posterior || prior),
                                          i.e. surprise about posterior assuming prior
                         The sign is + if the posterior mean exceeds the prior mean
                         (over-represented tuple), - otherwise.

    Returns:
    - Dictionary of scores per tuple, sorted in decreasing order
    """
    result_dictionary = {}

    for key, (alpha_prior, beta_prior) in priors.items():
        if with_space == 0 and -1 in key:
            continue

        alpha_posterior, beta_posterior = posteriors[key]

        if number_of_points == 0:
            x_prior = betaincinv(alpha_prior, beta_prior, quantiles)
            x_posterior = betaincinv(alpha_posterior, beta_posterior, quantiles)

            # Drop 10 points evenly between the two peaks to avoid numerical issues when the two distributions are far apart

            mean_prior = alpha_prior / (alpha_prior + beta_prior)
            mean_posterior = alpha_posterior / (alpha_posterior + beta_posterior)
            x_bridge = np.linspace(min(mean_prior, mean_posterior),
                                   max(mean_prior, mean_posterior), 10)

            x = np.unique(np.concatenate((x_prior, x_posterior, x_bridge)))

        else:
            a = betaincinv(alpha_prior, beta_prior, 0.001)
            b = betaincinv(alpha_prior, beta_prior, 0.999)
            x_prior = np.linspace(a, b, number_of_points)
            a = betaincinv(alpha_posterior, beta_posterior, 0.001)
            b = betaincinv(alpha_posterior, beta_posterior, 0.999)
            x_posterior = np.linspace(a, b, number_of_points)

            x = np.unique(np.concatenate((x_prior, x_posterior)))

        mean_prior = alpha_prior / (alpha_prior + beta_prior)
        mean_posterior = alpha_posterior / (alpha_posterior + beta_posterior)
        sign = np.sign(mean_posterior - mean_prior)

        if measure == 'js_distance':
            js_div = js_continuous_fast(alpha_prior, beta_prior,
                                        alpha_posterior, beta_posterior,
                                        x)
            value = np.sqrt(js_div) * sign

        elif measure == 'kl_divergence':
            ### KL(posterior || prior): exact closed-form solution (no grid 'x' needed)
            kl = kl_beta_exact(alpha_posterior, beta_posterior,
                               alpha_prior, beta_prior)
            value = kl * sign

        else:
            raise ValueError(f"Unknown measure '{measure}'. "
                             f"Choose 'js_distance' or 'kl_divergence'.")

        result_dictionary[key] = value

    sorted_result_dictionary = dict(
        sorted(result_dictionary.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_result_dictionary
