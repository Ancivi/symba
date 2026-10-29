# SYMBA

Supplementary code for the SYMBA paper (full citation to be added).

SYMBA finds over-represented motifs (groups of symbols that occur close together) in
symbolic sequences. For each tuple of consecutive symbols it compares a Beta prior, built
from the tuple's expected probability under a null model, with the Beta posterior after
observing the tuple's count, and scores the tuple by the Jensen–Shannon distance between
the two. The repository contains the method (`src/symba/`) and the notebooks that
reproduce the analyses and figures of the paper, including the comparison with
shuffle-based (p-value, z-score) and batch z-score baselines.

## Repository structure

```
.
├── src/symba/              the SYMBA method and supporting code (installable package)
│   ├── sequences.py         symbolic sequences: synthetic data, events → sequence, Markov chains
│   ├── noise.py             noise models for synthetic sequences
│   ├── probability.py       observed counts, expected probabilities, batch z-score baseline
│   ├── bayesian.py          SYMBA scores (Beta prior/posterior, Jensen–Shannon distance)
│   ├── shuffling.py         sequence shuffles used by the shuffle baseline
│   ├── metrics.py           shuffle baseline scores, PR/ROC evaluation, method comparison, BJS thresholds from shuffles
│   ├── plotting.py          figures
│   ├── datasets.py          Enron email parsing helpers
│   └── utils.py             set_seed
├── notebooks/
│   ├── demo_and_figures.ipynb                analyses and figures of the paper
│   └── brain_neuropixel_preprocessing.ipynb  builds the brain symbolic sequence from Allen Institute data
├── data/                   small results shipped with the repository; data/README.md lists the rest
├── figures/                reference copies of the figures in the paper
├── outputs/                figures written when running the notebooks (not tracked by git)
├── requirements.txt        tested package versions (Python 3.14)
├── pyproject.toml
├── requirements-brain-preprocessing.txt   separate environment for the brain preprocessing notebook
└── setup_brain_preprocessing_env.sh       installs that environment
```

## Installation

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions the code was tested with (Python 3.14).
`pip install -e .` alone installs the package with unpinned dependencies; add the `stock`
extra (`pip install -e ".[stock]"`) for the stock-market case study, which needs `yfinance`.
The notebooks also work without installing the package: their first cell adds `src/` to
the Python path and moves to the repository root, whether they are opened from
`notebooks/` or from the root.

The brain preprocessing notebook needs a separate environment (Python 3.11, `allensdk`),
which conflicts with the one above: see [Brain data](#brain-data).

## Reproducing the analyses

Run `notebooks/demo_and_figures.ipynb` from top to bottom. It reproduces the figures of the
main text and a selection of the supplementary ones, as worked examples of the method
(figures combining several panels were assembled from the individual plots). With the
default settings this takes about 10 minutes: the long simulations are not rerun, and their
figures are made from the results shipped in `data/`. Figures are saved to `outputs/figures/`.

| Notebook section | Data | Rerun the simulations with | Time |
|---|---|---|---|
| Robustness to symbol noise (substitution and deletion) | `data/aggregated_results_*` | `RECOMPUTE_NOISE = True` | seconds (sweep: ~20 min) |
| Convergence of the shuffle null model to the analytical expected probability | computed | — | ~3 min |
| Classification performance: SYMBA vs shuffle (p-value, z-score) vs batch z-score | `data/four_methods_summary_*` | `RECOMPUTE_FOUR_METHODS = True` | seconds (sweep: several hours) |
| Markov-generated sequences | `data/markovsequences/` | `RECOMPUTE_MARKOV = True` | seconds |
| Stock market (unsigned and signed) | downloaded from Yahoo Finance at run time | — | ~2 min |
| Enron emails | `data/emails.csv` (Kaggle, see `data/README.md`) | — | ~2 min |
| Brain areas (Allen Institute Neuropixels) | `data/sequences_session29_…_brainareas_multinomial.csv` | — | ~1.5 min |

`RECOMPUTE = True` in the first cell reruns every simulation.

The brain section analyses 100,000 consecutive spikes of the recording, to keep the comparison with the shuffle
method feasible in time; the significant tuples reported in the paper are computed on the whole sequence
(about 40 million spikes), in the last cell of that section.

In the stock-market section, an event is recorded for a stock on every day its closing price changes by more
than 2% (unsigned), or separately for rises and drops (signed).

### Reproducibility

- All random steps (synthetic sequences, noise, shuffles, and the order of events that occur
  at the same time) are seeded with `SEED` in the first cell, so the same data and seed give
  the same results. The results in the paper were obtained before seeding was added, so they
  are reproduced statistically, not digit for digit.
- The stock prices are downloaded when the notebook runs; the results in the paper use the download of
  2 October 2026. Yahoo Finance occasionally revises historical data, and events on the same day are put in
  random order, so with a later download the number of significant tuples can vary slightly; the strongest
  motifs are stable.
- Enron emails: each email is counted once (the dataset stores copies of the same email in
  several folders of the sender's mailbox) and ordered by its timestamp.

## Brain data

`notebooks/brain_neuropixel_preprocessing.ipynb` builds the brain-area symbolic sequence
used by the brain section from the Allen Institute Neuropixels Visual Coding data:

1. Set up its environment: `./setup_brain_preprocessing_env.sh` (conda instructions in the
   script header). It pins `pynwb`/`hdmf` versions that can read the Allen Institute files.
2. Set `ALLOW_DOWNLOAD = True` in the notebook's setup cell and run it once to download the
   metadata and the session used in the paper (~2.5 GB, not the whole ~130 GB dataset); then
   set it back to `False`.
3. Run the notebook.

Details, and a known AllenSDK issue this setup protects against, are in `data/README.md`.

## Using SYMBA on your own data

```python
import symba

sequence = [...]                    # list of symbols (ints or strings); -1 can mark gaps
times = list(range(len(sequence)))  # one event time per symbol

quantiles = [0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]  # as in the notebook
observed, expected = symba.observed_counts_and_expected_probs(
    sequence, times, max_size=3, ordered_words=0)   # ordered_words=1: order within a motif matters
scores = symba.js_score_per_size(
    sequence, observed, expected, max_size=3, quantiles_list=quantiles)

significant_pairs = {t: s for t, s in scores[2].items() if s >= 0.6}
```

`scores[k]` maps each over-represented tuple of length `k` to its SYMBA score. To build a
sequence from timestamped events (a table indexed by time with a `unit_id` column), use
`symba.events_to_symbolic_sequences`.

## License and citation

To be added.
