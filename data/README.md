# Data

## Files shipped with the repository

Small derived results, so that the notebook makes its figures without rerunning the
long simulations (see `RECOMPUTE` in the first cell of `notebooks/demo_and_figures.ipynb`):

| File | Size | Used by (section of `demo_and_figures.ipynb`) |
|---|---|---|
| `four_methods_summary_alph30_2w20x10_3w20x10_4w20x10_gamma1.0_ord1_nruns20_B1000.pkl` | 0.6 MB | Four-method comparison, main-text configuration (20 distinct ordered motifs of length 2, 3 and 4, each repeated 10 times; 20 runs) |
| `four_methods_summary_alph30_2w0x0_3w20x10_4w0x0_gamma1.0_ord1_nruns50_B1000.pkl` | 0.4 MB | Four-method comparison, supplementary configuration (only length-3 motifs, 10 repetitions; 50 runs) |
| `four_methods_summary_alph30_2w0x0_3w20x5_4w0x0_gamma1.0_ord1_nruns50_B1000.pkl` | 0.4 MB | Four-method comparison, supplementary configuration (only length-3 motifs, 5 repetitions; 50 runs) |
| `four_methods_thresholded_alph30_2w20x10_3w20x10_4w20x10_gamma1.0_ord1_nruns20_B1000.pkl` | 3 KB | Four-method comparison, main-text configuration: planted motifs and other tuples detected per sequence at fixed thresholds (SYMBA > 0.6; shuffle p-value < 0.05, Bonferroni, Benjamini–Hochberg), computed from the complete results |
| `aggregated_results_substitution_alph20_g1_n20_r30_ord0_nsr10.pkl`, `aggregated_results_deletion_alph20_g1_n20_r30_ord0_nsr10.pkl` | 3 KB each | Robustness to symbol noise (unordered motifs) |
| `markovsequences/motifs_vs_ps_seqlen_5100_alph_5_thresh_0.5.pkl`, `markovsequences/motifs_vs_ps_seqlen_100100_alph_20_thresh_0.5.pkl` | 0.8 MB, 16 MB | Markov-generated sequences, alphabet size 5 and 20 |

The four-method summaries contain every scalar metric and the averaged PR/ROC curves of
the complete results listed below (made with `symba.slim_summary_data`), and reproduce
the four-method figures exactly.

## Files not shipped

Too large for GitHub (100 MB per file) or third-party data that should be obtained from
the original source.

| File | Size | What it is | How to obtain it |
|---|---|---|---|
| `complete_results_four_methods_faster_rns_1_300__alph30_2w20x10_3w20x10_4w20x10_gamma1.0_ord1_nruns20_B1000.pkl` | 622 MB | Complete results (per-run curves and scores) of the main-text four-method comparison | Run the four-method section with `RECOMPUTE_FOUR_METHODS = True` (several hours); this also rewrites the summary above |
| `complete_results_four_methods_faster_rns_1_300__alph30_2w0x0_3w20x10_4w0x0_gamma1.0_ord1_nruns50_B1000.pkl` | 642 MB | Same, supplementary configuration (length-3 motifs, 10 repetitions) | As above, with the parameters in the file name |
| `complete_results_four_methods_rns_1_300__alph30_2w0x0_3w20x5_4w0x0_gamma1.0_ord1_nruns50_B1000.pkl` | 409 MB | Same, supplementary configuration (length-3 motifs, 5 repetitions), from an earlier version of the code | As above, with the parameters in the file name |
| `enron-email-dataset.zip`, `emails.csv` | 358 MB, 1.3 GB | Enron email dataset (Kaggle) | `kaggle datasets download -d wcukierski/enron-email-dataset -p data`, then unzip in `data/`; the "download from kaggle" cell of the email section does both (set `get_data = True`; needs a Kaggle API token) |
| `sequences_session29_allstimuli_brainareas_multinomial.csv` | 945 MB | Brain-area symbolic sequence used by the brain section | Made by `notebooks/brain_neuropixel_preprocessing.ipynb` (see below) |
| `sequences_session29_allstimuli_multinomial.csv` | 1.17 GB | Optional cache of the neuron-level sequence, the step before the brain-area mapping | Saved by the preprocessing notebook if `SAVE_NEURON_CACHE = True`; reused if present |
| `ecephys_cache_dir/` | ~2.5 GB needed (~130 GB full dataset) | Allen Institute Neuropixels Visual Coding data | Downloaded by `allensdk` from the preprocessing notebook (see below) |

The stock-market data are not stored: they are downloaded from Yahoo Finance each time the
stock sections run (see the note in the notebook).

## Brain (Allen Institute) case study

`notebooks/brain_neuropixel_preprocessing.ipynb` builds
`sequences_session29_allstimuli_brainareas_multinomial.csv` from the Allen Institute
Neuropixels Visual Coding data.

1. **Environment.** The notebook needs its own environment: run
   `./setup_brain_preprocessing_env.sh` (it installs into `brain_preprocessing_env/`; the
   script header also lists the equivalent conda commands). See
   `requirements-brain-preprocessing.txt` for the pinned versions and why they are needed.
2. **Download.** Set `ALLOW_DOWNLOAD = True` in the notebook's setup cell and run it: `allensdk`
   downloads the metadata tables and the session analysed in the paper (ID 797828357,
   ~2.5 GB) into `data/ecephys_cache_dir/`. Set `ALLOW_DOWNLOAD = False` again afterwards.
3. **Run** the notebook to build the sequence.

"session29" in the file names refers to the session's position (index 29) in the list of
`brain_observatory_1.1` sessions. Simultaneous spikes are put in random order when the
sequence is built (seeded), so a new download may not reproduce the file bit for bit;
the results are consistent.

**Known issue.** By default, AllenSDK deletes a cached `.nwb` file and downloads it again
whenever reading it fails, for any reason. Reading these files with recent `pynwb`/`hdmf`
versions fails (they were written with an older NWB schema), which is why those versions are
pinned. As a safeguard, with `ALLOW_DOWNLOAD = False` (the default) the notebook stops
AllenSDK from downloading anything (`fetch_tries=0`, download functions blocked), so a read
failure raises an error instead. You can also make your local cache read-only
(`chmod -R a-w data/ecephys_cache_dir`; undo with `chmod -R u+w data/ecephys_cache_dir`).
