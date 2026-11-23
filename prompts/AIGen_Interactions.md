# Generative AI Interactions

Record of how generative AI tools were used during the project, what was accepted and what was changed.

## 2026-11-21 - Nicolas Mejia - Repository setup and dependencies

- **Tool:** AI assistant (LLM).
- **Request:** propose the folder structure required by the assignment, `requirements.txt` and `environment.yml` for Python 3.11 on Windows, a `.gitignore`, and initial `README.md` / `data/README.md` stubs.
- **Review and adjustments:** checked the structure against the assignment statement; verified with pip that the chosen version ranges resolve on Python 3.11 / Windows (TensorFlow 2.20 is the newest version inside the `<2.21` range) and installed the lightweight packages in a local virtual environment; made the `.gitignore` keep `data/raw/.gitkeep` while ignoring the raw data files; README sections "How to run" and "Results" left as placeholders to be completed by the team.

## 2026-11-21 - Nicolas Mejia - Data loading and official sentiment mapping

- **Tool:** AI assistant (LLM).
- **Request:** functions in `src/preprocessing.py` to download and load the filtered GoEmotions files, read the official `sentiment_mapping.json` and assign a sentiment to each comment; documentation of the source, citation and counts.
- **Review and adjustments:** the Kaggle download needs a login, so the same filtered files were taken from the authors' repository and this is stated in `data/README.md`. Checked that the three TSV files load to exactly 54,263 rows with quoting disabled. Tested two rules for neutral combined with other emotions; only "neutral is its own group, any mix of groups is mixed" reproduces the expected counts (19,020 / 10,217 / 4,805 / 16,021 / 4,200), and the other rule was rejected. Added an assert (`check_sentiment_counts`) so the pipeline stops if the counts change. Checked the emotion order against `emotions.txt`.

## 2026-11-21 - Nicolas Mejia - Balanced dataset and stratified split

- **Tool:** AI assistant (LLM).
- **Request:** functions in `src/preprocessing.py` to deduplicate the positive/negative comments, sample a balanced 500/500 dataset, split it 80/20 with stratification and save `data/processed/{balanced,train,test}.csv`; documentation of the counts.
- **Review and adjustments:** checked the duplicate counts by hand with a separate groupby (124 repeated texts, 213 exact copies, 1 conflicting text with 3 copies) and looked at the conflicting example. Kept the deduplication statistics in `DataFrame.attrs` so the function still returns a plain DataFrame. Stored `emotions` as a comma-separated string so the CSV files are easy to read. Reloaded the saved CSV files and ran the checks again (1,000 rows, 500/500, 400/400 and 100/100, no repeated texts, no train/test overlap). Noted that pandas must read the files with `keep_default_na=False`.

## 2026-11-21 - Nicolas Mejia - Text cleaning, tokenization and stopword policy

- **Tool:** AI assistant (LLM).
- **Request:** `clean_text`, `tokenize`, `get_stopwords` and the `NEGATIONS` set in `src/preprocessing.py`, regenerate `data/processed` with the cleaned text and document the preprocessing.
- **Review and adjustments:** chose one consistent form for negations (every `n't` contraction, also written without apostrophe like "dont" or "cant", is expanded to "<aux> not", so only "not" survives stopword removal) and checked it on comments from the dataset. Found that 150 comments use the typographic apostrophe, so apostrophes are normalized first; otherwise "don’t" would lose the negation. Dropped skin-tone suffixes from emoji names. Avoided NLTK's punkt tokenizer (regex split instead). Verified that the regenerated CSV files keep exactly the same comment ids and order, that no text is empty after cleaning, and reviewed the before/after examples by hand. Noted that intensifiers such as "so" and "very" are removed because they are NLTK stopwords.

## 2026-11-21 - Nicolas Mejia - Train-only text vectorization

- **Tool:** AI assistant (LLM).
- **Request:** `load_splits`, `train_val_split`, `build_vectorizer`, `vectorize`, `vocabulary_size`, `oov_rate` and `save_vocabulary` in `src/preprocessing.py`; save `data/processed/vocab.txt` and document the numbers.
- **Review and adjustments:** kept the TensorFlow import inside the functions so loading and cleaning still work without it. Checked that the vocabulary is exactly the set of training tokens, that the 407 test-only words map to `[UNK]` (id 1), that padding is 0 and only at the end, and the shapes 800 x 30 and 200 x 30. Decided to keep the whole training vocabulary (`max_tokens=None`) after comparing OOV in test (30.73% vs 34.21% with 2,000 tokens). Saved the vocabulary without the special tokens and with Unix line endings, and checked that a layer rebuilt from the file gives the same matrix and that two fits give the same order.

## 2026-11-21 - Nicolas Mejia - Evaluation metrics module

- **Tool:** AI assistant (LLM).
- **Request:** `src/evaluate.py` with `compute_metrics`, `predict_labels`, `metrics_row`, `save_metrics`, `load_all_metrics`, `confusion` and `summarize`, following the column contract `model, split, seed, accuracy, precision, recall, f1, f1_macro, kappa, train_time_s, n_params`.
- **Review and adjustments:** computed two small examples by hand (confusion counts, precision, recall, F1, macro F1 and kappa) and checked that the module gives the same values; the calculation is written in `logs/experiment_notes.md`. Ran the edge cases with warnings turned into errors: a constant prediction gives precision 0 and kappa 0 without warnings. Found that scikit-learn returns NaN (with a warning) for kappa when both vectors contain one single identical class, and set it to 0 in that case. Added input checks (only 0/1 labels, same length, probability shapes (n,) or (n, 1)). Tested saving twice to the same file (rows with the same model/split/seed are replaced, the rest kept) and that `grid_*.csv` files are skipped, using temporary files that were deleted afterwards.

## 2026-11-21 - Nicolas Mejia - Plotting helpers

- **Tool:** AI assistant (LLM).
- **Request:** `src/visualize.py` with `save_fig`, `top_ngrams` and the plot functions for the EDA and the model results (class balance, length histogram, top n-grams, word cloud, confusion matrix, learning curves, model comparison and grid-search heatmap).
- **Review and adjustments:** ran every function with `data/processed` and with synthetic inputs (an invented training history, confusion matrix, metrics table and grid), with warnings turned into errors, opened the PNG files to check them and confirmed they are saved at 300 dpi; the test figures were deleted afterwards. Checked that a script uses the Agg backend and that a notebook keeps the inline backend and shows each figure once. Turned off the word cloud's own stopword list, because it would remove "not", which the preprocessing keeps on purpose. Moved the value labels of the comparison chart above the error bars, forced integer ticks for token counts and epochs, and checked `top_ngrams` on a small example counted by hand. `plot_model_comparison` accepts both the raw metrics table and the output of `evaluate.summarize`.

## 2026-11-22 - Nicolas Mejia - Exploratory data analysis notebook

- **Tool:** AI assistant (LLM).
- **Request:** `notebooks/01_eda.ipynb` that imports the project modules and covers data quality, the sentiment mapping, the balanced dataset, sentence length, the effect of the preprocessing, the vocabulary and the conclusions, saving the figures with the `eda_` prefix.
- **Review and adjustments:** ran the notebook from start to finish with `jupyter nbconvert --execute` (twice, same outputs) and checked every number written in the markdown against the cell outputs. Corrected the first draft of the sequence-length text, which said that no cleaned comment of the positive/negative pool is longer than 30 tokens: 2 of the 29,021 are (emoji spam, each emoji becomes a token), so the text now says it and shows them. Rewrote the n-gram comments after looking at the real counts (`not` is the most frequent token in both classes, 146 vs 69 in train). Added the comparison with the full NLTK stopword list (only 3.8% / 1.8% of the comments would keep a negation) and the check that `MAX_LEN = 30` truncates no comment of train or test. Opened the saved figures to check them.

## 2026-11-22 - Nicolas Mejia - DummyClassifier baseline

- **Tool:** AI assistant (LLM).
- **Request:** `notebooks/02_baseline_model.ipynb` with the three `DummyClassifier` strategies evaluated on train and test through `src/evaluate.py`, the metrics saved to `outputs/metrics/dummy.csv`, confusion matrices saved as `baseline_*.png` and a written explanation of the results; plus the "How to run" section of the README.
- **Review and adjustments:** ran the notebook from start to finish with `jupyter nbconvert --execute` and checked every number in the markdown against the cell outputs. Confirmed in the notebook that, with a 0.5/0.5 prior, `most_frequent` always predicts negative only because `argmax` picks the first class, and recomputed its kappa by hand from the confusion matrix. Ran `most_frequent` with a single seed (it is deterministic) and the random strategies with 5 seeds, and showed only the mean when there is a single run instead of a misleading "± 0". Added the binomial check (about 0.56 accuracy at the 95% quantile on 200 comments) after noticing that one random seed already reached 0.56, so the threshold for the neural models is stated with that margin.

## 2026-11-22 - Nicolas Mejia - Report skeleton, introduction, data section and references

- **Tool:** AI assistant (LLM).
- **Request:** `docs/report/main.tex` with the IEEE conference template, the abstract and an `\input` per section; full text for `01_introduction.tex` and `02_data_preprocessing.tex` with two EDA figures; empty section files for the models, evaluation, Transformer, comparison and conclusions; `docs/references.bib` with the base references; LaTeX build files in `.gitignore` and the `docs/report/` line in the README structure.
- **Review and adjustments:** checked every number of the data section against `logs/experiment_notes.md` and the outputs of `notebooks/01_eda.ipynb` (mapping counts, 213 duplicates plus 1 conflicting text, 800/200 and 680/120 splits, 6.98 tokens on average, vocabulary of 2,364 tokens, 30.7% test OOV). Corrected the cleaning description to say that only digit-only tokens are removed and that the kept negations include more words than `not`, `no`, `nor` and `never`, as in `NEGATIONS`. Checked the bibliographic data of each reference and left out DOIs that could not be confirmed (Vaswani et al., Pérez et al., Sipser, Bird et al.). The other sections were left empty for their owners, without invented content. No LaTeX compiler was available locally, so braces, environments, labels, citation keys and figure paths were checked with a script; the PDF is built in Overleaf.

## 2026-11-22 - Daniel Barragan - Dense, RNN and LSTM architectures

- **Tool:** Codex.
- **Request:** Build three binary sentiment classifiers that accept token IDs from the existing vectorizer. Average non-padding embeddings and feed them to a ReLU layer for Dense. Use the final recurrent state for SimpleRNN and LSTM. Give all three builders configurable embedding and hidden sizes, dropout and learning rate, and use a sigmoid output with binary cross-entropy so they can share the same training routine.
- **Implementation and checks:** The three builders share the embedding, dropout, sigmoid output and optimizer. Masking excludes padding from both the Dense average and the recurrent states. Static checks verified syntax, builder signatures and encoder types. All 800 training comments have tokens after cleaning. TensorFlow was unavailable locally, so training and prediction remain unverified.

## 2026-11-22 - Daniel Barragan - Reproducible training

- **Tool:** Codex.
- **Request:** Implement set_seeds and train_model in src/train.py so each architecture uses the same training procedure. Apply the seed before building the model, fit on the training split with separate validation data, monitor validation loss with patience 3 and restore the best weights. Return the history for learning curves and the duration for cost comparisons. Append the model, configuration and outcome to logs/training_logs.txt for each run.
- **Review and adjustments:** A test with a simulated Keras model checked seed-before-build ordering, validation arguments, callback settings, history and timing returns, best-epoch numbering and separate log entries for two successes and one failure. The test used a temporary log. Actual TensorFlow fitting, weight restoration and numerical reproducibility still require an environment with TensorFlow installed.

## 2026-11-22 - Daniel Barragan - Manual hyperparameter grid search

- **Tool:** Codex.
- **Request:** Implement a common search for Dense, vanilla RNN and LSTM. Explore embedding dimensions 32/64, hidden units 16/32/64, dropout 0.2/0.5, learning rates 0.001/0.003 and batch sizes 16/32. Reuse the seeded training routine with early stopping, select by validation macro F1, return the ranked table and save each architecture's grid under outputs/metrics for the tuning notebooks.
- **Review and adjustments:** Validation predictions come from the model restored by the training routine. Lower validation loss breaks macro-F1 ties. A four-trial simulation with controlled predictions checked the number of trials, training dimensions, validation inputs, ranking, tie-breaking and CSV round-trip. The full grid has 48 configurations per architecture. TensorFlow training and scikit-learn scoring were simulated for this check, and the temporary results were discarded.

## 2026-11-22 - Daniel Barragan - Tune the Dense network

- **Tool:** Codex.
- **Request:** Use the shared split, vectorization and training helpers to compare 48 Dense configurations. Keep the test set held out, show the top five validation results, refit the selected model with the same seed and interpret its training and validation curves.
- **Review and adjustments:** Verified that the code cells compile, their project imports exist, and the source training CSV has 400 examples per class and no empty cleaned comments. The notebook calculates its observations from the grid and refit history when executed. TensorFlow was unavailable in the checked local runtime, so no scores or figures from a real run were recorded.
## 2026-11-22 - Daniel Barragan - Vanilla RNN tuning

- **Tool:** Codex.
- **Request:** Evaluate a vanilla SimpleRNN on the same 680 training and 120 validation comments as Dense. Vary embedding size, recurrent units, dropout, learning rate and batch size across 48 trials with seed 42. Rank by validation macro F1, breaking ties by validation loss. Save the rankings, display the top five in notebook 03 and refit the winner to compare its learning curves and validation performance with Dense.
- **Review and adjustments:** Reused the Dense training and validation split and the shared grid search. Executed the RNN cells with the project Python environment after the Jupyter kernel did not start training. Checked the 48 distinct configurations, ranking, top five, learning curves and winning refit against the saved results.

## 2026-11-22 - Daniel Barragan - LSTM tuning

- **Tool:** Codex.
- **Request:** Tune an LSTM on the same tokenized comments, seed and 48 hyperparameter combinations as the other two models to assess whether its gated memory improves sentiment classification. Rank by validation macro F1 with validation loss as the tie-breaker. Save the full grid, display the five best settings, refit the winner and interpret its curves alongside the Dense and SimpleRNN validation results. Reserve the test set for final evaluation.
- **Review and adjustments:** The winning LSTM reached validation macro F1 of 0.7330. Its lowest validation loss occurred at epoch 2, when training accuracy was 91.5% and validation accuracy was 73.3%. The comparison uses one split and one seed.

## 2026-11-22 - Daniel Barragan - Five-seed final evaluation

- **Tool:** Codex.
- **Request:** Compare the tuned Dense, SimpleRNN and LSTM models across five random initializations. Use validation to choose the training duration and the representative model for export. Evaluate every final fit on both train and held-out test comments, record classification scores, training time and parameter count, and report the mean and standard deviation per split. Save the selected models so the team can reproduce their predictions and use the results for the cost comparison.
- **Review and adjustments:** The notebook uses seeds 42 through 46 on a fixed dataset. Validation sets the epoch count before each full-training refit. The notebook records the chosen seed per architecture, and each model CSV contains ten rows. Fit times measure the full-training refit only. The standard deviations capture variation in initialization and training order on one test set.

## 2026-11-22 - Daniel Barragan - Neural classifier error analysis

- **Tool:** Codex.
- **Request:** Compare training and validation curves for the three tuned models, then use their saved final checkpoints to locate false positives and false negatives on the held-out test set. Inspect the same comments for SimpleRNN and LSTM, including cases with negation, possible irony, emoji and masked names. Relate each prediction to the dataset label and the cleaned tokens the model actually received.
- **Review and adjustments:** Checked each checkpoint against its seed-specific test metrics and retained the full 200-comment denominator. The recurrent-model comparison reports both directions of disagreement and shared errors. The frowning emoji becomes an unknown token, so its presence in the original comment cannot establish that the LSTM recognized the emoji.

## 2026-11-22 - Daniel Barragan - Neural models and evaluation report

- **Tool:** Codex.
- **Request:** Write the model and evaluation sections of the English report from the existing experiments. Explain the recurrent update and LSTM gates, the search grid, early stopping and final refits. Summarize accuracy, positive-class precision, recall and F1, macro F1 and kappa across seeds, then use the checkpoint confusion matrices to interpret the trade-off between false positives and false negatives.
- **Review and adjustments:** The main table uses the stratified dummy because it has runs for the same five seeds. Its entries and the neural entries were computed from the stored CSVs using sample standard deviations. Checked the seed-specific confusion counts against the test metrics and verified the section inputs and references. PDF layout remains unverified because the TeX bundle was unavailable locally.

## 2026-11-22 - Daniel Barragan - Reproducing the sentiment experiments

- **Tool:** Codex.
- **Request:** Check that a teammate can reproduce the EDA, dummy baseline and neural results from the documented inputs. Rerun the three notebooks in fresh kernels, compare regenerated rankings, scores, figures and checkpoints with the original experiment, and document how to run notebook 03 or inspect its saved models.
- **Review and adjustments:** Ran each notebook in a separate Python process with a fresh in-process kernel because the standard Jupyter launcher stalled during local socket initialization. Checked the raw dataset hashes, compared seven regenerated CSVs excluding fit time, verified the figures opened and confirmed that the three saved models reproduced the original layer weights exactly.
