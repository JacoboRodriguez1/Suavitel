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
