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
