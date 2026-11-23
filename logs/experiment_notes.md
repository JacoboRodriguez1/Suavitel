# Experiment Notes

Notes on data checks, design decisions and experiment results.

## Template

### YYYY-MM-DD - Title

- **Author:**
- **Goal:**
- **Setup:** (data, model, hyperparameters, seed)
- **Result:**
- **Conclusion / next step:**

## Notes

### 2026-11-21 - Sentiment mapping check

- **Author:** Nicolas Mejia
- **Goal:** assign a sentiment to each GoEmotions comment with the official `sentiment_mapping.json` and check the counts expected by the assignment.
- **Setup:** filtered version (train + dev + test = 54,263 rows; 43,410 / 5,426 / 5,427), TSV files read with `quoting=csv.QUOTE_NONE`. Each emotion is mapped to positive / negative / ambiguous, and neutral is its own group.
- **Rule that matches:** if all the emotions of a comment fall in one group, that group is the sentiment; otherwise the comment is mixed (neutral together with any other emotion also counts as mixed).
- **Result:** positive 19,020, negative 10,217, ambiguous 4,805, neutral 16,021, mixed 4,200 (total 54,263): exact match, enforced by `check_sentiment_counts()`.
- **Rejected variant:** ignoring neutral when it appears with other emotions gives positive 19,763, negative 10,819, ambiguous 5,190, neutral 16,021, mixed 2,470, which does not match. 1,751 comments have neutral plus another emotion.
- **Conclusion / next step:** only positive and negative comments go on to the balanced dataset.

### 2026-11-21 - Balanced dataset and stratified split

- **Author:** Nicolas Mejia
- **Goal:** build the balanced 500/500 positive/negative dataset and the 80/20 train/test split.
- **Setup:** train + dev + test joined (54,263 rows), only positive and negative kept, duplicates removed on `text.strip().lower()` (repeated text with different sentiments: all copies dropped; otherwise keep the first), 500 per class sampled with `SEED = 42`, stratified `train_test_split` (test_size=0.2, random_state=42).
- **Candidates:** positive 19,020 -> 18,833 and negative 10,217 -> 10,188 after deduplication (29,237 -> 29,021).
- **Duplicates:** 124 repeated texts; 213 exact copies removed (mostly short comments such as "thank you!", "happy cake day!"); 1 conflicting text ("Weird flex but ok": negative twice, positive once) removed with its 3 copies.
- **Result:** balanced.csv 1,000 rows (500/500), train.csv 800 (400/400), test.csv 200 (100/100), no repeated texts and no comment_id shared between train and test (checked by `check_processed()`). Comment length in the balanced set: 1 to 29 words, mean 13.1.
- **Note:** deduplication only looks at positive/negative comments. One text of the balanced set ("YES!!!!", positive) also appears elsewhere in the corpus as neutral and as mixed; it was kept because those copies are not candidates.
- **Conclusion / next step:** `clean_text` is still a copy of `text`; text cleaning and the stopword policy come next.

### 2026-11-21 - Text cleaning and stopword policy

- **Author:** Nicolas Mejia
- **Goal:** fill `clean_text` with a real cleaning step and keep negations when removing stopwords.
- **Setup:** `clean_text()` in `src/preprocessing.py`: apostrophe normalization, `[NAME]` -> `nametoken`, `[RELIGION]` -> `religiontoken`, URLs removed, emojis -> `emoji.demojize` names as single tokens (skin tone dropped), lowercase, negation contractions expanded to `<aux> not` (with or without apostrophe), punctuation and digit-only tokens removed, NLTK English stopwords removed except `NEGATIONS` (177 stopwords left).
- **Same rows:** `build_and_save_all()` was run again; `comment_id` and row order of `balanced.csv`, `train.csv` and `test.csv` are identical to the previous files, and `text`, `label`, `sentiment`, `emotions` did not change. Only `clean_text` changed.
- **Result:** 0 empty texts after cleaning (25 comments keep a single token). Mean length 13.05 -> 6.98 tokens on the balanced set (train 12.97 -> 6.97, test 13.40 -> 7.03); max 29 -> 20 tokens. Without stopword removal the mean is 13.52 (emojis and expanded contractions add tokens). Vocabulary of `clean_text`: 2,771 tokens.
- **Negations:** 312 comments contain a negation token after cleaning (208 negative, 104 positive); `not` is the most frequent token (267 times). 25 comments contain emojis; 117 contain `[NAME]` and 2 `[RELIGION]`; no URLs in the sample.
- **Conclusion / next step:** `MAX_LEN = 30` covers every comment both before and after cleaning. Next: vectorization fitted only on train.

### 2026-11-21 - Train-only text vectorization

- **Author:** Nicolas Mejia
- **Goal:** turn `clean_text` into fixed-length integer sequences for the Dense, RNN and LSTM models, with a vocabulary learned only from the training set.
- **Setup:** TensorFlow 2.20.0 (Keras 3.15.1, CPU). `build_vectorizer()` in `src/preprocessing.py`: Keras `TextVectorization` with `standardize=None` (texts are already cleaned), `split="whitespace"`, `output_sequence_length=MAX_LEN=30`, `max_tokens=None`, adapted on the 800 training texts only. `load_splits()` reads `train.csv`/`test.csv`; `train_val_split()` keeps 15% of train for validation (stratified, `SEED = 42`).
- **Result:** train matrix 800 x 30, test matrix 200 x 30 (int64). Vocabulary size 2,366 = 2,364 training tokens + padding (id 0, `""`) + `[UNK]` (id 1). Validation split 680 / 120 (340/340 and 60/60).
- **OOV:** 0% in train (by construction). Test: 1,406 tokens, 432 of them unknown -> **30.73% OOV**; 407 of the 874 distinct test tokens never appear in train; 162 of the 200 test comments have at least one `[UNK]` and 5 are only `[UNK]`. With the vocabulary fitted on the 680 training rows, validation OOV is 28.23% (vocabulary 2,138), so validation gives a realistic picture of the test condition.
- **Leakage check:** the vocabulary is exactly the set of training tokens; every token that appears only in test maps to id 1. Fitting on train + test would give 2,773 ids (the 2,771 tokens of the balanced set), which is what the train-only fit avoids.
- **Padding:** sequences are padded with 0 at the end; every non-padding position is > 0. The longest cleaned comment has 20 tokens (train) and 15 (test), so nothing is truncated with `MAX_LEN = 30`.
- **max_tokens:** kept at `None` (whole training vocabulary). The vocabulary is small (2,364 tokens; 1,597 appear only once and 767 at least twice), an embedding of 2,366 x 64 is about 151k weights, and cutting the vocabulary only increases OOV (`max_tokens=2000` -> 34.21% OOV in test).
- **Saved:** `data/processed/vocab.txt` (2,364 lines, ids 2..2,365 in order, special tokens not written). Rebuilding a `TextVectorization` from that file gives the same test matrix; adapting twice gives the same vocabulary order.
- **Conclusion / next step:** high OOV is expected with 800 short comments; the models must rely on frequent words (`not`, `love`, `thanks`, ...). Next: metrics module.

### 2026-11-21 - Evaluation metrics module

- **Author:** Nicolas Mejia
- **Goal:** one shared module (`src/evaluate.py`) that every model uses to compute and store its metrics, so all results have the same definitions and the same CSV columns.
- **Setup:** scikit-learn 1.8.0. positive = 1, negative = 0. `precision`, `recall` and `f1` are for the positive class; `f1_macro` averages the F1 of both classes; `kappa` is Cohen's kappa. `zero_division=0` everywhere. Metrics files: `outputs/metrics/<model>.csv` with columns `model, split, seed, accuracy, precision, recall, f1, f1_macro, kappa, train_time_s, n_params`; saving again replaces the rows with the same model/split/seed. `grid_*.csv` files are ignored by `load_all_metrics()`.
- **Hand check 1:** y_true = [1,1,1,0,0,0,0,1], y_pred = [1,0,1,0,1,0,0,1].
  - Confusion: TP = 3 (positions 0, 2, 7), FN = 1 (position 1), FP = 1 (position 4), TN = 3.
  - accuracy = (3 + 3) / 8 = 0.75; precision = 3 / (3 + 1) = 0.75; recall = 3 / (3 + 1) = 0.75; F1 = 2 * 0.75 * 0.75 / 1.5 = 0.75.
  - Negative class: precision = 3/4, recall = 3/4, F1 = 0.75, so F1 macro = 0.75.
  - Kappa: p_o = 0.75; true positives 4/8 and predicted positives 4/8, so p_e = 0.5 * 0.5 + 0.5 * 0.5 = 0.5; kappa = (0.75 - 0.5) / (1 - 0.5) = **0.5**.
  - `compute_metrics` returns exactly 0.75 / 0.75 / 0.75 / 0.75 / 0.75 / 0.5.
- **Hand check 2 (not symmetric):** y_true = [1,1,1,1,1,0,0,0,0,0], y_pred = [1,1,1,0,0,1,0,0,0,0].
  - TP = 3, FN = 2, FP = 1, TN = 4. accuracy = 7/10 = 0.7; precision = 3/4 = 0.75; recall = 3/5 = 0.6; F1 = 2 * 0.75 * 0.6 / 1.35 = 2/3 = 0.6667.
  - Negative class: precision = 4/6, recall = 4/5, F1 = 8/11 = 0.7273; F1 macro = (0.6667 + 0.7273) / 2 = 0.6970.
  - Kappa: p_o = 0.7; p_e = 0.5 * 0.4 + 0.5 * 0.6 = 0.5; kappa = 0.2 / 0.5 = **0.4**.
  - `compute_metrics` matches all six values.
- **Edge cases (run with warnings turned into errors, none raised):**
  - Always predicting negative on a balanced set: accuracy 0.5, precision/recall/F1 = 0, F1 macro 0.333, kappa 0.
  - Always predicting positive: accuracy 0.5, precision 0.5, recall 1, F1 0.667, kappa 0. This is what the baseline should give: kappa measures agreement beyond chance, and a constant prediction has none.
  - y_true and y_pred both of one single class: scikit-learn's kappa is 0/0 (NaN plus a warning); `compute_metrics` returns 0 in that case.
  - `predict_labels` accepts shapes (n,) and (n, 1) (Keras sigmoid output), uses `>= threshold` and rejects other shapes; labels other than 0/1 and vectors of different length raise `ValueError`.
- **Save/load check:** temporary files in `outputs/metrics` (deleted afterwards): writing 4 rows and then 2 more rows (one with an existing model/split/seed) gives 5 rows with the old row replaced; a `grid_*.csv` and a CSV without the contract columns are skipped by `load_all_metrics()`; `summarize()` gives n_runs, mean and std (ddof = 1) per model and split, checked against numpy.
- **Conclusion / next step:** the module is ready for the baseline and for the neural models. With only 200 test comments each misclassified comment moves accuracy by 0.5 points, so results are reported as mean +- std over seeds.

## Dense network tuning (notebook 03)

The Dense experiment built its vocabulary from the 680 training comments and evaluated all 48 configurations on the same 120 validation comments using seed 42. Configurations were ranked by validation macro F1, with validation loss as the tie-breaker.

The best configuration (embedding dimension 32, 32 hidden units, dropout 0.5, learning rate 0.003 and batch size 32) achieved a validation macro F1 of 0.7582 and an accuracy of 75.83%. Its macro F1 was 0.008 higher than the fifth-ranked configuration. When retrained, the winning model reached its lowest validation loss at epoch 4, with 98.7% training accuracy and 75.8% validation accuracy.

## Vanilla RNN tuning (notebook 03)

On the same 120 validation comments and seed 42 used for Dense, the best of 48 SimpleRNN configurations achieved macro F1 of 0.6283 and accuracy of 63.33%. It used 32-dimensional embeddings, 64 recurrent units, dropout 0.5, learning rate 0.001 and batch size 16. The winning refit reached its lowest validation loss at epoch 1 (0.6843) and stopped after epoch 4. Its macro F1 was 0.021 above fifth place and 0.130 below the selected Dense model on this split. The held-out test set remains reserved for the final evaluation.

## LSTM tuning (notebook 03)

The best LSTM used 64-dimensional embeddings, 32 recurrent units, dropout 0.5, learning rate 0.003 and batch size 32. On the same 120 validation comments and seed 42, its macro F1 was 0.7330, above SimpleRNN (0.6283) and below Dense (0.7582). The winning refit reached its minimum validation loss (0.5800) at epoch 2 and stopped after epoch 5. At epoch 2 training accuracy was 91.5% while validation accuracy was 73.3%, an 18.2 percentage point gap.

## Five-seed final evaluation (notebook 03)

Each architecture used its winning grid configuration. For each seed from 42 to 46, the original 680/120 split determined the epoch with the lowest validation loss. Validation macro F1 ranked the candidate seeds for export, with lower validation loss and then lower seed breaking ties. Each seed was then trained from scratch on all 800 training comments for its selected number of epochs. Final inputs use the token IDs in data/processed/vocab.txt. The train and test rows for a seed share the time of that single full-data fit and the same parameter count.

| Model | Train macro F1 | Test macro F1 | Exported seed | Epochs for exported fit |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.9902 +/- 0.0064 | 0.7408 +/- 0.0432 | 42 | 4 |
| SimpleRNN | 0.8120 +/- 0.1074 | 0.5507 +/- 0.0470 | 46 | 2 |
| LSTM | 0.9452 +/- 0.0412 | 0.7006 +/- 0.0213 | 42 | 2 |

Values are means and sample standard deviations across five seeds. Dense had the highest mean test macro F1, while LSTM had the smallest variation across seeds on the same 200 test comments. The complete scores and costs for the comparison are in outputs/metrics/{dense,rnn,lstm}.csv. The corresponding .keras files in outputs/saved_models accept padded integer token sequences of length 30.

## Test confusion matrices and error analysis

Notebook 03 uses the three tuning curves to compare the epochs with minimum validation loss (Dense 4, SimpleRNN 1, LSTM 2). The test matrices use the validation-selected full-training checkpoints from the five-seed experiment. The test set contains 100 comments per class.

| Checkpoint | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Dense (seed 42) | 75 | 25 | 20 | 80 |
| SimpleRNN (seed 46) | 39 | 61 | 21 | 79 |
| LSTM (seed 42) | 77 | 23 | 34 | 66 |

Both recurrent models were correct on 94 comments and wrong on 33. Only LSTM was correct on 49, and only SimpleRNN on 24. The notebook examines four test IDs covering possible irony, negation, a frowning emoji mapped to the unknown token, and a masked name in a comment expressing mixed cues. These examples illustrate disagreements for the exported checkpoints.

## Reproducibility check for notebooks 01 through 03

The check used an isolated copy of commit 561855b on Windows with Python 3.11.9, TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.3.5, pandas 2.3.3 and scikit-learn 1.8.0. Each notebook started with a fresh in-process IPython kernel (ipykernel 7.4.0). Local socket initialization blocked the standard Jupyter launcher in this environment. The notebook cells, seeds and model code were used as committed. The output directories started empty.

- **EDA:** The five raw GoEmotions files matched their upstream Git blob hashes. All 54,263 comments were accounted for, including 19,020 positive and 10,217 negative comments. All final expression results in the EDA code cells matched their stored text representations. The full-training vocabulary had 2,366 IDs, and 30.73% of test tokens were unknown.
- **Baseline:** All 22 train/test strategy and seed rows were regenerated. The stratified dummy averaged 0.517 test accuracy over seeds 42 through 46.
- **Neural tuning:** All 144 configuration rows matched the reference grids within numerical tolerance, including ranking, validation scores and epoch selection. The winning validation macro F1 values were 0.7582 (Dense), 0.6283 (SimpleRNN) and 0.7330 (LSTM). Fit times were measured anew and excluded from numerical comparisons.
- **Final evaluation:** All 30 neural train/test rows reproduced the committed scores and parameter counts (relative tolerance 1e-6, absolute tolerance 1e-7). All three saved models reproduced the original layer weights exactly. Mean test macro F1 was 0.7408 for Dense, 0.5507 for SimpleRNN and 0.7006 for LSTM. Both recurrent models again missed the same 33 comments in total. All 55 code cells completed, and the regenerated PNG files opened successfully.

## 2026-11-23 - DistilBERT and five-model comparison

- **Author:** Jacobo Rodriguez
- **Goal:** complete the Transformer extension and generate the five-model comparison artifacts for J9.
- **Setup:** `distilbert-base-uncased` on original comment text; the existing 680/120 training/validation split and 200-comment test split; maximum length 64; batch size 16; three epochs; weight decay 0.01. Validation macro F1 selected between learning rates 2e-5 and 5e-5, then the selected rate was evaluated with seeds 42, 43 and 44.
- **Environment:** Python 3.11.9, Transformers 5.24.0, PyTorch 2.15.0+cpu, TensorFlow 2.20.0, scikit-learn 1.8.0. No local GPU was available. The public pretrained checkpoint was downloaded once and loaded from its local cache; no weights were added to the repository.
- **Learning-rate search:** 2e-5 scored validation macro F1 0.866667 at epoch 3; 5e-5 scored 0.866630 at epoch 2. The higher validation score selected 2e-5; the difference was 0.000037.
- **DistilBERT test macro F1:** seed 42 = 0.8397; seed 43 = 0.8400; seed 44 = 0.8496. Mean ± sample standard deviation = 0.8431 ± 0.0056. Test accuracy = 0.8433 ± 0.0058; Cohen's kappa = 0.6867 ± 0.0115. Results are descriptive across three runs on the same held-out test comments.
- **Five-model result:** DistilBERT had the highest mean test macro F1 (0.8431), followed by Dense (0.7408), LSTM (0.7006), SimpleRNN (0.5507), and the stratified dummy (0.5164). The comparison is quality-first and is not a significance test.
- **Inference benchmark:** five repeats of 1,000 already-tokenized inputs on CPU measured 47.193 s for DistilBERT, 0.184 s Dense, 0.185 s SimpleRNN, 0.221 s LSTM and 0.00046 s for the stratified dummy. These are CPU timings and exclude preprocessing/tokenization.
- **Artifacts:** `outputs/metrics/distilbert.csv`, `outputs/metrics/distilbert_lr_search.csv`, `outputs/metrics/comparison.csv`, `outputs/figures/attention_cls_{negation,possible_irony,emoji}.png` and `outputs/figures/comparison.png`. Attention plots are qualitative diagnostics, not causal explanations.
- **Conclusion / next step:** the report and README use the five-model outputs, qualify the different seed counts and identify DistilBERT as the highest mean-F1 model. A larger independent test set and GPU inference timing would better assess generalization and deployment costs.
