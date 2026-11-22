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
