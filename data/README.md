# Data

## Source

GoEmotions: Reddit comments labeled with 27 emotion categories plus neutral.

- Kaggle (link given in the assignment): https://www.kaggle.com/datasets/debarshichanda/goemotions
- Authors' repository: https://github.com/google-research/google-research/tree/master/goemotions

The project uses the **filtered version** of the dataset (`train.tsv`, `dev.tsv`, `test.tsv`, 54,263 comments).
The Kaggle download requires a logged-in account, so the files are downloaded from the authors'
repository (`goemotions/data/`), which hosts the same filtered version together with
`emotions.txt` and the official `sentiment_mapping.json`.

Citation:

> Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020).
> GoEmotions: A Dataset of Fine-Grained Emotions. In *Proceedings of the 58th Annual Meeting of the
> Association for Computational Linguistics* (pp. 4040-4054). https://doi.org/10.18653/v1/2020.acl-main.372

## How to download

From the repository root (with the virtual environment active):

```bash
python -c "from src.preprocessing import download_raw; download_raw()"
```

This saves `train.tsv`, `dev.tsv`, `test.tsv`, `emotions.txt` and `sentiment_mapping.json` in `data/raw/`
(existing files are skipped; use `download_raw(overwrite=True)` to download them again).

## Folders

- `raw/`: original dataset files. They are **not** uploaded to the repository (see above).
- `processed/`: balanced dataset and train/test splits produced by the project code.

## Raw files

The three TSV files have no header and three tab-separated columns: comment text, comma-separated
emotion ids and Reddit comment id. They are read with quoting disabled, so quote characters inside
the comments are kept as they are. Split sizes: train 43,410, dev 5,426, test 5,427.

`load_goemotions()` joins them into one DataFrame and `add_sentiment()` adds the sentiment:

| Column | Type | Description |
|---|---|---|
| `text` | str | Comment text (names and religions are already masked as `[NAME]` and `[RELIGION]`) |
| `emotion_ids` | list[int] | Emotion ids, indices into `emotions.txt` (0 = admiration, ..., 27 = neutral) |
| `comment_id` | str | Reddit comment id |
| `split` | str | Original split: train, dev or test |
| `emotions` | list[str] | Emotion names for `emotion_ids` |
| `sentiment` | str | positive, negative, ambiguous, neutral or mixed |

## Official sentiment mapping

`sentiment_mapping.json` groups the 27 emotions (neutral is not part of any group):

- **positive (12):** amusement, excitement, joy, love, desire, optimism, caring, pride, admiration, gratitude, relief, approval
- **negative (11):** fear, nervousness, remorse, embarrassment, disappointment, sadness, grief, disgust, anger, annoyance, disapproval
- **ambiguous (4):** realization, surprise, curiosity, confusion

Rule used by `assign_sentiment()`: every emotion of a comment is mapped to its group, with neutral
as a group of its own. If all the emotions fall in one group, the comment gets that group; otherwise
it is **mixed** (this includes neutral together with any other emotion). Only positive and negative
comments are used for the binary classifier.

Resulting counts over the 54,263 comments (checked by `check_sentiment_counts()`):

| Sentiment | Comments |
|---|---:|
| positive | 19,020 |
| negative | 10,217 |
| ambiguous | 4,805 |
| neutral | 16,021 |
| mixed | 4,200 |
| **total** | **54,263** |

## Balanced dataset and train/test split

Built by `build_and_save_all()` (or `python -m src.preprocessing` from the repository root), which
runs the whole chain from `data/raw/` and writes `data/processed/balanced.csv`, `train.csv` and
`test.csv` (UTF-8, no index). Steps of `build_balanced_dataset()`:

1. Join train, dev and test (54,263 comments; the original splits are not kept) and assign the sentiment.
2. Keep only **positive** and **negative** comments (19,020 and 10,217 candidates).
3. Remove duplicates, comparing texts after `strip()` and lowercasing. If a repeated text has
   different sentiments, all its copies are removed; otherwise only the first copy is kept.
4. Sample **500 comments per class** with `random_state=SEED` (`SEED = 42`) and shuffle.

Duplicate removal:

| | positive | negative | total |
|---|---:|---:|---:|
| Candidates before | 19,020 | 10,217 | 29,237 |
| Candidates after | 18,833 | 10,188 | 29,021 |

124 texts were repeated: 213 exact copies removed, and 1 text with conflicting labels
("Weird flex but ok", twice negative and once positive) removed with its 3 copies.

`split_dataset()` then makes a stratified 80/20 split by `label` (`train_test_split`, `random_state=SEED`):

| File | Rows | negative (0) | positive (1) |
|---|---:|---:|---:|
| `balanced.csv` | 1,000 | 500 | 500 |
| `train.csv` | 800 | 400 | 400 |
| `test.csv` | 200 | 100 | 100 |

`check_processed()` asserts these sizes, that no text is repeated and that train and test share no
`comment_id`. The test set is only used for the final evaluation.

Columns of the three CSV files:

| Column | Type | Description |
|---|---|---|
| `comment_id` | str | Reddit comment id (unique) |
| `text` | str | Original comment text |
| `clean_text` | str | Cleaned text produced by `clean_text()` (see "Preprocessing") |
| `label` | int | 0 = negative, 1 = positive (`LABELS`) |
| `sentiment` | str | negative or positive |
| `emotions` | str | Comma-separated emotion names, e.g. `anger,annoyance` |

When reading the CSV files with pandas, use `keep_default_na=False` so that texts such as "NA" or
"null" are not turned into missing values.

## Preprocessing

`clean_text()` in `src/preprocessing.py` fills the `clean_text` column of the three CSV files
(`build_balanced_dataset()` applies it after sampling, so the rows and the split are the same as before).
The `text` column keeps the original comment (used later by the Transformer, which has its own tokenizer).

Steps, in order:

1. Normalize apostrophes: `’`, `‘`, `´` and the backtick become `'` (150 of the 1,000 comments use `’`).
2. Replace the dataset placeholders: `[NAME]` -> `nametoken`, `[RELIGION]` -> `religiontoken`
   (single tokens that survive punctuation removal).
3. Remove URLs (`http(s)://...` and `www....`). None of the 1,000 sampled comments has one,
   but the step is kept for new texts.
4. Emojis -> words with the `emoji` package (`emoji.demojize` names): each emoji becomes one token,
   e.g. a red heart -> `red_heart`, a crying face -> `crying_face`. Skin-tone modifiers are dropped
   (`thumbs_up_light_skin_tone` -> `thumbs_up`). 25 comments contain emojis.
5. Lowercase.
6. Expand negation contractions, with or without apostrophe, to `<aux> not`:
   `don't` / `dont` -> `do not`, `isn't` -> `is not`, `can't` / `cant` / `cannot` -> `can not`,
   `won't` / `wont` -> `will not`, `ain't` -> `is not` (`expand_negations()`). The auxiliary is a
   stopword, so after step 9 **every negation contraction becomes the single token `not`**.
7. Remove punctuation (any character that is not a letter, digit, underscore or space). Other
   contractions are split at the apostrophe (`i'm` -> `i m`) and their pieces are stopwords.
8. Tokenize with `tokenize()` (whitespace split with a simple regex, no external tokenizer needed),
   drop tokens made only of digits (`3`, `1000`) and loose underscores.
9. Remove stopwords (`remove_stopwords=True` by default): `get_stopwords()` returns the NLTK English
   list **minus `NEGATIONS`**. The NLTK list is downloaded automatically the first time if it is missing
   (default `nltk_data` folder of the user).

**Stopword policy.** NLTK's English list contains `not`, `no`, `nor`, `don't`, `isn't`, `wasn't`, etc.
Removing them would turn "not happy" into "happy" and flip the sentiment, so they are kept.
`NEGATIONS` holds: not, no, nor, never, none, nobody, nothing, neither, nowhere, cannot and the `n't`
forms (don't, doesn't, didn't, isn't, aren't, wasn't, weren't, haven't, hasn't, hadn't, can't,
couldn't, won't, wouldn't, shan't, shouldn't, mustn't, mightn't, needn't, ain't), also without
apostrophe (dont, isnt, ...). After cleaning, 312 of the 1,000 comments contain a negation token
(208 negative, 104 positive). The remaining stopword list has 177 words. Known side effect:
intensifiers such as "so", "very" and "too" are NLTK stopwords and are removed.

Examples (from `balanced.csv`, except the URL one, which is synthetic because the sample has no URLs):

| Case | Original `text` | `clean_text` |
|---|---|---|
| Negation (`dont`, `cant` without apostrophe) | i disagree with that, you cant say everyone in the country does that, you dont know every [NAME] and their entire life stories stories | disagree not say everyone country not know every nametoken entire life stories stories |
| Negation + `no` | Sad truth is you cant run away from who you are no matter how hard you try :( | sad truth not run away no matter hard try |
| `[NAME]` + curly apostrophes | [NAME] “ I’m afraid if I don’t have money she’ll leave me”. Yuh, you’re exactly right [NAME]. | nametoken afraid not money leave yuh exactly right nametoken |
| `never` + `can’t` | Hello! [NAME] is beautiful but I’ve only been once. I never had the whisky so I can’t even argue haha. | hello nametoken beautiful never whisky not even argue haha |
| `[RELIGION]` | It's normal school except they sexually identify as [RELIGION] | normal school except sexually identify religiontoken |
| Emoji | Thanks for pointing them out, it's hard to see through my tears 😢 | thanks pointing hard see tears crying_face |
| Emoji (glued to a word) | I like your baby😍 | like baby smiling_face_with_heart_eyes |
| Contraction + digits | good. can't even describe just how terrible this entire thing was. just make 1000% sure this is the right guy. | good not even describe terrible entire thing make sure right guy |
| `wouldn’t` | I completely understand why some people would want to report this, I just said that I wouldn’t. You do what you think is right. | completely understand people would want report said would not think right |
| URL (synthetic) | see https://www.reddit.com/r/x and www.google.com 100% | see |
| Only stopwords left | This is so good! | good |

Results on the 1,000 comments (`cleaning_stats()`, also stored in `balanced.attrs["stats"]`):

| | Mean tokens | Max tokens |
|---|---:|---:|
| `text` (whitespace split) | 13.05 | 29 |
| `clean_text` without stopword removal | 13.52 | - |
| `clean_text` (final) | 6.98 | 20 |

No comment is empty after cleaning (`empty_clean_texts = 0`); 25 comments keep a single token
(e.g. "Cheers!!" -> `cheers`, "That was fun." -> `fun`). If a future change produced empty texts,
`build_balanced_dataset()` prints a warning and the count is reported in the stats. The vocabulary of
`clean_text` in the balanced set has 2,771 distinct tokens; the most frequent are `not` (267),
`nametoken` (139), `like`, `love` and `no`.

## Vectorization

The Dense, RNN and LSTM models share the same text front-end, defined in `src/preprocessing.py`
(TensorFlow is only imported inside these functions):

- `load_splits()` -> `(X_train_text, y_train, X_test_text, y_test)` from `train.csv` and `test.csv` (`clean_text` by default).
- `train_val_split()` keeps 15% of the training set for validation (stratified, `SEED = 42`): 680 / 120 rows.
- `build_vectorizer()` creates a Keras `TextVectorization` layer (`standardize=None` because the text is
  already cleaned, `split="whitespace"`, `output_sequence_length=MAX_LEN = 30`) and adapts it **only on the
  training texts**. The test set never touches the vocabulary.
- `vectorize()` returns the integer matrix; `vocabulary_size()`, `oov_rate()` and `save_vocabulary()` complete the set.

Ids: `0` = padding (added at the end of each sequence), `1` = `[UNK]` (token not seen in train), `2...` = training
tokens sorted by frequency (`not`, `nametoken`, `like`, `no`, `good`, ...). `max_tokens=None`: the whole training
vocabulary is kept, because it is small and cutting it would only raise the OOV rate. The longest cleaned comment has
20 tokens, so `MAX_LEN = 30` never truncates.

| | Value |
|---|---:|
| Train matrix | 800 x 30 |
| Test matrix | 200 x 30 |
| Training tokens in the vocabulary | 2,364 |
| Vocabulary size (with padding and `[UNK]`) | 2,366 |
| OOV tokens in test | 30.73% (432 of 1,406) |
| Test comments with at least one `[UNK]` | 162 of 200 |

`data/processed/vocab.txt` holds the vocabulary, one token per line in id order starting at id 2 (padding and
`[UNK]` are not written), so it can be loaded back with `TextVectorization(vocabulary="data/processed/vocab.txt", ...)`.
To rebuild it from the repository root:

```python
from src.preprocessing import load_splits, build_vectorizer, save_vocabulary
X_train_text, y_train, X_test_text, y_test = load_splits()
save_vocabulary(build_vectorizer(X_train_text))
```
