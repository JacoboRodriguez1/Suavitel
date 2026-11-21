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
| `clean_text` | str | Cleaned text (for now a copy of `text`; cleaning is added in the next step) |
| `label` | int | 0 = negative, 1 = positive (`LABELS`) |
| `sentiment` | str | negative or positive |
| `emotions` | str | Comma-separated emotion names, e.g. `anger,annoyance` |

When reading the CSV files with pandas, use `keep_default_na=False` so that texts such as "NA" or
"null" are not turned into missing values.

## Preprocessing

To be documented.
