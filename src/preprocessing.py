"""Data loading and preprocessing utilities for the GoEmotions sentiment project."""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path

import emoji
import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
MAX_LEN = 30
LABELS = {"negative": 0, "positive": 1}
CSV_COLUMNS = ["comment_id", "text", "clean_text", "label", "sentiment", "emotions"]

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

BASE_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data/"
)
RAW_FILES = ["train.tsv", "dev.tsv", "test.tsv", "emotions.txt", "sentiment_mapping.json"]
SPLITS = ["train", "dev", "test"]

# Same order as emotions.txt (the ids in the TSV files index this list).
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
NEUTRAL_ID = EMOTIONS.index("neutral")  # 27

# Expected sentiment counts over the 54,263 comments of the filtered version.
EXPECTED_COUNTS = {
    "positive": 19020,
    "negative": 10217,
    "ambiguous": 4805,
    "neutral": 16021,
    "mixed": 4200,
}


def download_raw(raw_dir: Path = RAW_DIR, overwrite: bool = False) -> None:
    """Download the filtered GoEmotions files from the authors' repository."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = raw_dir / name
        if target.exists() and not overwrite:
            print(f"{name}: already present, skipping")
            continue
        urllib.request.urlretrieve(BASE_URL + name, target)
        print(f"{name}: downloaded ({target.stat().st_size:,} bytes)")


def load_goemotions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load train/dev/test TSV files into one DataFrame.

    Columns: text, emotion_ids (list[int]), comment_id, split.
    """
    frames = []
    for split in SPLITS:
        part = pd.read_csv(
            Path(raw_dir) / f"{split}.tsv",
            sep="\t",
            header=None,
            names=["text", "emotion_ids", "comment_id"],
            quoting=csv.QUOTE_NONE,
            dtype=str,
            keep_default_na=False,
        )
        part["split"] = split
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["emotion_ids"] = df["emotion_ids"].apply(lambda s: [int(i) for i in s.split(",")])
    return df


def load_emotion_names(path: Path = RAW_DIR / "emotions.txt") -> list[str]:
    """Read the 28 emotion names from emotions.txt (one per line)."""
    return Path(path).read_text(encoding="utf-8").split()


def load_sentiment_mapping(path: Path = RAW_DIR / "sentiment_mapping.json") -> dict[str, list[str]]:
    """Read the official emotion -> sentiment group mapping."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _emotion_to_group(mapping: dict[str, list[str]]) -> dict[str, str]:
    """Invert the mapping: emotion name -> group, with neutral as its own group."""
    groups = {emotion: group for group, emotions in mapping.items() for emotion in emotions}
    groups["neutral"] = "neutral"
    return groups


def assign_sentiment(emotion_ids: list[int], mapping: dict[str, list[str]] | None = None) -> str:
    """Return positive, negative, ambiguous, neutral or mixed for one comment.

    Each emotion is mapped to its group (neutral is its own group). If all the
    emotions fall in the same group, that group is the sentiment; otherwise
    the comment is mixed (this includes neutral together with another emotion).
    """
    if mapping is None:
        mapping = load_sentiment_mapping()
    groups = _emotion_to_group(mapping)
    found = {groups[EMOTIONS[i]] for i in emotion_ids}
    return found.pop() if len(found) == 1 else "mixed"


def add_sentiment(df: pd.DataFrame, mapping: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Add the columns emotions (names) and sentiment to a copy of df."""
    if mapping is None:
        mapping = load_sentiment_mapping()
    out = df.copy()
    out["emotions"] = out["emotion_ids"].apply(lambda ids: [EMOTIONS[i] for i in ids])
    out["sentiment"] = out["emotion_ids"].apply(lambda ids: assign_sentiment(ids, mapping))
    return out


def check_sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Assert that the sentiment counts match EXPECTED_COUNTS and return them."""
    counts = df["sentiment"].value_counts()
    assert len(df) == sum(EXPECTED_COUNTS.values()), f"expected 54,263 rows, got {len(df):,}"
    for sentiment, expected in EXPECTED_COUNTS.items():
        got = int(counts.get(sentiment, 0))
        assert got == expected, f"{sentiment}: expected {expected:,}, got {got:,}"
    return counts


# ---------------------------------------------------------------------------
# Text cleaning and tokenization
# ---------------------------------------------------------------------------

# Negation words kept even though most of them are NLTK stopwords:
# removing them would flip the meaning ("not happy" -> "happy").
NEGATION_CONTRACTIONS = [
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "can't", "couldn't", "won't", "wouldn't",
    "shan't", "shouldn't", "mustn't", "mightn't", "needn't", "ain't",
]
NEGATIONS = frozenset(
    ["not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere", "cannot"]
    + NEGATION_CONTRACTIONS
    + [word.replace("'", "") for word in NEGATION_CONTRACTIONS]
)

# Irregular contractions; the rest follow the pattern "<aux>n't" -> "<aux> not".
_IRREGULAR_NEGATIONS = {
    "can't": "can not", "cant": "can not", "cannot": "can not",
    "won't": "will not", "wont": "will not",
    "shan't": "shall not", "shant": "shall not",
    "ain't": "is not", "aint": "is not",
}
_NEGATION_STEMS = sorted(
    {word[:-3] for word in NEGATION_CONTRACTIONS if word not in _IRREGULAR_NEGATIONS}
)
_IRREGULAR_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _IRREGULAR_NEGATIONS) + r")\b")
# Matches both "didn't" and "didnt" (apostrophe optional).
_NT_RE = re.compile(r"\b(" + "|".join(_NEGATION_STEMS) + r")n'?t\b")

_APOSTROPHES_RE = re.compile("[\N{LEFT SINGLE QUOTATION MARK}\N{RIGHT SINGLE QUOTATION MARK}\N{ACUTE ACCENT}`]")
_PLACEHOLDERS = {"[NAME]": " nametoken ", "[RELIGION]": " religiontoken "}
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SKIN_TONE_RE = re.compile(r"_(light|medium_light|medium|medium_dark|dark)_skin_tone")
_NON_WORD_RE = re.compile(r"[^\w\s]")
_TOKEN_RE = re.compile(r"\S+")


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """NLTK English stopwords minus the negation words (downloads them if missing)."""
    try:
        words = nltk.corpus.stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("english")
    return frozenset(words) - NEGATIONS


def _emoji_to_token(chars: str, data: dict) -> str:
    """Turn one emoji into a word token, e.g. a red heart -> ' red_heart '."""
    name = emoji.demojize(chars).strip(":").lower()
    name = _SKIN_TONE_RE.sub("", name)
    return " " + re.sub(r"[^a-z0-9]+", "_", name).strip("_") + " "


def expand_negations(text: str) -> str:
    """Rewrite negation contractions as '<aux> not' (expects lowercase text).

    "don't" / "dont" -> "do not", "can't" / "cannot" -> "can not",
    "won't" -> "will not", "ain't" -> "is not". After stopword removal only
    "not" is left, so every negation ends up as the same token.
    """
    text = _IRREGULAR_RE.sub(lambda m: _IRREGULAR_NEGATIONS[m.group(1)], text)
    return _NT_RE.sub(r"\1 not", text)


def tokenize(text: str) -> list[str]:
    """Split an already cleaned text into tokens (whitespace-separated)."""
    return _TOKEN_RE.findall(text)


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normalize one comment for the bag-of-words / sequence models.

    Steps: normalize apostrophes, mask placeholders ([NAME] -> nametoken,
    [RELIGION] -> religiontoken), remove URLs, emojis -> name tokens,
    lowercase, expand negations, drop punctuation and digit-only tokens,
    collapse spaces and (optionally) remove stopwords except negations.
    Can return an empty string (e.g. a comment made only of stopwords).
    """
    text = _APOSTROPHES_RE.sub("'", text)
    for placeholder, token in _PLACEHOLDERS.items():
        text = text.replace(placeholder, token)
    text = _URL_RE.sub(" ", text)
    text = emoji.replace_emoji(text, replace=_emoji_to_token)
    text = text.lower()
    text = expand_negations(text)
    text = _NON_WORD_RE.sub(" ", text)  # other apostrophes too: "i'm" -> "i m"
    tokens = [tok.strip("_") for tok in tokenize(text)]
    tokens = [tok for tok in tokens if tok and not tok.isdigit()]
    if remove_stopwords:
        stopwords = get_stopwords()
        tokens = [tok for tok in tokens if tok not in stopwords]
    return " ".join(tokens)


def cleaning_stats(texts: pd.Series, cleaned: pd.Series) -> dict[str, float]:
    """Empty outputs and mean length in tokens before and after cleaning."""
    return {
        "empty_clean_texts": int((cleaned.str.len() == 0).sum()),
        "mean_tokens_before": round(float(texts.str.split().str.len().mean()), 2),
        "mean_tokens_after": round(float(cleaned.apply(lambda s: len(tokenize(s))).mean()), 2),
    }


def deduplicate_texts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove repeated comments, comparing texts after strip + lowercase.

    If a repeated text has different sentiments, all its copies are dropped
    (conflicting labels). Otherwise only the first copy is kept.
    Returns the deduplicated DataFrame and a dict with the counts.
    """
    norm = df["text"].str.strip().str.lower()
    n_sentiments = df.groupby(norm)["sentiment"].transform("nunique")
    conflicting = n_sentiments > 1
    exact = norm.duplicated(keep="first") & ~conflicting
    out = df[~conflicting & ~exact].copy()
    stats = {
        "rows_before": len(df),
        "repeated_texts": int(norm[norm.duplicated(keep=False)].nunique()),
        "exact_duplicates_removed": int(exact.sum()),
        "conflicting_texts": int(norm[conflicting].nunique()),
        "conflicting_rows_removed": int(conflicting.sum()),
        "rows_after": len(out),
    }
    return out, stats


def build_balanced_dataset(
    df: pd.DataFrame, n_per_class: int = 500, seed: int = SEED
) -> pd.DataFrame:
    """Build the balanced positive/negative dataset from the loaded corpus.

    Steps: add sentiment, keep positive and negative, remove duplicates,
    sample n_per_class comments per class, shuffle and clean the texts.
    The duplicate, candidate and cleaning counts are printed and stored in
    out.attrs["stats"].
    """
    data = add_sentiment(df)
    data = data[data["sentiment"].isin(LABELS)]
    before = data["sentiment"].value_counts()
    data, stats = deduplicate_texts(data)
    after = data["sentiment"].value_counts()
    for sentiment in LABELS:
        stats[f"{sentiment}_before_dedup"] = int(before.get(sentiment, 0))
        stats[f"{sentiment}_after_dedup"] = int(after.get(sentiment, 0))

    parts = [
        data[data["sentiment"] == sentiment].sample(n=n_per_class, random_state=seed)
        for sentiment in LABELS
    ]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    out["clean_text"] = out["text"].apply(clean_text)
    stats.update(cleaning_stats(out["text"], out["clean_text"]))
    if stats["empty_clean_texts"]:
        print(f"warning: {stats['empty_clean_texts']} comments are empty after cleaning")
    out["label"] = out["sentiment"].map(LABELS)
    out["emotions"] = out["emotions"].apply(",".join)
    out = out[CSV_COLUMNS]
    out.attrs["stats"] = stats

    for key, value in stats.items():
        print(f"{key}: {value:,}")
    return out


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split by label."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_processed(
    balanced: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = PROCESSED_DIR,
) -> None:
    """Write balanced.csv, train.csv and test.csv to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("balanced", balanced), ("train", train), ("test", test)]:
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
        print(f"{name}.csv: {len(frame)} rows")


def check_processed(balanced: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Assert the expected sizes, class balance and no train/test overlap."""
    assert list(balanced.columns) == CSV_COLUMNS, "unexpected columns"
    assert len(balanced) == 1000, f"expected 1000 rows, got {len(balanced)}"
    assert balanced["label"].value_counts().to_dict() == {0: 500, 1: 500}
    assert not balanced["text"].str.strip().str.lower().duplicated().any(), "repeated texts"
    assert balanced["comment_id"].is_unique
    assert len(train) == 800 and train["label"].value_counts().to_dict() == {0: 400, 1: 400}
    assert len(test) == 200 and test["label"].value_counts().to_dict() == {0: 100, 1: 100}
    assert not set(train["comment_id"]) & set(test["comment_id"]), "train/test overlap"


def build_and_save_all(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    """Run the whole chain from data/raw to data/processed and check the result."""
    df = load_goemotions(raw_dir)
    check_sentiment_counts(add_sentiment(df))
    balanced = build_balanced_dataset(df)
    train, test = split_dataset(balanced)
    check_processed(balanced, train, test)
    save_processed(balanced, train, test, out_dir)
    return balanced.attrs["stats"]


# ---------------------------------------------------------------------------
# Train-only text vectorization (front-end of the Dense, RNN and LSTM models)
# ---------------------------------------------------------------------------
# TensorFlow is imported inside the functions, so the rest of this module
# (loading, cleaning, splitting) works without it.

PAD_ID = 0  # index of the padding token "" in the vocabulary
OOV_ID = 1  # index of "[UNK]" (tokens not seen in train)
VOCAB_PATH = PROCESSED_DIR / "vocab.txt"


def load_splits(
    processed_dir: Path = PROCESSED_DIR, text_col: str = "clean_text"
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Read train.csv and test.csv and return (X_train_text, y_train, X_test_text, y_test)."""
    frames = [
        pd.read_csv(Path(processed_dir) / f"{name}.csv", keep_default_na=False, encoding="utf-8")
        for name in ("train", "test")
    ]
    train, test = frames
    return (
        train[text_col].astype(str).tolist(),
        train["label"].to_numpy(dtype=np.int64),
        test[text_col].astype(str).tolist(),
        test["label"].to_numpy(dtype=np.int64),
    )


def train_val_split(
    X: list[str], y: np.ndarray, val_size: float = 0.15, seed: int = SEED
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Stratified split of the training set into (X_tr, X_val, y_tr, y_val)."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        list(X), np.asarray(y), test_size=val_size, stratify=y, random_state=seed
    )
    return X_tr, X_val, y_tr, y_val


def build_vectorizer(train_texts: list[str], max_tokens: int | None = None, max_len: int = MAX_LEN):
    """Keras TextVectorization adapted only on the training texts.

    The texts are already cleaned, so there is no extra standardization and
    the split is on whitespace (same tokens as tokenize()). Index 0 is the
    padding and index 1 is [UNK]. max_tokens=None keeps the whole training
    vocabulary; an int keeps the most frequent tokens (counting the 2 special ones).
    Sequences are padded / truncated at the end to max_len.
    """
    import tensorflow as tf

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorizer.adapt(tf.constant(list(train_texts)))
    return vectorizer


def vectorize(vectorizer, texts: list[str]) -> np.ndarray:
    """Turn texts into a (n_texts, max_len) int matrix of token ids."""
    import tensorflow as tf

    return np.asarray(vectorizer(tf.constant(list(texts)))).astype(np.int64)


def vocabulary_size(vectorizer) -> int:
    """Number of ids the Embedding layer needs (vocabulary plus padding and [UNK])."""
    return int(vectorizer.vocabulary_size())


def oov_rate(vectorizer, texts: list[str]) -> float:
    """Percentage of tokens in texts that are not in the vocabulary (they map to [UNK]).

    Counted on the whitespace tokens before padding/truncation.
    """
    vocab = set(vectorizer.get_vocabulary(include_special_tokens=False))
    tokens = [tok for text in texts for tok in tokenize(text)]
    if not tokens:
        return 0.0
    return 100.0 * sum(tok not in vocab for tok in tokens) / len(tokens)


def save_vocabulary(vectorizer, path: Path = VOCAB_PATH) -> int:
    """Write the vocabulary, one token per line, in id order starting at id 2.

    The padding ("" = 0) and [UNK] (1) are not written, so the file can be
    passed back as TextVectorization(vocabulary=path). Returns the number of lines.
    """
    tokens = vectorizer.get_vocabulary(include_special_tokens=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8", newline="\n")
    return len(tokens)


if __name__ == "__main__":
    build_and_save_all()
