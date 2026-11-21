"""Data loading and preprocessing utilities for the GoEmotions sentiment project."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

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
    sample n_per_class comments per class and shuffle. The duplicate and
    candidate counts are printed and stored in out.attrs["stats"].
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

    out["clean_text"] = out["text"]  # real cleaning is added in a later step
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


if __name__ == "__main__":
    build_and_save_all()
