"""Data loading and preprocessing utilities for the GoEmotions sentiment project."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import pandas as pd

SEED = 42
MAX_LEN = 30
LABELS = {"negative": 0, "positive": 1}

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
