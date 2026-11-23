"""Evaluation metrics shared by every model of the sentiment project."""

from __future__ import annotations

from collections.abc import Callable
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"

# Column contract of every outputs/metrics/<model>.csv file.
METRIC_COLUMNS = [
    "model", "split", "seed",
    "accuracy", "precision", "recall", "f1", "f1_macro", "kappa",
    "train_time_s", "n_params",
]
KEY_COLUMNS = ["model", "split", "seed"]
SCORE_COLUMNS = ["accuracy", "precision", "recall", "f1", "f1_macro", "kappa"]
CLASS_NAMES = ["negative", "positive"]  # label 0, label 1


def _as_labels(y: Sequence[int] | np.ndarray | pd.Series) -> np.ndarray:
    """Return a flat int array of 0/1 labels."""
    arr = np.asarray(y).reshape(-1).astype(int)
    if not np.isin(arr, [0, 1]).all():
        raise ValueError("labels must be 0 (negative) or 1 (positive)")
    return arr


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    """Accuracy, precision/recall/F1 of the positive class (1), macro F1 and Cohen's kappa."""
    y_true = _as_labels(y_true)
    y_pred = _as_labels(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} labels but y_pred has {len(y_pred)}")

    # Kappa is undefined (0/0) when both vectors hold one single, identical class.
    if np.unique(np.concatenate([y_true, y_pred])).size == 1:
        kappa = 0.0
    else:
        kappa = cohen_kappa_score(y_true, y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "kappa": float(kappa),
    }


def predict_labels(probs: Sequence[float] | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Turn positive-class probabilities of shape (n,) or (n, 1) into 0/1 labels."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 2 and probs.shape[1] == 1:
        probs = probs[:, 0]
    if probs.ndim != 1:
        raise ValueError(f"expected shape (n,) or (n, 1), got {probs.shape}")
    return (probs >= threshold).astype(int)


def metrics_row(
    model: str,
    split: str,
    seed: int,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    train_time_s: float | None = None,
    n_params: int | None = None,
) -> dict:
    """One row of the metrics table, with the columns of METRIC_COLUMNS."""
    row = {"model": model, "split": split, "seed": int(seed)}
    row.update(compute_metrics(y_true, y_pred))
    row["train_time_s"] = None if train_time_s is None else float(train_time_s)
    row["n_params"] = None if n_params is None else int(n_params)
    return {col: row[col] for col in METRIC_COLUMNS}


def _row_keys(df: pd.DataFrame) -> pd.Series:
    """String key model|split|seed used to match rows between runs."""
    return df["model"].astype(str) + "|" + df["split"].astype(str) + "|" + df["seed"].astype(int).astype(str)


def save_metrics(rows: Iterable[dict] | pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Write rows to a CSV; existing rows with the same model/split/seed are replaced."""
    path = Path(path)
    new = pd.DataFrame(list(rows) if not isinstance(rows, pd.DataFrame) else rows)
    missing = [col for col in METRIC_COLUMNS if col not in new.columns]
    if missing:
        raise ValueError(f"missing metric columns: {missing}")
    new = new[METRIC_COLUMNS]

    if path.exists():
        old = pd.read_csv(path)[METRIC_COLUMNS]
        old = old[~_row_keys(old).isin(set(_row_keys(new)))]
        new = pd.concat([old, new], ignore_index=True)

    new = new.drop_duplicates(subset=KEY_COLUMNS, keep="last").reset_index(drop=True)
    new["seed"] = new["seed"].astype(int)
    new["n_params"] = new["n_params"].astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def load_all_metrics(metrics_dir: str | Path = METRICS_DIR) -> pd.DataFrame:
    """Concatenate every metrics CSV in metrics_dir (grid_*.csv files are skipped)."""
    frames = []
    for csv_path in sorted(Path(metrics_dir).glob("*.csv")):
        if csv_path.name.startswith("grid_"):
            continue
        df = pd.read_csv(csv_path)
        if set(METRIC_COLUMNS).issubset(df.columns):
            frames.append(df[METRIC_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["n_params"] = out["n_params"].astype("Int64")
    return out


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    """2x2 confusion matrix (rows = true class, columns = predicted class)."""
    cm = confusion_matrix(_as_labels(y_true), _as_labels(y_pred), labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(CLASS_NAMES, name="true"),
        columns=pd.Index(CLASS_NAMES, name="predicted"),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof=1) of each metric per model and split.

    Columns are `<metric>_mean` and `<metric>_std`, plus `n_runs`; the std is NaN
    when a model/split has a single run.
    """
    metrics = SCORE_COLUMNS + ["train_time_s", "n_params"]
    data = df.copy()
    data[metrics] = data[metrics].apply(pd.to_numeric, errors="coerce").astype(float)
    grouped = data.groupby(["model", "split"], sort=True)
    stats = grouped[metrics].agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    stats.insert(0, "n_runs", grouped.size())
    return stats.reset_index()


# ---------------------------------------------------------------------------
# Cross-model comparison and inference cost
# ---------------------------------------------------------------------------

COMPARISON_METRICS = SCORE_COLUMNS + ["train_time_s", "n_params"]


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide train/test means, standard deviations and score gaps per model.

    Each score has `<split>_<metric>_mean` and `<split>_<metric>_std` columns,
    plus `train_test_gap_<metric>` (train mean minus test mean). Run counts are
    included per split so comparisons with different numbers of seeds are clear.
    """
    if df.empty:
        raise ValueError("df must contain model metrics")
    required = {"model", "split", *COMPARISON_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing comparison columns: {sorted(missing)}")

    stats = summarize(df)
    lookup = {
        (str(row.model), str(row.split)): row
        for row in stats.itertuples(index=False)
    }
    rows = []
    for model in pd.unique(df["model"].astype(str)):
        row: dict[str, str | int | float] = {"model": model}
        for split in ("train", "test"):
            values = lookup.get((model, split))
            row[f"{split}_n_runs"] = 0 if values is None else int(values.n_runs)
            for metric in COMPARISON_METRICS:
                row[f"{split}_{metric}_mean"] = np.nan if values is None else getattr(values, f"{metric}_mean")
                row[f"{split}_{metric}_std"] = np.nan if values is None else getattr(values, f"{metric}_std")

        for metric in SCORE_COLUMNS:
            train_mean = row[f"train_{metric}_mean"]
            test_mean = row[f"test_{metric}_mean"]
            row[f"train_test_gap_{metric}"] = (
                float(train_mean) - float(test_mean)
                if pd.notna(train_mean) and pd.notna(test_mean)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _repeat_rows(values, n_rows: int):
    """Repeat rows to n_rows while preserving common array and pandas types."""
    if isinstance(values, pd.DataFrame):
        if values.empty:
            raise ValueError("X must contain at least one row")
        repeats = ceil(n_rows / len(values))
        return pd.concat([values] * repeats, ignore_index=True).iloc[:n_rows]
    if isinstance(values, pd.Series):
        if values.empty:
            raise ValueError("X must contain at least one row")
        repeats = ceil(n_rows / len(values))
        return pd.concat([values] * repeats, ignore_index=True).iloc[:n_rows]

    if isinstance(values, np.ndarray):
        if values.ndim == 0 or len(values) == 0:
            raise ValueError("X must contain at least one row")
        repeats = ceil(n_rows / len(values))
        return np.concatenate([values] * repeats, axis=0)[:n_rows]

    values = list(values)
    if not values:
        raise ValueError("X must contain at least one row")
    repeats = ceil(n_rows / len(values))
    return (values * repeats)[:n_rows]


def measure_inference_time(
    predict_fn: Callable,
    X,
    *,
    n_comments: int = 1000,
    repeats: int = 5,
    warmup: int = 1,
) -> float:
    """Measure mean wall-clock seconds for predicting n_comments examples.

    When X has fewer rows than requested, its rows are repeated to create the
    benchmark batch. predict_fn should return materialized predictions (for
    example, a NumPy array or a CPU tensor) so asynchronous device work is done
    before the timer stops. One short warm-up call is run before timing.
    """
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")
    if n_comments < 1 or repeats < 1 or warmup < 0:
        raise ValueError("n_comments and repeats must be positive; warmup cannot be negative")

    benchmark_batch = _repeat_rows(X, n_comments)
    warmup_size = min(n_comments, 32)
    warmup_batch = benchmark_batch.iloc[:warmup_size] if isinstance(
        benchmark_batch, (pd.DataFrame, pd.Series)
    ) else benchmark_batch[:warmup_size]
    for _ in range(warmup):
        predict_fn(warmup_batch)

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        predict_fn(benchmark_batch)
        durations.append(perf_counter() - start)
    return float(np.mean(durations))
