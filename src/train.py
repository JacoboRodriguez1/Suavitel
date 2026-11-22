"""Reproducible training for the Dense, vanilla RNN and LSTM experiments."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf

from .evaluate import METRICS_DIR, compute_metrics, predict_labels
from .preprocessing import ROOT_DIR, SEED

TRAINING_LOG = ROOT_DIR / "logs" / "training_logs.txt"
BASE_GRID = {
    "embed_dim": (32, 64),
    "units": (16, 32, 64),
    "dropout": (0.2, 0.5),
    "lr": (1e-3, 3e-3),
    "batch_size": (16, 32),
}


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow and use deterministic TensorFlow ops."""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def train_model(
    builder: Callable[..., tf.keras.Model],
    model_params: dict[str, int | float],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int = SEED,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0,
    log_path: Path = TRAINING_LOG,
) -> tuple[tf.keras.callbacks.History, float]:
    """Return the Keras history (history.model holds the best weights) and fit seconds."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    config = f"{model_params}, seed={seed}, epochs={epochs}, batch_size={batch_size}"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        prefix = f"{timestamp} | {builder.__name__} | {config} | "
        try:
            set_seeds(seed)
            model = builder(**model_params)
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=3, restore_best_weights=True
            )
            start = perf_counter()
            history = model.fit(
                X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
                batch_size=batch_size, callbacks=[early_stop], verbose=verbose,
            )
            elapsed = perf_counter() - start
            losses = np.asarray(history.history["val_loss"])
            if not np.isfinite(losses).all():
                raise ValueError("Non-finite validation loss")
            best = int(losses.argmin())
            result = (
                f"epochs_run={len(losses)}, best_epoch={best + 1}, "
                f"best_val_loss={losses[best]:.6f}, train_time_s={elapsed:.3f}"
            )
        except Exception as error:
            log.write(f"{prefix}failed: {type(error).__name__}: {str(error).splitlines()[:1]}\n")
            raise
        log.write(f"{prefix}{result}\n")
    return history, elapsed


def grid_search(
    builder: Callable[..., tf.keras.Model],
    grid: dict[str, Sequence[int | float]],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    vocab_size: int | None = None,
    seed: int = SEED,
    epochs: int = 30,
    metrics_dir: Path = METRICS_DIR,
    log_path: Path = TRAINING_LOG,
) -> pd.DataFrame:
    """Rank validation trials by macro F1, breaking ties by validation loss.

    Pass the training vectorizer's vocabulary size if some IDs are absent
    from X_tr (for example, if long training sequences were truncated).
    """
    if set(grid) != set(BASE_GRID) or any(len(values) == 0 for values in grid.values()):
        raise ValueError(f"Provide at least one value for each of {tuple(BASE_GRID)}")
    if not 1 <= epochs <= 30:
        raise ValueError("epochs must be between 1 and 30")

    if vocab_size is None:
        vocab_size = max(2, int(X_tr.max()) + 1)
    rows = []
    for values in product(*(grid[key] for key in BASE_GRID)):
        params = dict(zip(BASE_GRID, values))
        batch_size = int(params.pop("batch_size"))
        params.update(vocab_size=vocab_size, max_len=X_tr.shape[1])
        tf.keras.backend.clear_session()
        history, elapsed = train_model(
            builder, params, X_tr, y_tr, X_val, y_val,
            seed=seed, epochs=epochs, batch_size=batch_size, log_path=log_path,
        )
        model = history.model
        probs = model.predict(X_val, verbose=0)
        if not np.isfinite(probs).all():
            raise ValueError("Non-finite validation prediction")
        scores = compute_metrics(y_val, predict_labels(probs))
        losses = history.history["val_loss"]
        rows.append({
            "model": model.name, "seed": seed, **params, "batch_size": batch_size,
            **{f"val_{key}": value for key, value in scores.items()},
            "best_val_loss": min(losses), "best_epoch": int(np.argmin(losses)) + 1,
            "epochs_run": len(losses), "train_time_s": elapsed,
        })
    results = pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "best_val_loss"], ascending=[False, True], ignore_index=True
    )
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / f"grid_{results.at[0, 'model']}.csv", index=False)
    return results
