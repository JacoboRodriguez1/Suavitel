"""Reproducible training for the Dense, vanilla RNN and LSTM experiments."""

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

from .preprocessing import ROOT_DIR, SEED

TRAINING_LOG = ROOT_DIR / "logs" / "training_logs.txt"


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
