"""Five-seed evaluation of the selected Dense, SimpleRNN and LSTM classifiers."""

from time import perf_counter

import numpy as np
import pandas as pd
import tensorflow as tf

from .evaluate import METRICS_DIR, compute_metrics, metrics_row, predict_labels, save_metrics
from .models import build_dense, build_lstm, build_rnn
from .preprocessing import (
    MAX_LEN, ROOT_DIR, SEED, VOCAB_PATH, build_vectorizer, load_splits,
    train_val_split, vectorize, vocabulary_size,
)
from .train import set_seeds, train_model


def _labels(model, tokens):
    probabilities = model.predict(tokens, verbose=0)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite predictions from {model.name}")
    return predict_labels(probabilities)


def run_final_evaluation():
    """Write one CSV and one validation-selected full-train model per architecture."""
    texts, labels, test_texts, test_labels = load_splits()
    fit_texts, val_texts, fit_labels, val_labels = train_val_split(texts, labels, seed=SEED)
    fit_vectorizer = build_vectorizer(fit_texts, max_len=MAX_LEN)
    fit_tokens = vectorize(fit_vectorizer, fit_texts)
    val_tokens = vectorize(fit_vectorizer, val_texts)

    final_vectorizer = build_vectorizer(texts, max_len=MAX_LEN)
    if final_vectorizer.get_vocabulary()[2:] != VOCAB_PATH.read_text(encoding="utf8").splitlines():
        raise ValueError("Full-training token IDs differ from data/processed/vocab.txt")
    train_tokens = vectorize(final_vectorizer, texts)
    test_tokens = vectorize(final_vectorizer, test_texts)
    model_dir = ROOT_DIR / "outputs" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    selections = []
    for name, builder in [("dense", build_dense), ("rnn", build_rnn), ("lstm", build_lstm)]:
        grid = pd.read_csv(METRICS_DIR / f"grid_{name}.csv")
        best = grid.sort_values(["val_f1_macro", "best_val_loss"], ascending=[False, True]).iloc[0]
        params = dict(max_len=MAX_LEN, embed_dim=int(best.embed_dim), units=int(best.units),
                      dropout=float(best.dropout), lr=float(best.lr))
        batch_size = int(best.batch_size)
        trials = []
        for seed in range(SEED, SEED + 5):
            tf.keras.backend.clear_session()
            history, _ = train_model(
                builder, dict(params, vocab_size=vocabulary_size(fit_vectorizer)),
                fit_tokens, fit_labels, val_tokens, val_labels, seed=seed, batch_size=batch_size,
            )
            losses = history.history["val_loss"]
            scores = compute_metrics(val_labels, _labels(history.model, val_tokens))
            trials.append(dict(model=name, seed=seed, epochs=int(np.argmin(losses)) + 1,
                               val_f1_macro=scores["f1_macro"], val_loss=min(losses)))

        winner = min(trials, key=lambda t: (-t["val_f1_macro"], t["val_loss"], t["seed"]))
        rows = []
        for trial in trials:
            seed = trial["seed"]
            tf.keras.backend.clear_session()
            set_seeds(seed)
            model = builder(**params, vocab_size=vocabulary_size(final_vectorizer))
            start = perf_counter()
            model.fit(train_tokens, labels, epochs=trial["epochs"], batch_size=batch_size, verbose=0)
            elapsed = perf_counter() - start
            for split, tokens, truth in [("train", train_tokens, labels), ("test", test_tokens, test_labels)]:
                rows.append(metrics_row(name, split, seed, truth, _labels(model, tokens),
                                        train_time_s=elapsed, n_params=model.count_params()))
            trial["saved"] = seed == winner["seed"]
            if trial["saved"]:
                model.save(model_dir / f"{name}.keras")
        save_metrics(rows, METRICS_DIR / f"{name}.csv")
        selections.extend(trials)
    return pd.DataFrame(selections)
