"""Plotting helpers for the EDA and the model results.

Every plot function returns the matplotlib Figure and, when `name` is given,
saves it to outputs/figures/<name>.png at 300 dpi.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

# Scripts use a non-interactive backend; notebooks keep their inline backend.
if "ipykernel" not in sys.modules and "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from wordcloud import WordCloud  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
DPI = 300

# Fixed color order so the same entity keeps the same color in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = {"negative": "#eb6834", "positive": "#2a78d6"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#1baf7a", "test": "#eb6834"}
TEXT_COLOR = "#333333"
SEQUENTIAL_CMAP = "Blues"


def set_style() -> None:
    """Apply the common figure style (called once on import)."""
    sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#bbbbbb",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


set_style()


def save_fig(fig: plt.Figure, name: str, figures_dir: str | Path = FIGURES_DIR) -> Path:
    """Save fig as <figures_dir>/<name>.png at 300 dpi and return the path."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if name.endswith(".png"):
        name = name[:-4]
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    return path


def _finish(fig: plt.Figure, name: str | None) -> plt.Figure:
    """Tighten the layout and save the figure when a name is given."""
    fig.tight_layout()
    if name:
        save_fig(fig, name)
    return fig


def _color_for(key: str, i: int) -> str:
    """Class color for negative/positive, otherwise the i-th palette color."""
    return CLASS_COLORS.get(str(key).lower(), PALETTE[i % len(PALETTE)])


# ---------------------------------------------------------------- EDA plots

def plot_class_balance(
    counts: Mapping[str, int] | pd.Series,
    title: str = "Class balance",
    name: str | None = None,
) -> plt.Figure:
    """Bar chart of the number of comments per class, with count and share labels."""
    counts = pd.Series(counts)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(counts) + 2), 4))
    colors = [_color_for(k, i) for i, k in enumerate(counts.index)]
    bars = ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.6)
    labels = [f"{v:,} ({v / total:.1%})" for v in counts.values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set_ylabel("Number of comments")
    ax.set_title(title)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    return _finish(fig, name)


def plot_length_hist(
    lengths_by_class: Mapping[str, Sequence[int]],
    title: str = "Comment length",
    xlabel: str = "Number of tokens",
    name: str | None = None,
    percentile: float | None = 95,
) -> plt.Figure:
    """Overlaid histograms of lengths per class, with an optional percentile line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    all_lengths = np.concatenate([np.asarray(v, dtype=int) for v in lengths_by_class.values()])
    bins = np.arange(all_lengths.min(), all_lengths.max() + 2) - 0.5
    for i, (label, lengths) in enumerate(lengths_by_class.items()):
        color = _color_for(label, i)
        lengths = np.asarray(lengths, dtype=int)
        ax.hist(lengths, bins=bins, color=color, alpha=0.25)
        ax.hist(lengths, bins=bins, color=color, histtype="step", linewidth=1.5,
                label=f"{label} (n={len(lengths)}, mean={lengths.mean():.1f})")
    if percentile is not None:
        p = np.percentile(all_lengths, percentile)
        ax.axvline(p, color=TEXT_COLOR, linestyle="--", linewidth=1,
                   label=f"{percentile:g}th percentile = {p:.0f}")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of comments")
    ax.set_title(title)
    ax.legend()
    return _finish(fig, name)


def top_ngrams(texts: Iterable[str], n: int = 1, k: int = 15) -> list[tuple[str, int]]:
    """The k most frequent n-grams of texts that are already tokenized by spaces."""
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = str(text).split()
        counter.update(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return counter.most_common(k)


def _barh(ax: plt.Axes, freqs: Sequence[tuple[str, int]], color: str, title: str) -> None:
    """Horizontal bars with the most frequent item on top."""
    words = [w for w, _ in freqs][::-1]
    values = [c for _, c in freqs][::-1]
    bars = ax.barh(words, values, color=color, height=0.7)
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)


def plot_top_ngrams(
    freqs: Sequence[tuple[str, int]] | Mapping[str, Sequence[tuple[str, int]]],
    n: int = 15,
    title: str = "Most frequent n-grams",
    name: str | None = None,
) -> plt.Figure:
    """Top-n frequency bars: one panel for a list, one panel per class for a dict."""
    panels = dict(freqs) if isinstance(freqs, Mapping) else {"": list(freqs)}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 0.3 * n + 1.5), squeeze=False)
    for i, (ax, (label, items)) in enumerate(zip(axes[0], panels.items())):
        _barh(ax, list(items)[:n], _color_for(label, i), label)
    if len(panels) > 1:
        fig.suptitle(title, fontweight="bold")
    else:
        axes[0, 0].set_title(title)
    return _finish(fig, name)


def plot_wordcloud(
    text_or_freqs: str | Iterable[str] | Mapping[str, int] | Sequence[tuple[str, int]],
    title: str = "Word cloud",
    name: str | None = None,
    color: str = PALETTE[0],
    max_words: int = 100,
) -> plt.Figure:
    """Word cloud from raw text (or a list of texts) or from word frequencies.

    Texts are expected to be cleaned already, so no extra stopwords are removed.
    """
    cloud = WordCloud(
        width=1200, height=600, background_color="white", max_words=max_words,
        stopwords=set(), collocations=False, random_state=42,
        color_func=lambda *args, **kwargs: color,
    )
    if isinstance(text_or_freqs, str):
        cloud.generate(text_or_freqs)
    elif isinstance(text_or_freqs, Mapping):
        cloud.generate_from_frequencies(dict(text_or_freqs))
    else:
        items = list(text_or_freqs)
        if items and isinstance(items[0], tuple):
            cloud.generate_from_frequencies(dict(items))
        else:
            cloud.generate(" ".join(map(str, items)))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return _finish(fig, name)


# ------------------------------------------------------------ result plots

def plot_confusion(
    cm: Sequence[Sequence[int]] | np.ndarray | pd.DataFrame | None = None,
    y_true: Sequence[int] | None = None,
    y_pred: Sequence[int] | None = None,
    labels: Sequence[str] = ("negative", "positive"),
    title: str = "Confusion matrix",
    name: str | None = None,
) -> plt.Figure:
    """Confusion matrix heatmap (rows = true, columns = predicted) with counts and row %."""
    if cm is None:
        if y_true is None or y_pred is None:
            raise ValueError("give either cm or both y_true and y_pred")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                              labels=list(range(len(labels))))
    cm = np.asarray(cm, dtype=int)
    if cm.shape != (len(labels), len(labels)):
        raise ValueError(f"cm has shape {cm.shape}, expected {(len(labels), len(labels))}")
    row_sums = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    annot = np.array([[f"{cm[i, j]}\n({share[i, j]:.0%})" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(share, annot=annot, fmt="", cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, cbar=False,
                square=True, linewidths=2, linecolor="white",
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"fontsize": 11})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _finish(fig, name)


def plot_learning_curves(
    history: Mapping[str, Sequence[float]] | object,
    title: str = "Learning curves",
    name: str | None = None,
) -> plt.Figure:
    """Train vs validation loss and accuracy per epoch (dict or Keras History).

    The dotted line marks the epoch with the lowest validation loss.
    """
    hist = dict(getattr(history, "history", history))
    metrics = [m for m in ("loss", "accuracy") if m in hist]
    if not metrics:
        raise ValueError("history needs at least 'loss' or 'accuracy'")
    best = int(np.argmin(hist["val_loss"])) + 1 if "val_loss" in hist else None

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 3.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        epochs = np.arange(1, len(hist[metric]) + 1)
        ax.plot(epochs, hist[metric], color=SPLIT_COLORS["train"], linewidth=2,
                marker="o", markersize=4, label="train")
        if f"val_{metric}" in hist:
            ax.plot(epochs, hist[f"val_{metric}"], color=SPLIT_COLORS["test"], linewidth=2,
                    marker="o", markersize=4, label="validation")
        if best is not None:
            ax.axvline(best, color=TEXT_COLOR, linestyle=":", linewidth=1,
                       label=f"best epoch ({best})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(metric.capitalize())
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.legend()
    fig.suptitle(title, fontweight="bold")
    return _finish(fig, name)


def plot_model_comparison(
    df: pd.DataFrame,
    metric: str = "f1_macro",
    title: str | None = None,
    name: str | None = None,
    order: Sequence[str] | None = None,
) -> plt.Figure:
    """Grouped bars of a metric per model and split, with std error bars.

    Accepts the raw metrics table (one row per model/split/seed) or a summary
    with `<metric>_mean` and `<metric>_std` columns (see evaluate.summarize).
    """
    if f"{metric}_mean" in df.columns:
        stats = df[["model", "split"]].copy()
        stats["mean"] = df[f"{metric}_mean"]
        stats["std"] = df[f"{metric}_std"] if f"{metric}_std" in df.columns else np.nan
    else:
        stats = (df.groupby(["model", "split"], sort=False)[metric]
                 .agg(["mean", "std"]).reset_index())
    models = list(order) if order is not None else list(dict.fromkeys(stats["model"]))
    splits = [s for s in ("train", "val", "test") if s in set(stats["split"])]
    splits += [s for s in dict.fromkeys(stats["split"]) if s not in splits]

    x = np.arange(len(models))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(models) + 2), 4))
    for i, split in enumerate(splits):
        part = stats[stats["split"] == split].set_index("model").reindex(models)
        offset = (i - (len(splits) - 1) / 2) * width
        std = part["std"].fillna(0)
        ax.bar(x + offset, part["mean"], width * 0.92, yerr=std, capsize=3,
               error_kw={"linewidth": 1}, color=SPLIT_COLORS.get(split, PALETTE[i % len(PALETTE)]),
               label=split)
        # Value labels above the error bars so they do not overlap.
        for xi, mean, sd in zip(x + offset, part["mean"], std):
            if pd.notna(mean):
                ax.annotate(f"{mean:.2f}", (xi, mean + sd), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, models)
    ax.set_ylabel(metric)
    if stats["mean"].between(-1, 1).all():
        ax.set_ylim(min(0, stats["mean"].min() - 0.05), 1.1)
    ax.set_title(title or f"Model comparison ({metric}, mean ± std)")
    ax.legend(ncols=len(splits), loc="upper left")
    ax.grid(axis="x", visible=False)
    return _finish(fig, name)


def plot_grid_heatmap(
    grid_df: pd.DataFrame,
    row: str,
    col: str,
    value: str,
    agg: str = "max",
    title: str | None = None,
    name: str | None = None,
) -> plt.Figure:
    """Heatmap of a grid-search score over two hyperparameters.

    Other hyperparameters are collapsed with `agg` (best value by default).
    """
    table = grid_df.pivot_table(index=row, columns=col, values=value, aggfunc=agg)
    fig, ax = plt.subplots(figsize=(1.1 * table.shape[1] + 2.5, 0.6 * table.shape[0] + 1.8))
    sns.heatmap(table, annot=True, fmt=".3f", cmap=SEQUENTIAL_CMAP, linewidths=2,
                linecolor="white", cbar_kws={"label": value}, ax=ax)
    ax.set_title(title or f"{value} ({agg} over other hyperparameters)")
    ax.tick_params(axis="y", rotation=0)
    return _finish(fig, name)


def plot_cls_attention(
    tokens: Sequence[str],
    weights: Sequence[float],
    title: str,
    name: str | None = None,
) -> plt.Figure:
    """Plot mean last-layer attention from [CLS] across the visible tokens."""
    tokens = list(tokens)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if len(tokens) != len(weights):
        raise ValueError(f"received {len(tokens)} tokens but {len(weights)} attention weights")
    if not tokens:
        raise ValueError("at least one visible token is required")

    height = max(2.8, 0.34 * len(tokens) + 1.2)
    fig, ax = plt.subplots(figsize=(8, height))
    positions = np.arange(len(tokens))
    colors = plt.get_cmap(SEQUENTIAL_CMAP)(
        np.zeros_like(weights) if weights.max(initial=0) == 0 else weights / weights.max()
    )
    ax.barh(positions, weights, color=colors, height=0.72)
    ax.set_yticks(positions, tokens)
    ax.invert_yaxis()
    ax.set_xlabel("Mean attention weight from [CLS]")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(float(weights.max()) * 1.15, 0.01))
    return _finish(fig, name)
