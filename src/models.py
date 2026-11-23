"""Keras architectures for binary sentiment classification on token IDs."""

from tensorflow import keras
from tensorflow.keras import layers


def _classifier(
    name: str,
    vocab_size: int,
    max_len: int,
    embed_dim: int,
    encoder: list[layers.Layer],
    dropout: float,
    lr: float,
) -> keras.Model:
    """Combine an encoder with a masked embedding and a binary output."""
    model = keras.Sequential(
        [
            keras.Input(shape=(max_len,), dtype="int64"),
            layers.Embedding(vocab_size, embed_dim, mask_zero=True),
            *encoder,
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name=name,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dense(
    vocab_size: int, max_len: int, embed_dim: int, units: int, dropout: float, lr: float
) -> keras.Model:
    """Classify the average embedding of the non-padding tokens."""
    encoder = [layers.GlobalAveragePooling1D(), layers.Dense(units, activation="relu")]
    return _classifier("dense", vocab_size, max_len, embed_dim, encoder, dropout, lr)


def build_rnn(
    vocab_size: int, max_len: int, embed_dim: int, units: int, dropout: float, lr: float
) -> keras.Model:
    """Classify the final hidden state of a vanilla recurrent network."""
    return _classifier("rnn", vocab_size, max_len, embed_dim, [layers.SimpleRNN(units)], dropout, lr)


def build_lstm(
    vocab_size: int, max_len: int, embed_dim: int, units: int, dropout: float, lr: float
) -> keras.Model:
    """Classify the final hidden state of a long short-term memory network."""
    return _classifier("lstm", vocab_size, max_len, embed_dim, [layers.LSTM(units)], dropout, lr)


def build_transformer(model_name: str = "distilbert-base-uncased"):
    """Load a pretrained Hugging Face sequence classifier for two sentiment labels.

    The dependency is imported lazily so users of the Keras models do not need
    to initialize PyTorch or Transformers just to import this module.
    """
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as error:
        raise ImportError(
            "Transformer support requires the optional 'transformers' package. "
            "Install the project requirements before building DistilBERT."
        ) from error

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
