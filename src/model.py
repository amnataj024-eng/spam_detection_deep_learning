"""Neural network architecture: Embedding -> Bidirectional LSTM -> Dense."""
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (DROPOUT, EMBEDDING_DIM, LEARNING_RATE, LSTM_UNITS,
                        MAX_SEQUENCE_LEN)


def build_model(vocab_size: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype="int32"),
            layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True, name="embedding"),
            layers.Bidirectional(layers.LSTM(LSTM_UNITS), name="bi_lstm"),
            layers.Dropout(DROPOUT),
            layers.Dense(32, activation="relu"),
            layers.Dropout(DROPOUT / 2),
            layers.Dense(1, activation="sigmoid", name="spam_probability"),
        ],
        name="spam_lstm",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model
