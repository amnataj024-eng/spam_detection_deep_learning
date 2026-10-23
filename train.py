"""Train the Embedding + Bidirectional-LSTM spam classifier.

Usage (from the project folder):
    python train.py
"""
import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src import config as C
from src.data_loader import load_dataset
from src.preprocessing import Vocabulary
from src.visualization import plot_class_distribution, plot_history, plot_length_distribution


def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def main() -> None:
    set_seeds(C.SEED)
    from tensorflow import keras
    from src.model import build_model

    C.MODELS_DIR.mkdir(exist_ok=True); C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print(f"Loaded {len(df)} messages | spam = {df['label'].sum()} | ham = {(df['label'] == 0).sum()}")
    plot_class_distribution(df, C.FIGURES_DIR / "class_distribution.png")
    plot_length_distribution(df, C.FIGURES_DIR / "length_distribution.png")

    # Stratified 70 / 15 / 15 split
    train_df, temp_df = train_test_split(
        df, test_size=C.TEST_SIZE + C.VAL_SIZE, stratify=df["label"], random_state=C.SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=C.TEST_SIZE / (C.TEST_SIZE + C.VAL_SIZE),
        stratify=temp_df["label"], random_state=C.SEED)
    print(f"Train {len(train_df)} | Validation {len(val_df)} | Test {len(test_df)}")
    test_df.to_csv(C.SPLIT_PATH, index=False)  # evaluate.py uses exactly this held-out set

    # The vocabulary is built from the TRAINING data only (no leakage)
    vocab = Vocabulary.build(train_df["text"])
    vocab.save(C.VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab)}")

    x_train, y_train = vocab.encode_batch(train_df["text"]), train_df["label"].values
    x_val, y_val = vocab.encode_batch(val_df["text"]), val_df["label"].values

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print("Class weights:", class_weight)

    model = build_model(len(vocab))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(C.MODEL_PATH), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, class_weight=class_weight,
        callbacks=callbacks, verbose=2)

    model.save(C.MODEL_PATH)  # best weights were restored by EarlyStopping
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    C.HISTORY_PATH.write_text(json.dumps(hist, indent=2), encoding="utf-8")
    plot_history(hist, C.FIGURES_DIR / "training_history.png")
    print(f"\nModel saved to {C.MODEL_PATH}\nNext step: python evaluate.py")


if __name__ == "__main__":
    main()
