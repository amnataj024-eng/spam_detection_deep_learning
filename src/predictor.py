"""Load the trained model once and classify new messages."""
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from src.config import LABELS, MODEL_PATH, THRESHOLD, VOCAB_PATH
from src.preprocessing import Vocabulary


@dataclass
class Prediction:
    text: str
    spam_probability: float
    label: str
    is_spam: bool

    @property
    def confidence(self) -> float:
        return self.spam_probability if self.is_spam else 1 - self.spam_probability


class SpamPredictor:
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH, threshold: float = THRESHOLD):
        if not model_path.exists() or not vocab_path.exists():
            raise FileNotFoundError(
                "Trained model not found. Run `python train.py` first "
                f"(expected {model_path} and {vocab_path})."
            )
        from tensorflow import keras  # imported lazily to keep `import src` light
        self.model = keras.models.load_model(model_path)
        self.vocab = Vocabulary.load(vocab_path)
        self.threshold = threshold

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vocab.encode_batch(texts)
        return self.model.predict(x, verbose=0).ravel()

    def predict(self, texts: Sequence[str]) -> List[Prediction]:
        probs = self.predict_proba(texts)
        results = []
        for text, p in zip(texts, probs):
            is_spam = bool(p >= self.threshold)
            results.append(Prediction(text, float(p), LABELS[int(is_spam)], is_spam))
        return results
