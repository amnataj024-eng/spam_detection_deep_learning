"""Text cleaning, vocabulary building and sequence conversion.

A small self-contained tokenizer is used (instead of a framework-specific one)
so the saved vocabulary behaves identically in training, evaluation, the CLI and
the Streamlit app, regardless of the installed Keras version.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import numpy as np

from src.config import MAX_SEQUENCE_LEN, MAX_VOCAB_SIZE, OOV_TOKEN, PAD_TOKEN

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\b\d{5,}\b")
NUMBER_RE = re.compile(r"\b\d+\b")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case a message and replace URLs, e-mails and numbers by tokens.

    Spam often contains links, phone numbers and prices, so we keep the *fact*
    that they exist (as urltoken, phonetoken ...) instead of deleting them.
    """
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def tokenize(text: str) -> List[str]:
    return clean_text(text).split()


class Vocabulary:
    """Word -> integer id mapping. id 0 = padding, id 1 = unknown word."""

    def __init__(self, word_to_id: dict):
        self.word_to_id = word_to_id

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = MAX_VOCAB_SIZE) -> "Vocabulary":
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        word_to_id = {PAD_TOKEN: 0, OOV_TOKEN: 1}
        for word, _ in counter.most_common(max_size - len(word_to_id)):
            word_to_id[word] = len(word_to_id)
        return cls(word_to_id)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.word_to_id, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.word_to_id)

    def encode(self, text: str, max_len: int = MAX_SEQUENCE_LEN) -> np.ndarray:
        oov = self.word_to_id[OOV_TOKEN]
        ids = [self.word_to_id.get(w, oov) for w in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))  # post-padding
        return np.array(ids, dtype="int32")

    def encode_batch(self, texts: Iterable[str], max_len: int = MAX_SEQUENCE_LEN) -> np.ndarray:
        return np.stack([self.encode(t, max_len) for t in texts])
