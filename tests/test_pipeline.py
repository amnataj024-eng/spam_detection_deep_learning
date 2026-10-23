"""Lightweight tests. Run with:  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MAX_SEQUENCE_LEN
from src.data_loader import load_dataset
from src.preprocessing import Vocabulary, clean_text


class PreprocessingTests(unittest.TestCase):
    def test_clean_text_replaces_urls_and_numbers(self):
        out = clean_text("WIN $1000 now!! Visit http://spam.com or call 08001234567")
        self.assertIn("urltoken", out)
        self.assertIn("phonetoken", out)
        self.assertNotIn("http", out)

    def test_encode_shape_and_padding(self):
        vocab = Vocabulary.build(["hello world", "free prize now"])
        ids = vocab.encode("hello unknownword")
        self.assertEqual(ids.shape, (MAX_SEQUENCE_LEN,))
        self.assertEqual(ids[2], 0)          # padding
        self.assertEqual(ids[1], 1)          # unknown word -> OOV id

    def test_dataset_loads(self):
        df = load_dataset()
        self.assertGreater(len(df), 5000)
        self.assertEqual(set(df["label"].unique()), {0, 1})


if __name__ == "__main__":
    unittest.main()
