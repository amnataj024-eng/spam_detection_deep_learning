"""Loading the SMS Spam Collection dataset."""
from pathlib import Path
import urllib.request

import pandas as pd

from src.config import DATA_FILE, DATA_URL


def download_dataset(destination: Path = DATA_FILE) -> Path:
    """Download the dataset (tab separated: label <TAB> message) if missing."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading dataset to {destination} ...")
        urllib.request.urlretrieve(DATA_URL, destination)
    return destination


def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """Return a DataFrame with columns ``label`` (1 = spam, 0 = ham) and ``text``.

    Supports two common file layouts:
    * ``SMSSpamCollection`` / ``sms.tsv`` : tab separated, no header, label<TAB>text
    * Kaggle ``spam.csv``                 : comma separated, columns v1 (label), v2 (text)
    """
    path = Path(path)
    if not path.exists():
        download_dataset(path)

    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, encoding="latin-1")
        df = raw.rename(columns={"v1": "label", "v2": "text"})[["label", "text"]]
    else:
        df = pd.read_csv(
            path, sep="\t", header=None, names=["label", "text"],
            encoding="utf-8", quoting=3,  # 3 = csv.QUOTE_NONE
        )

    df = df.dropna().copy()
    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)
