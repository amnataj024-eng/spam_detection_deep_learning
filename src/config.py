"""Central configuration: paths and hyper-parameters used across the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

DATA_FILE = DATA_DIR / "SMSSpamCollection.tsv"
DATA_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)

MODEL_PATH = MODELS_DIR / "spam_lstm.keras"
VOCAB_PATH = MODELS_DIR / "vocab.json"
METRICS_PATH = MODELS_DIR / "metrics.json"
HISTORY_PATH = MODELS_DIR / "history.json"
SPLIT_PATH = MODELS_DIR / "test_split.csv"

# Reproducibility
SEED = 42

# Text -> sequence
MAX_VOCAB_SIZE = 8000     # most frequent words kept
MAX_SEQUENCE_LEN = 50     # messages are padded/truncated to this many tokens
PAD_TOKEN = "<PAD>"
OOV_TOKEN = "<OOV>"

# Model
EMBEDDING_DIM = 64
LSTM_UNITS = 64
DROPOUT = 0.4

# Training
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 3
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Decision threshold on P(spam)
THRESHOLD = 0.5

LABELS = {0: "Ham (Not Spam)", 1: "Spam"}
