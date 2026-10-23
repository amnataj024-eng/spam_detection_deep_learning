# Dataset

**SMS Spam Collection** (Almeida & Hidalgo, UCI Machine Learning Repository) - 5,574 English SMS messages
labelled `ham` (4,827) or `spam` (747).

The file `SMSSpamCollection.tsv` (format: `label<TAB>message`, no header) is already included.
If it is ever missing, run `python download_data.py` from the project folder.

Using the Kaggle version instead? Put `spam.csv` in this folder and change `DATA_FILE` in `src/config.py`
to `DATA_DIR / "spam.csv"` - the loader understands both layouts.
