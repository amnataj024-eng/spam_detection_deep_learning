# Spam Detection Using Deep Learning

Detect spam messages with a neural network: an **Embedding + Bidirectional LSTM** classifier built with
TensorFlow/Keras, trained on the SMS Spam Collection dataset, with a **Streamlit** web app.

| Test-set result (837 unseen messages) | Value |
|---|---|
| Accuracy | **98.45 %** |
| Spam precision | 93.04 % |
| Spam recall | 95.54 % |
| Spam F1-score | 94.27 % |
| ROC-AUC | 0.985 |

## Project structure

```
spam_detection_deep_learning/
├── data/                 SMSSpamCollection.tsv (dataset) + README.md
├── models/               spam_lstm.keras, vocab.json, metrics.json, history.json, test_split.csv, misclassified.csv
├── notebooks/            01_exploration_and_experiments.ipynb (EDA, training, threshold experiment)
├── report/               report.md (university report) + figures/
├── src/                  config.py, data_loader.py, preprocessing.py, model.py, predictor.py, visualization.py
├── tests/                test_pipeline.py
├── train.py              trains the model and saves it to models/
├── evaluate.py           evaluates on the held-out test set, saves metrics and plots
├── predict.py            classify messages from the command line
├── app.py                Streamlit web application
├── download_data.py      re-downloads the dataset if needed
├── requirements.txt
└── .gitignore
```

## Setup on Windows 11 (VS Code)

1. Install **Python 3.10 - 3.12** from python.org (tick *Add python.exe to PATH*).
2. In VS Code: **File > Open Folder...** and choose `spam_detection_deep_learning`.
3. Open a terminal (**Terminal > New Terminal**, PowerShell) and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.
In VS Code press `Ctrl+Shift+P` > *Python: Select Interpreter* > choose `.venv`.

## Run it

The trained model is already included, so you can launch the app immediately:

```powershell
streamlit run app.py          # web app at http://localhost:8501
```

To reproduce everything from scratch:

```powershell
python train.py               # ~1 minute on CPU; saves models/spam_lstm.keras
python evaluate.py            # metrics + confusion matrix + ROC curve
python predict.py "You won a free prize! Call now"
python -m unittest discover -s tests
```

For the notebook open `notebooks/01_exploration_and_experiments.ipynb` in VS Code and choose the `.venv` kernel.

## How it works

1. **Preprocessing** (`src/preprocessing.py`): lower-casing; URLs, e-mails, long numbers and numbers become the
   tokens `urltoken`, `phonetoken`, `numtoken`; punctuation removed. A vocabulary of the 8,000 most frequent training
   words is built (training data only, so there is no data leakage); messages are padded/truncated to 50 tokens.
2. **Model** (`src/model.py`): `Embedding(64)` -> `Bidirectional(LSTM(64))` -> `Dropout(0.4)` -> `Dense(32, relu)` ->
   `Dropout(0.2)` -> `Dense(1, sigmoid)` (about 479 k parameters).
3. **Training** (`train.py`): stratified 70/15/15 train/validation/test split, Adam, binary cross-entropy,
   class weights for the imbalance (13 % spam), early stopping on validation loss, fixed seed 42.
4. **Evaluation** (`evaluate.py`): precision/recall/F1, confusion matrix, ROC-AUC on the untouched test set.

## Limitations

- Trained on English SMS from ~2012; modern phishing and e-mail spam may look different.
- Short, ambiguous messages (e.g. "Have you been practising your curtsey?") are sometimes misclassified.
- Educational project; do not use as the only spam defence in production.

## Dataset and credits

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the study of SMS spam filtering.*
ACM DocEng. Dataset: UCI Machine Learning Repository, "SMS Spam Collection".
