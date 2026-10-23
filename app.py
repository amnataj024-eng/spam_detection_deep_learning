"""Streamlit web app for the spam detector.

Run with:
    streamlit run app.py
"""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st

from src import config as C
from src.predictor import SpamPredictor

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

EXAMPLES = {
    "Prize scam": "Congratulations! You have won a FREE £1000 prize. Call 09061701461 now to claim your reward.",
    "Phishing link": "URGENT! Your account has been suspended. Click http://bit.ly/verify-now to restore access.",
    "Friendly message": "Hey, are we still meeting for lunch tomorrow at 1? Let me know!",
    "Work message": "Can you send me the slides before the meeting this afternoon? Thanks.",
}


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> SpamPredictor:
    return SpamPredictor()


def sidebar() -> None:
    st.sidebar.header("About")
    st.sidebar.write(
        "Embedding + Bidirectional LSTM neural network trained on the "
        "SMS Spam Collection dataset (5,574 messages)."
    )
    if C.METRICS_PATH.exists():
        m = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
        st.sidebar.subheader("Test-set performance")
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        st.sidebar.metric("Spam precision", f"{m['precision_spam']:.2%}")
        st.sidebar.metric("Spam recall", f"{m['recall_spam']:.2%}")
        st.sidebar.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.sidebar.caption("Educational project - not a replacement for a production spam filter.")


def show_result(pred) -> None:
    if pred.is_spam:
        st.error(f"🚨 **SPAM** - confidence {pred.confidence:.1%}")
    else:
        st.success(f"✅ **HAM (Not Spam)** - confidence {pred.confidence:.1%}")
    st.progress(min(max(pred.spam_probability, 0.0), 1.0), text=f"Spam probability: {pred.spam_probability:.1%}")


def main() -> None:
    st.title("📩 Spam Detection using Deep Learning")
    st.write("Paste an SMS or e-mail text and the neural network will classify it as **Spam** or **Ham**.")
    sidebar()

    try:
        predictor = get_predictor()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()

    tab_single, tab_batch = st.tabs(["Single message", "Batch (CSV)"])

    with tab_single:
        example = st.selectbox("Try an example (optional)", ["-"] + list(EXAMPLES))
        default = EXAMPLES.get(example, "")
        text = st.text_area("Message", value=default, height=150, key=f"msg_{example}",
                            placeholder="Type or paste a message here...")
        if st.button("Classify", type="primary"):
            if not text.strip():
                st.warning("Please enter a message first.")
            else:
                show_result(predictor.predict([text])[0])

    with tab_batch:
        st.write("Upload a CSV file with a column named **text**. Results can be downloaded.")
        file = st.file_uploader("CSV file", type="csv")
        if file is not None:
            df = pd.read_csv(file)
            if "text" not in df.columns:
                st.error("The CSV must contain a column named 'text'.")
            else:
                preds = predictor.predict(df["text"].astype(str).tolist())
                df["prediction"] = [p.label for p in preds]
                df["spam_probability"] = [round(p.spam_probability, 4) for p in preds]
                st.dataframe(df, use_container_width=True)
                st.download_button("Download results", df.to_csv(index=False).encode("utf-8"),
                                   "spam_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
