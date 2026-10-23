"""Classify new messages from the command line.

Examples:
    python predict.py "Congratulations! You won a free prize, call now to claim"
    python predict.py            (interactive mode, type 'quit' to exit)
"""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.predictor import SpamPredictor


def show(pred) -> None:
    print(f"  -> {pred.label}  (spam probability = {pred.spam_probability:.4f}, "
          f"confidence = {pred.confidence:.1%})")


def main() -> None:
    predictor = SpamPredictor()
    if len(sys.argv) > 1:
        for p in predictor.predict([" ".join(sys.argv[1:])]):
            show(p)
        return
    print("Spam detector - type a message and press Enter (type 'quit' to exit).")
    while True:
        text = input("\nMessage> ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            show(predictor.predict([text])[0])


if __name__ == "__main__":
    main()
