"""Plot helpers (saved as PNG files for the README and the report)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend: works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


def plot_class_distribution(df, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].map({0: "Ham", 1: "Spam"}).value_counts()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette={"Ham": "#4C9F70", "Spam": "#D1495B"}, legend=False, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, str(v), ha="center")
    ax.set_title("Class distribution"); ax.set_ylabel("Messages")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_length_distribution(df, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tmp = df.assign(words=df["text"].str.split().str.len(),
                    Class=df["label"].map({0: "Ham", 1: "Spam"}))
    sns.histplot(data=tmp, x="words", hue="Class", bins=40, element="step",
                 palette={"Ham": "#4C9F70", "Spam": "#D1495B"}, ax=ax)
    ax.set_title("Message length (words)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_history(history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0].plot(epochs, history["loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return float(score)
