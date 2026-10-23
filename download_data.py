"""Download the SMS Spam Collection dataset into data/ (only needed if the file is missing)."""
from src.data_loader import download_dataset

if __name__ == "__main__":
    print("Dataset ready at:", download_dataset())
