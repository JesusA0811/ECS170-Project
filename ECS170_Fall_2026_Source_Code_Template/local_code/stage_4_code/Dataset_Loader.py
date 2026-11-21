"""Load and encode the Stage 4 movie reviews."""

import re
from collections import Counter
from html import unescape
from pathlib import Path

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    # Padding fills batches. Unknown marks words outside the training vocabulary.
    PAD_ID = 0
    UNK_ID = 1
    LABELS = {"neg": 0, "pos": 1}

    def __init__(self, dataset_path):
        super().__init__("IMDb movie reviews", "Binary sentiment classification")
        self.dataset_path = Path(dataset_path)
        self.vocabulary = {}

    @staticmethod
    def tokenize(text):
        # Decode HTML entities and normalize case and apostrophes before splitting.
        text = unescape(text).lower().replace("’", "'")
        # Treat HTML line breaks as spaces so adjacent words stay separate.
        text = re.sub(r"<br\s*/?>", " ", text)
        # Keep contractions such as "didn't" intact.
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)

    def _read_split(self, split):
        reviews = []
        labels = []
        # The folder name supplies the sentiment label.
        for folder, label in self.LABELS.items():
            label_path = self.dataset_path / split / folder
            # Sort filenames so reviews are loaded in the same order each run.
            files = sorted(label_path.glob("*.txt"))
            if not files:
                raise FileNotFoundError(f"No review files in {label_path}")
            for path in files:
                # Each file is one review, with one matching sentiment label.
                reviews.append(self.tokenize(path.read_text(encoding="utf-8")))
                labels.append(label)
        return reviews, labels

    def encode(self, tokens):
        ids = [self.vocabulary.get(token, self.UNK_ID) for token in tokens]
        # Give empty reviews one token so every sequence has a timestep.
        return ids or [self.UNK_ID]

    def load(self):
        train_reviews, train_labels = self._read_split("train")
        # Build the vocabulary from training reviews only.
        word_counts = Counter(word for review in train_reviews for word in review)
        self.vocabulary = {"<pad>": self.PAD_ID, "<unk>": self.UNK_ID}
        # Word IDs start at 2 because padding and unknown take IDs 0 and 1.
        for index, (word, _) in enumerate(word_counts.most_common(), start=2):
            self.vocabulary[word] = index

        train_ids = [self.encode(review) for review in train_reviews]
        # Encode test reviews with the same vocabulary used for training.
        test_reviews, test_labels = self._read_split("test")
        # X contains token sequences and y contains their labels. Pad later by batch.
        return {
            "train": {"X": train_ids, "y": train_labels},
            "test": {"X": [self.encode(review) for review in test_reviews], "y": test_labels},
        }
