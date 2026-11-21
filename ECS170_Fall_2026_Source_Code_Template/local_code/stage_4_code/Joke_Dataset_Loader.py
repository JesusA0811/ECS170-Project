"""Load the Stage 4 jokes for next-token prediction."""

import csv
import re
from collections import Counter
from pathlib import Path

from local_code.base_class.dataset import dataset


class Joke_Dataset_Loader(dataset):
    PAD_ID = 0
    UNK_ID = 1
    EOS_ID = 2

    def __init__(self, dataset_path):
        super().__init__("Short jokes", "Next-token prediction")
        self.dataset_path = Path(dataset_path)
        self.vocabulary = {}
        self.id_to_token = []

    @staticmethod
    def tokenize(text):
        text = text.lower().replace("’", "'")
        # Keep contractions and punctuation so generated jokes can include both.
        return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)

    def encode(self, tokens):
        return [self.vocabulary.get(token, self.UNK_ID) for token in tokens]

    def load(self):
        # The supplied file is CSV. Each row contains a joke ID and its text.
        with self.dataset_path.open(encoding="utf-8-sig", newline="") as source:
            jokes = [row["Joke"].strip() for row in csv.DictReader(source)]

        tokenized = [self.tokenize(joke) for joke in jokes]
        counts = Counter(token for joke in tokenized for token in joke)
        self.id_to_token = ["<pad>", "<unk>", "<eos>"]
        self.id_to_token.extend(token for token, _ in counts.most_common())
        self.vocabulary = {token: index for index, token in enumerate(self.id_to_token)}

        inputs = []
        targets = []
        for tokens in tokenized:
            ids = self.encode(tokens)
            # Shift targets by one token and teach the model when to stop.
            inputs.append(ids)
            targets.append(ids[1:] + [self.EOS_ID])

        # Keep the original text alongside each pair for comparing generated jokes.
        return {"X": inputs, "y": targets, "jokes": jokes}
