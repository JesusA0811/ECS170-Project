"""Train a recurrent model to predict the next token in a joke."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from local_code.base_class.method import method
from local_code.stage_4_code.Joke_Dataset_Loader import Joke_Dataset_Loader
from local_code.stage_4_code.Method_RNN import RECURRENT_LAYERS


class Method_RNN_Generator(method, nn.Module):
    def __init__(self, vocab_size, embedding_size=128, hidden_size=128, cell="rnn"):
        method.__init__(self, cell.upper(), "Joke generator")
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=Joke_Dataset_Loader.PAD_ID
        )
        self.rnn = RECURRENT_LAYERS[cell](embedding_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)

        self.max_epoch = 150
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @staticmethod
    def _collate_batch(batch):
        inputs, targets = zip(*batch)
        # Pad inputs and next-token targets to the longest joke in the batch.
        return (
            pad_sequence([torch.tensor(ids) for ids in inputs], batch_first=True),
            pad_sequence([torch.tensor(ids) for ids in targets], batch_first=True),
        )

    def forward(self, tokens, hidden=None):
        # Produce vocabulary scores at every position for next-token training.
        states, hidden = self.rnn(self.embedding(tokens), hidden)
        return self.output(states), hidden

    def fit(self, batches):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss(
            ignore_index=Joke_Dataset_Loader.PAD_ID, reduction="sum"
        )
        self.loss_history = []

        for epoch in range(1, self.max_epoch + 1):
            total_loss = 0.0
            total_tokens = 0
            for tokens, targets in batches:
                # Count real tokens so padding never affects the reported loss.
                token_count = targets.ne(Joke_Dataset_Loader.PAD_ID).sum().item()
                tokens = tokens.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits, _ = self(tokens)
                batch_loss = loss_function(logits.transpose(1, 2), targets)
                (batch_loss / token_count).backward()
                optimizer.step()
                total_loss += batch_loss.item()
                total_tokens += token_count

            loss = total_loss / total_tokens
            self.loss_history.append(loss)
            print(f"Epoch {epoch}/{self.max_epoch} | Training loss: {loss:.4f}")

    def run(self):
        print("Using device:", self.device)
        batches = DataLoader(
            list(zip(self.data["X"], self.data["y"], strict=True)),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_batch,
        )
        self.fit(batches)

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=60):
        self.eval()
        tokens = torch.tensor([prompt_ids], device=self.device)
        hidden = None
        generated = list(prompt_ids)

        for _ in range(max_new_tokens):
            logits, hidden = self(tokens, hidden)
            scores = logits[0, -1]
            scores[Joke_Dataset_Loader.PAD_ID] = -torch.inf
            scores[Joke_Dataset_Loader.UNK_ID] = -torch.inf
            next_id = scores.argmax().item()
            if next_id == Joke_Dataset_Loader.EOS_ID:
                return generated, True

            generated.append(next_id)
            # Carry the hidden state forward and feed back only the new token.
            tokens = torch.tensor([[next_id]], device=self.device)

        return generated, False
