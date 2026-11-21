"""Plain RNN for binary sentiment classification."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from local_code.base_class.method import method
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader


RECURRENT_LAYERS = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}


class Method_RNN(method, nn.Module):
    def __init__(self, vocab_size, embedding_size=64, hidden_size=64, cell="rnn", pooling="last"):
        method.__init__(self, cell.upper(), "IMDb sentiment classifier")
        nn.Module.__init__(self)
        # Learn a vector for each word ID. The padding vector stays at zero.
        self.embedding = nn.Embedding(
            vocab_size, embedding_size, padding_idx=Dataset_Loader.PAD_ID
        )
        self.rnn = RECURRENT_LAYERS[cell](embedding_size, hidden_size, batch_first=True)
        # Output two scores per review, one for each sentiment label.
        self.output = nn.Linear(hidden_size, 2)
        self.pooling = pooling
        self.dropout = nn.Dropout(0.3 if pooling == "mean" else 0.0)
        self.embedding_dropout = nn.Dropout(0.0)
        self.weight_decay = 0.0

        self.max_epoch = 5
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.max_review_length = 256
        # Validation runs stop after three epochs without a better score.
        self.patience = 3
        # Use the NVIDIA GPU when PyTorch detects CUDA.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _collate_batch(self, batch):
        reviews, labels = zip(*batch)
        # Limit long reviews, then pad only to the longest review in this batch.
        sequences = [
            torch.tensor(review[:self.max_review_length], dtype=torch.long)
            for review in reviews
        ]
        # Save each review's length before padding adds extra positions.
        lengths = torch.tensor([len(sequence) for sequence in sequences])
        tokens = pad_sequence(
            sequences, batch_first=True, padding_value=Dataset_Loader.PAD_ID
        )
        return tokens, lengths, torch.tensor(labels, dtype=torch.long)

    def _batches(self, split, shuffle):
        return DataLoader(
            list(zip(split["X"], split["y"], strict=True)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_batch,
        )

    def forward(self, tokens, lengths):
        embeddings = self.embedding_dropout(self.embedding(tokens))
        # Keep the hidden state from each word position.
        states, _ = self.rnn(embeddings)
        # Read the hidden state at the last real word in each review.
        rows = torch.arange(tokens.size(0), device=tokens.device)
        representation = states[rows, lengths - 1]
        if self.pooling == "mean":
            # Average real word states, excluding the padded positions.
            mask = torch.arange(tokens.size(1), device=tokens.device)[None, :] < lengths[:, None]
            representation = (states * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1)
        return self.output(self.dropout(representation))

    def fit(self, batches, validation=None):
        self.train()
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Cross-entropy takes raw scores and the correct class ID.
        loss_function = nn.CrossEntropyLoss()
        self.loss_history = []
        self.validation_history = []
        best_accuracy = -1.0
        epochs_without_improvement = 0

        for epoch in range(1, self.max_epoch + 1):
            self.train()
            total_loss = 0.0
            for tokens, lengths, labels in batches:
                tokens = tokens.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                # Clear the previous gradients, then update weights for this batch.
                optimizer.zero_grad()
                loss = loss_function(self(tokens, lengths), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                # Weight each batch by its size to get the mean loss per review.
                total_loss += loss.item() * labels.size(0)

            average_loss = total_loss / len(batches.dataset)
            self.loss_history.append(average_loss)
            print(f"Epoch {epoch}/{self.max_epoch} | Training loss: {average_loss:.4f}")
            if validation is not None:
                predictions = self.test(self._batches(validation, shuffle=False))
                accuracy = (predictions == torch.tensor(validation["y"])).float().mean().item()
                self.validation_history.append(accuracy)
                print(f"Validation accuracy: {accuracy:.4f}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    epochs_without_improvement = 0
                    self.best_epoch = epoch
                    best_weights = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Early stopping after {self.patience} epochs without validation improvement.")
                    break

        if validation is not None:
            self.load_state_dict(best_weights)
            print(
                f"Restored epoch {self.best_epoch} weights "
                f"(best validation accuracy: {best_accuracy:.2%}) before testing."
            )

    @torch.no_grad()
    def test(self, batches):
        self.eval()
        predictions = []
        for tokens, lengths, _ in batches:
            logits = self(tokens.to(self.device), lengths.to(self.device))
            # Pick the class with the highest score and move it to the CPU.
            predictions.append(logits.argmax(dim=1).cpu())
        return torch.cat(predictions)

    def run(self):
        print("Using device:", self.device)
        print(f"Training {self.method_name}...")
        self.fit(self._batches(self.data["train"], shuffle=True), self.data.get("validation"))

        print(f"Testing {self.method_name}...")
        test_data = self.data["test"]
        # Keep test batches in order so predictions line up with the labels.
        return {
            "pred_y": self.test(self._batches(test_data, shuffle=False)),
            "true_y": test_data["y"],
        }
