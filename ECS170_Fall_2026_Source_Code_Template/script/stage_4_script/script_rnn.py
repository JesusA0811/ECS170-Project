"""Train and evaluate the Stage 4 RNN sentiment classifier."""

import argparse
import json
from itertools import count
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN import Method_RNN, RECURRENT_LAYERS


def main(cells=("rnn",), improved=False, tuned=False):
    # Resolve the dataset path relative to this script.
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "stage_4_data" / "text_classification"

    print("Loading IMDb reviews...")
    loader = Dataset_Loader(data_path)
    splits = loader.load()
    if improved:
        # Select the epoch using held-out training reviews.
        training = splits["train"]
        train_ids, val_ids = train_test_split(
            range(len(training["y"])), test_size=0.1, random_state=2,
            stratify=training["y"],
        )
        for name, indices in (("train", train_ids), ("validation", val_ids)):
            splits[name] = {key: [values[i] for i in indices] for key, values in training.items()}

    result_dir = project_root / "result" / "stage_4_result"
    for cell in cells:
        run_classifier(cell, splits, len(loader.vocabulary), result_dir, improved, tuned)


def run_classifier(cell, splits, vocab_size, result_dir, improved=False, tuned=False):
    # Reset the seed for each model so batch shuffling can be controlled.
    torch.manual_seed(2)
    # The embedding needs one row for every ID in the loader's vocabulary.
    model = Method_RNN(
        vocab_size=vocab_size, cell=cell,
        embedding_size=128 if improved else 64,
        hidden_size=128 if improved else 64,
        pooling="mean" if improved else "last",
    )
    if improved:
        model.max_epoch = 10
        model.max_review_length = 400
    if tuned:
        # Drop word features during training to reduce memorization.
        model.embedding_dropout.p = 0.3
        model.dropout.p = 0.5
        model.max_review_length = 600
        model.weight_decay = 1e-5
        model.max_epoch = 15
        model.patience = 4
    model.data = splits
    results = model.run()

    # Reuse the Stage 2 evaluator for accuracy, precision, recall, and F1.
    evaluator = Evaluate_Accuracy(f"{model.method_name} test metrics", "")
    evaluator.data = results
    evaluator.evaluate()

    plot_path = save_loss_plot(
        model.loss_history,
        result_dir,
        f"{cell}_classification{'_tuned' if tuned else '_improved' if improved else ''}_training_loss",
        f"{model.method_name} Sentiment Classifier Training Loss",
    )
    print("Saved training loss plot:", plot_path)
    precision, recall, f1, _ = precision_recall_fscore_support(
        results["true_y"], results["pred_y"], average="weighted", zero_division=0
    )
    metrics = {
        "cell": cell, "improved": improved,
        "tuned": tuned,
        "accuracy": accuracy_score(results["true_y"], results["pred_y"]),
        "precision": precision, "recall": recall, "f1": f1,
        "training_loss": model.loss_history,
        "validation_accuracy": model.validation_history,
        "selected_epoch": getattr(model, "best_epoch", model.max_epoch),
        "max_review_length": model.max_review_length,
        "epochs_completed": len(model.loss_history),
        "selection": "best validation accuracy" if improved else "final epoch",
        "configuration": {
            "embedding_size": model.embedding.embedding_dim,
            "hidden_size": model.rnn.hidden_size,
            "layers": model.rnn.num_layers,
            "pooling": model.pooling,
            "dropout": model.dropout.p,
            "embedding_dropout": model.embedding_dropout.p,
            "weight_decay": model.weight_decay,
            "learning_rate": model.learning_rate,
            "batch_size": model.batch_size,
            "max_epochs": model.max_epoch,
            "patience": model.patience,
            "seed": 2,
            "train_examples": len(splits["train"]["y"]),
            "validation_examples": len(splits.get("validation", {"y": []})["y"]),
            "test_examples": len(splits["test"]["y"]),
        },
    }
    plot_path.with_suffix(".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Save the selected weights alongside the experiment metrics.
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, plot_path.with_suffix(".pth"))


def save_loss_plot(loss_history, result_dir, filename, title):
    # Save the average training loss by epoch for the report.
    result_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(loss_history) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss_history, marker="o")
    if len(epochs) <= 10:
        ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.grid(True)

    # Open the next available filename exclusively so runs cannot overwrite a plot.
    for run_number in count(1):
        plot_path = result_dir / f"{filename}_{run_number}.png"
        try:
            plot_file = plot_path.open("xb")
        except FileExistsError:
            continue
        with plot_file:
            ax.set_title(f"{title} (Run {run_number})")
            fig.savefig(plot_file, format="png", dpi=300, bbox_inches="tight")
        break

    plt.close(fig)
    return plot_path


def parse_cells(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cell", nargs="+", choices=RECURRENT_LAYERS, default=["rnn"])
    return parser.parse_args().cell


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", nargs="+", choices=RECURRENT_LAYERS, default=["rnn"])
    parser.add_argument("--improved", action="store_true")
    parser.add_argument("--tuned", action="store_true")
    args = parser.parse_args()
    main(args.cell, args.improved or args.tuned, args.tuned)
