"""Train the Stage 4 RNN joke generator and compare sample continuations."""

import json
import re
from pathlib import Path

import torch

from local_code.stage_4_code.Joke_Dataset_Loader import Joke_Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from script.stage_4_script.script_rnn import parse_cells, save_loss_plot


def main(cells=("rnn",)):
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "stage_4_data" / "text_generation" / "data"
    result_dir = project_root / "result" / "stage_4_result"

    loader = Joke_Dataset_Loader(data_path)
    data = loader.load()
    for cell in cells:
        run_generator(cell, loader, data, result_dir)


def run_generator(cell, loader, data, result_dir):
    torch.manual_seed(2)
    model = Method_RNN_Generator(vocab_size=len(loader.vocabulary), cell=cell)
    print(f"Training {model.method_name} on {len(data['jokes']):,} jokes...")
    model.data = data
    model.run()

    plot_path = save_loss_plot(
        model.loss_history,
        result_dir,
        f"{cell}_generation_training_loss",
        f"{model.method_name} Joke Generator Training Loss",
    )
    print("Saved training loss plot:", plot_path)

    comparisons = []
    # Use three-word openings from the training corpus for the assignment comparison.
    for prompt in ("what did the", "don't you hate", "two artists had"):
        prompt_ids = loader.encode(loader.tokenize(prompt))
        generated, ended = model.generate(prompt_ids)
        words = " ".join(loader.id_to_token[index] for index in generated)
        text = re.sub(r" +([.,!?;:])", r"\1", words)

        references = [
            (ids, joke)
            for ids, joke in zip(data["X"], data["jokes"], strict=True)
            if ids[:len(prompt_ids)] == prompt_ids
        ]
        exact_match = next((joke for ids, joke in references if ids == generated), None)
        reference = exact_match or references[0][1]
        stop = "end token" if ended else "length limit"
        comparisons.append(
            f"Prompt: {prompt}\n"
            f"Generated: {text}\n"
            f"Original: {reference}\n"
            f"Matching training openings: {len(references)}\n"
            f"Exact training match: {'yes' if exact_match else 'no'}\n"
            f"Stopped at: {stop}"
        )

    report = "\n\n".join(comparisons)
    report_path = plot_path.with_name(
        plot_path.stem.replace("training_loss", "samples") + ".txt"
    )
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print("Saved generation samples:", report_path)
    # Save weights and vocabulary so the trained generator can be used again.
    configuration = {
        "cell": cell,
        "embedding_size": model.embedding.embedding_dim,
        "hidden_size": model.rnn.hidden_size,
        "layers": model.rnn.num_layers,
        "epochs": model.max_epoch,
        "batch_size": model.batch_size,
        "learning_rate": model.learning_rate,
        "seed": 2,
        "jokes": len(data["jokes"]),
        "vocabulary_size": len(loader.vocabulary),
        "decoding": "greedy",
        "max_new_tokens": 60,
        "training_loss": model.loss_history,
        "prompts": ["what did the", "don't you hate", "two artists had"],
    }
    plot_path.with_suffix(".json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": loader.vocabulary,
            "id_to_token": loader.id_to_token,
            "configuration": configuration,
        },
        plot_path.with_suffix(".pth"),
    )


if __name__ == "__main__":
    main(parse_cells(__doc__))
