"""Cross-dataset generalization check: train on Devign, test on BigVul.

This script does NOT train anything. It loads a checkpoint produced by the
normal `devign-experiment` training run (the .pt file saved under
runs/devign/<strategy>/<seed>/best_model.pt) and evaluates it on BigVul's
already-processed test split, with no fine-tuning.

Usage (run from the project root, same place you run `devign-experiment`):

    python scripts/cross_dataset_eval.py \
        --checkpoint runs/devign/baseline/42/best_model.pt \
        --config runs/devign/baseline/42/config.json \
        --target-dataset bigvul \
        --output runs/cross_dataset/devign_baseline_42_on_bigvul.json

Run it once per checkpoint you want to test (e.g. once for baseline,
once for class_weighted).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from devign_imbalance.data import load_split, IndexedJsonlGraphDataset, validate_graphs
from devign_imbalance.metrics import binary_metrics
from devign_imbalance.model import Devign


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
def predict(model, loader, device) -> tuple[list[int], list[float]]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = batch.to(device)
        probability = torch.sigmoid(model(batch))
        labels.extend(batch.y.int().cpu().tolist())
        probabilities.extend(probability.cpu().tolist())
    return labels, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to best_model.pt from a completed Devign training run")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to the config.json saved alongside that checkpoint "
                             "(used to rebuild the model with matching dimensions)")
    parser.add_argument("--target-dataset", default="bigvul", choices=["bigvul", "reveal", "devign"],
                        help="Which processed dataset's TEST split to evaluate on")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"),
                        help="Root folder containing <dataset>/test.jsonl etc.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold, kept at 0.5 to match all other reported runs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write the resulting metrics JSON")
    args = parser.parse_args()

    device = resolve_device(args.device)

    # Rebuild the exact same architecture the checkpoint was trained with.
    train_config = json.loads(args.config.read_text(encoding="utf-8"))
    model_cfg = train_config["model"]
    model = Devign(
        input_dim=int(model_cfg["input_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_steps=int(model_cfg["num_steps"]),
        num_edge_types=int(model_cfg["num_edge_types"]),
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    # Load the TARGET dataset's test split only. We never touch its train/valid
    # splits, and we never fine-tune -- this is a pure zero-shot transfer check.
    target_root = args.data_root / args.target_dataset
    test_graphs = load_split(target_root, "test")
    if isinstance(test_graphs, IndexedJsonlGraphDataset):
        if test_graphs.feature_dim != int(model_cfg["input_dim"]):
            raise ValueError(
                f"{args.target_dataset} feature width ({test_graphs.feature_dim}) does not "
                f"match the source model's input_dim ({model_cfg['input_dim']}); "
                "cross-dataset comparison requires identical feature dimensions."
            )
        if test_graphs.num_edge_types != int(model_cfg["num_edge_types"]):
            raise ValueError(
                f"{args.target_dataset} edge vocabulary ({test_graphs.num_edge_types}) does not "
                f"match the source model's num_edge_types ({model_cfg['num_edge_types']})."
            )
        validate_graphs([test_graphs[0], test_graphs[len(test_graphs) - 1]],
                        int(model_cfg["input_dim"]), int(model_cfg["num_edge_types"]))
    else:
        validate_graphs(test_graphs, int(model_cfg["input_dim"]), int(model_cfg["num_edge_types"]))

    loader = DataLoader(
        test_graphs,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    y_true, y_prob = predict(model, loader, device)
    metrics = binary_metrics(y_true, y_prob, args.threshold)

    result = {
        "source_dataset": train_config["data"]["dataset"],
        "source_strategy": train_config["training"]["strategy"],
        "source_seed": train_config["seed"],
        "target_dataset": args.target_dataset,
        "evaluation_type": "zero_shot_cross_dataset",
        "note": "Model trained on source_dataset only; no fine-tuning on target_dataset.",
        **metrics,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
