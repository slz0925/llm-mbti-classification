"""Evaluate a selected classifier checkpoint on the held-out test split."""

import argparse
import json
from pathlib import Path

from data import FeatureCache, TASK_COLUMN, atomic_file, encoder_settings, write_json
from train import load_checkpoint, metrics, predict_classes, resolve_device


def evaluate_checkpoint(feature_dir, model, state, device, batch_size):
    cache = FeatureCache(feature_dir, state["dataset"])
    if cache.feature_dim != state["feature_dim"]:
        raise ValueError("Test cache feature width does not match the checkpoint")
    trained_encoder = encoder_settings(state["encoder"])
    if cache.encoder is None or trained_encoder is None:
        matching_encoder = cache.encoder is trained_encoder
    else:
        matching_encoder = all(trained_encoder.get(key) == value for key, value in cache.encoder.items())
    cache_id = state["cache_id"] if state["version"] == 2 else None
    if (not matching_encoder or cache.cache_id != cache_id
            or (state["version"] == 2 and cache.split_rows != state["split_rows"])):
        raise ValueError("Test cache metadata does not match the checkpoint's training cache")
    test = cache.load("test")
    if test.ids:
        seen = set()
        for split in ("train", "validation"):
            _, ids = cache.load_split_labels(split)
            if seen.intersection(ids):
                raise ValueError("Train and validation contain overlapping author IDs")
            seen.update(ids)
        if seen.intersection(test.ids):
            raise ValueError("Test and development splits contain overlapping author IDs")
    targets = test.labels[:, TASK_COLUMN[state["task"]]]
    predictions = predict_classes(model, test.features, device, batch_size, feature_dim=state["feature_dim"])
    majority = [state["training"]["majority_class"]] * len(targets)
    report = {
        "dataset": state["dataset"], "classifier": state["classifier"], "task": state["task"],
        "classes": state["classes"], "split": "test",
        "metrics": metrics(targets, predictions, len(state["classes"])),
        "majority_baseline": metrics(targets, majority, len(state["classes"])),
    }
    if "epoch" in state["training"]:
        report["selected_epoch"] = state["training"]["epoch"]
    else:
        report["selected_parameters"] = state["training"]["parameters"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path, help="Local feature cache directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow loading a trusted .joblib file")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == args.checkpoint.resolve():
            raise ValueError("Evaluation output must differ from the checkpoint path")
        device = resolve_device(args.device)
        model, state = load_checkpoint(args.checkpoint, device, trust_checkpoint=args.trust_checkpoint)
        report = evaluate_checkpoint(args.features, model, state, device, args.batch_size)
        if args.output:
            with atomic_file(args.output) as temporary:
                write_json(temporary, report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
