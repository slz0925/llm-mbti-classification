"""Predict MBTI labels from cached features or one author's raw text."""

import argparse
import json
from pathlib import Path

from data import LEGACY_FEATURE_DIM, POOLING, SPLITS, encoder_settings, legacy_feature_definition
from data import read_features, read_manifest
from extract_features import LLMExtractor
from prepare_data import prepare_text
from train import load_checkpoint, predict_classes, resolve_device


def predict_text(model, state, posts, extractor, device):
    text = prepare_text(state["dataset"], posts)
    if not text:
        raise ValueError("No usable text remains after dataset preprocessing")
    features = extractor.encode([text])
    return predict_classes(model, features, device, 1, feature_dim=state["feature_dim"])


def prediction_rows(state, predictions):
    return [
        {"row": row, "task": state["task"], "class_id": label, "label": state["classes"][label]}
        for row, label in enumerate(predictions, start=1)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--trust-checkpoint", action="store_true", help="Allow loading a trusted .joblib file")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--features", type=Path, help="Feature CSV with optional sample IDs")
    inputs.add_argument("--text", action="append", help="One post, repeat for more posts from the same author")
    inputs.add_argument("--text-file", type=Path, help="UTF-8 text file containing one post")
    parser.add_argument("--model-dir", type=Path, help="Local Transformers model directory for raw text")
    parser.add_argument("--encoder-device", default="auto")
    parser.add_argument("--encoder-dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--pooling", choices=POOLING, help="Defaults to the checkpoint's pooling strategy")
    parser.add_argument("--device", default="auto", help="Classifier device")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.features is None and args.model_dir is None:
        parser.error("Raw text prediction requires --model-dir")
    try:
        device = resolve_device(args.device)
        model, state = load_checkpoint(args.checkpoint, device, trust_checkpoint=args.trust_checkpoint)
        if args.features is not None:
            directory = args.features.parent
            if args.features.stem in SPLITS and (directory / "manifest.json").is_file():
                manifest = read_manifest(directory, "mbti-features")
                actual = encoder_settings(manifest["encoder"])
                expected = state["encoder"]
                if actual is None or expected is None or any(actual.get(k) != v for k, v in expected.items()):
                    raise ValueError("Feature cache encoder settings do not match the checkpoint")
            features, _ = read_features(args.features, feature_dim=state["feature_dim"])
            predictions = predict_classes(model, features, device, args.batch_size, feature_dim=state["feature_dim"])
        else:
            posts = args.text if args.text is not None else [args.text_file.read_text(encoding="utf-8-sig")]
            encoder = state["encoder"]
            if encoder is None:
                if state["feature_dim"] != LEGACY_FEATURE_DIM:
                    raise ValueError("Raw text prediction requires encoder settings in the checkpoint")
                encoder = {**legacy_feature_definition(), "dtype": args.encoder_dtype or "bfloat16"}
            dtype = args.encoder_dtype or encoder["dtype"]
            extractor = LLMExtractor(
                args.model_dir, args.encoder_device, dtype, expected=encoder, pooling=args.pooling,
            )
            predictions = predict_text(model, state, posts, extractor, device)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    for row in prediction_rows(state, predictions):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
