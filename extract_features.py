"""Extract and cache a final-layer representation for each author."""

import argparse
import csv
import json
from pathlib import Path
from uuid import uuid4

import torch

from data import (
    LABEL_COLUMNS, LEGACY_MODEL_ID, MAX_TOKENS, POOLING, SPLITS, encode_labels,
    encoder_settings, feature_columns, new_directory, read_prepared, write_json,
)


def add_encoder_arguments(parser):
    parser.add_argument("--model-dir", type=Path, required=True, help="Local Transformers model directory")
    parser.add_argument("--encoder-device", default="auto", help="auto, cpu or a CUDA device such as cuda:0")
    parser.add_argument("--encoder-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--pooling", choices=POOLING, default="last-position")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Maximum input length in tokens (default: {MAX_TOKENS})")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow execution of reviewed custom model code")


def model_input_device(model):
    embeddings = model.get_input_embeddings()
    device = embeddings.weight.device
    if device.type == "meta":
        for module in (embeddings, model):
            execution_device = getattr(getattr(module, "_hf_hook", None), "execution_device", None)
            if execution_device is not None:
                return execution_device
        raise ValueError("Could not resolve the execution device for the model's input embeddings")
    return device


class LLMExtractor:
    def __init__(self, model_dir, device="auto", dtype="bfloat16", expected=None, *,
                 pooling=None, max_tokens=None, trust_remote_code=False):
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_dir = Path(model_dir).resolve()
        if not (model_dir / "config.json").is_file():
            raise ValueError(f"Expected a local Transformers model directory: {model_dir}")
        expected = encoder_settings(expected)
        if max_tokens is None:
            max_tokens = expected["max_tokens"] if expected else MAX_TOKENS
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if expected and max_tokens != expected["max_tokens"]:
            raise ValueError("Encoder max_tokens does not match the classifier's feature cache")

        local = {"local_files_only": True, "trust_remote_code": trust_remote_code}
        config = AutoConfig.from_pretrained(model_dir, **local)
        if config.is_encoder_decoder:
            raise ValueError(f"{config.model_type} is an encoder–decoder model; a decoder-only causal LM is required")
        text_config = config.get_text_config()
        width = getattr(text_config, "hidden_size", None)
        if type(width) is not int or width < 1:
            raise ValueError(f"{config.model_type} does not declare a valid hidden_size")
        context = getattr(text_config, "max_position_embeddings", None)
        if type(context) is int and 0 < context < max_tokens:
            raise ValueError(f"max_tokens={max_tokens} exceeds the model's configured context of {context} tokens")
        method = POOLING[pooling] if pooling else (
            expected["pooling"] if expected else POOLING["last-position"]
        )
        name = model_dir.name
        repository = model_dir.parent.parent.name
        if model_dir.parent.name == "snapshots" and repository.startswith("models--"):
            name = repository.removeprefix("models--").replace("--", "/")
        if name.casefold().split("/")[-1] == LEGACY_MODEL_ID.casefold().split("/")[-1]:
            name = LEGACY_MODEL_ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right", **local)
        pad_policy = expected["pad_token"] if expected else (
            "eos" if method == POOLING["last-position"] and self.tokenizer.eos_token_id is not None
            else "native" if self.tokenizer.pad_token_id is not None else "eos"
        )
        if pad_policy == "eos":
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer needs a padding token or an EOS token for fixed-length inputs")
        self.provenance = {
            "model_id": name, "model_type": config.model_type, "feature_dim": width,
            "max_tokens": max_tokens, "padding_side": "right", "pad_token": pad_policy,
            "pad_token_id": self.tokenizer.pad_token_id, "pooling": method, "dtype": dtype,
            "tokenizer_class": type(self.tokenizer).__name__, "tokenizer_size": len(self.tokenizer),
        }
        if expected is not None:
            legacy = "model_type" not in expected and expected["model_id"] == LEGACY_MODEL_ID
            if legacy and config.model_type != "llama":
                raise ValueError("Historical features require a compatible Llama 3 8B encoder")
            for field, value in expected.items():
                actual = self.provenance[field]
                if field == "model_id":
                    if legacy:
                        continue
                    # A local folder carries the model name without a Hub namespace.
                    if "/" not in actual or "/" not in value:
                        actual, value = actual.rsplit("/", 1)[-1], value.rsplit("/", 1)[-1]
                if actual != value:
                    raise ValueError(f"Encoder {field} does not match the classifier's feature cache")
        self.provenance["transformers_version"] = transformers.__version__
        self.provenance["torch_version"] = str(torch.__version__)
        placement = "auto" if device == "auto" else {"": str(torch.device(device))}
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, config=config, dtype=getattr(torch, dtype), device_map=placement, **local,
            ).eval()
        except ValueError as exc:
            raise ValueError(f"Could not load {config.model_type} as a Transformers causal LM: {exc}") from exc
        self.device = model_input_device(self.model)
        self.feature_dim = width

    @torch.inference_mode()
    def encode(self, texts):
        tokens = self.tokenizer(
            texts, return_tensors="pt", padding="max_length",
            max_length=self.provenance["max_tokens"], truncation=True,
        )
        mask = tokens["attention_mask"].to(self.device)
        result = self.model(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=mask,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        states = getattr(result, "hidden_states", None)
        if (not isinstance(states, (tuple, list)) or not states
                or not torch.is_tensor(states[-1]) or states[-1].ndim != 3
                or states[-1].shape[:2] != mask.shape):
            raise ValueError("Encoder must return token-level hidden states for the input sequence")
        hidden = states[-1]
        if self.provenance["pooling"] == POOLING["last-position"]:
            features = hidden[:, -1, :]
        else:
            mask = mask.to(hidden.device)
            positions = torch.arange(hidden.shape[1], device=hidden.device)
            last = torch.where(mask.bool(), positions, -1).amax(dim=1)
            if (last < 0).any():
                raise ValueError("Cannot pool an input with no unmasked tokens")
            features = hidden[torch.arange(len(texts), device=hidden.device), last]
        features = features.to(device="cpu", dtype=torch.float32)
        if features.shape != (len(texts), self.feature_dim) or not torch.isfinite(features).all():
            raise ValueError("Encoder returned invalid author representations")
        return features.numpy()


def extract_cache(prepared, output, extractor, batch_size):
    if batch_size < 1:
        raise ValueError("Encoder batch size must be positive")
    prepared_manifest, splits = read_prepared(prepared)
    width = encoder_settings(extractor.provenance)["feature_dim"]
    manifest = {
        "format": "mbti-features", "version": 2, "dataset": prepared_manifest["dataset"],
        "cache_id": uuid4().hex, "split_seed": prepared_manifest["split_seed"],
        "encoder": {**extractor.provenance, "batch_size": batch_size}, "splits": {},
    }
    with new_directory(output) as destination:
        for split in SPLITS:
            feature_path = destination / f"{split}.csv"
            label_path = destination / f"{split}_label.csv"
            rows = splits[split]
            with (feature_path.open("w", encoding="utf-8", newline="") as feature_file,
                  label_path.open("w", encoding="utf-8", newline="") as label_file):
                features = csv.writer(feature_file)
                labels = csv.writer(label_file)
                features.writerow(("sample_id", *feature_columns(width)))
                labels.writerow(("sample_id", *LABEL_COLUMNS))
                for start in range(0, len(rows), batch_size):
                    batch = rows[start:start + batch_size]
                    vectors = extractor.encode([row["text"] for row in batch])
                    if vectors.shape != (len(batch), width):
                        raise ValueError("Encoder feature width does not match its cache metadata")
                    features.writerows((row["id"], *vector) for row, vector in zip(batch, vectors, strict=True))
                    labels.writerows((row["id"], *encode_labels(row["mbti"])) for row in batch)
            manifest["splits"][split] = {"rows": len(rows)}
        write_json(destination / "manifest.json", manifest)
    return {split: len(rows) for split, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    add_encoder_arguments(parser)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Use a positive batch size and a new output directory")
    try:
        extractor = LLMExtractor(
            args.model_dir, args.encoder_device, args.encoder_dtype, pooling=args.pooling,
            max_tokens=args.max_tokens, trust_remote_code=args.trust_remote_code,
        )
        counts = extract_cache(args.prepared, args.output, extractor, args.batch_size)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
