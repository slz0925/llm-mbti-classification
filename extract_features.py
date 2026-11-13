"""Extract and cache a final-layer representation for each author."""

import argparse
import csv
import json
import re
from pathlib import Path
from uuid import uuid4

import torch

from data import (
    LABEL_COLUMNS, LEGACY_MODEL_ID, MAX_TOKENS, POOLING, SPLITS, encode_labels,
    encoder_settings, feature_columns, new_directory, read_prepared, write_json,
)


def add_model_source_arguments(parser, *, required=False):
    sources = parser.add_mutually_exclusive_group(required=required)
    sources.add_argument("--model-dir", type=Path, help="Local Transformers model directory")
    sources.add_argument("--model-id", help="Hugging Face Hub model ID")


def add_encoder_arguments(parser):
    add_model_source_arguments(parser, required=True)
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


def load_encoder_asset(loader, source, options, **kwargs):
    try:
        return loader(source, **options, **kwargs)
    except (OSError, ValueError) as exc:
        if "revision" in options:
            raise ValueError(f"Could not load Hub model {source} at revision {options['revision']}: {exc}") from exc
        raise


class LLMExtractor:
    def __init__(self, model_dir=None, device="auto", dtype="bfloat16", expected=None, *,
                 model_id=None, pooling=None, max_tokens=None, trust_remote_code=False):
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if (model_dir is None) == (model_id is None):
            raise ValueError("Specify exactly one of --model-dir and --model-id")
        expected = encoder_settings(expected)
        expected_revision = expected.get("revision") if expected else None
        revision = None
        options = {"local_files_only": model_dir is not None, "trust_remote_code": trust_remote_code}
        if model_dir is not None:
            source = Path(model_dir).resolve()
            if not (source / "config.json").is_file():
                raise ValueError(f"Expected a local Transformers model directory: {source}")
            name = source.name
            repository = source.parent.parent.name
            if source.parent.name == "snapshots" and repository.startswith("models--"):
                name = repository.removeprefix("models--").replace("--", "/")
                if re.fullmatch(r"[0-9a-f]{40}", source.name):
                    revision = source.name
            elif name.casefold() == LEGACY_MODEL_ID.casefold().split("/")[-1]:
                name = LEGACY_MODEL_ID
        else:
            from huggingface_hub.utils import validate_repo_id

            validate_repo_id(model_id)
            # Transformers also accepts local paths in from_pretrained().
            if Path(model_id).exists():
                raise ValueError(f"A local path shadows Hub model ID {model_id}; run from another directory")
            source = name = model_id

        if max_tokens is None:
            max_tokens = expected["max_tokens"] if expected else MAX_TOKENS
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if expected and max_tokens != expected["max_tokens"]:
            raise ValueError("Encoder max_tokens does not match the classifier's feature cache")

        if expected is not None:
            trained_name = expected["model_id"]
            legacy = "model_type" not in expected and trained_name == LEGACY_MODEL_ID
            # A regular local folder may carry only the model name.
            same_name = name == trained_name or (
                ("/" not in name or "/" not in trained_name)
                and name.rsplit("/", 1)[-1] == trained_name.rsplit("/", 1)[-1]
            )
            if not same_name and not (legacy and model_dir is not None):
                raise ValueError("Encoder model_id does not match the classifier's feature cache")
            if revision and expected_revision and revision != expected_revision:
                raise ValueError(f"Encoder revision {revision} does not match checkpoint revision {expected_revision}")

        if model_id is not None:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RevisionNotFoundError

            # The Hub returns the configuration path in its resolved commit snapshot.
            try:
                config_path = Path(hf_hub_download(model_id, "config.json", revision=expected_revision))
            except (HfHubHTTPError, OSError, ValueError) as exc:
                requested = f"revision {expected_revision}" if expected_revision else "its default revision"
                if isinstance(exc, RevisionNotFoundError):
                    detail = "revision not found"
                elif isinstance(exc, GatedRepoError):
                    detail = "access restricted; check your Hugging Face access and sign in"
                elif isinstance(exc, HfHubHTTPError) and getattr(exc.response, "status_code", None) == 404:
                    detail = "revision or config.json not found"
                else:
                    detail = str(exc)
                raise ValueError(f"Could not resolve Hub model {model_id} at {requested}: {detail}") from exc
            revision = config_path.parent.name
            if not re.fullmatch(r"[0-9a-f]{40}", revision) or (expected_revision and revision != expected_revision):
                raise ValueError(f"Could not resolve the requested Hub revision for {model_id}: {revision}")
            options["revision"] = revision
        config = load_encoder_asset(AutoConfig.from_pretrained, source, options)
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
        self.tokenizer = load_encoder_asset(AutoTokenizer.from_pretrained, source, options, padding_side="right")
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
        if revision:
            self.provenance["revision"] = revision
        if expected is not None:
            if legacy and config.model_type != "llama":
                raise ValueError("Historical features require a compatible Llama-3-8B encoder")
            for field, value in expected.items():
                if field == "tokenizer_class" and self.provenance[field] != value:
                    raise ValueError(
                        f"Encoder tokenizer_class does not match the classifier's feature cache "
                        f"(expected {value}, got {self.provenance[field]}). "
                        "Use the same tokenizer implementation and Transformers version as for feature extraction"
                    )
                if field not in ("model_id", "revision") and self.provenance[field] != value:
                    raise ValueError(f"Encoder {field} does not match the classifier's feature cache")
        self.provenance["transformers_version"] = transformers.__version__
        self.provenance["torch_version"] = str(torch.__version__)
        placement = "auto" if device == "auto" else {"": str(torch.device(device))}
        try:
            self.model = load_encoder_asset(
                AutoModelForCausalLM.from_pretrained, source, options,
                config=config, dtype=getattr(torch, dtype), device_map=placement,
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
            args.model_dir, args.encoder_device, args.encoder_dtype, model_id=args.model_id,
            pooling=args.pooling, max_tokens=args.max_tokens, trust_remote_code=args.trust_remote_code,
        )
        counts = extract_cache(args.prepared, args.output, extractor, args.batch_size)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
