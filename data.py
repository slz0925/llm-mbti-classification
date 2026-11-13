"""MBTI labels and the local data and feature cache interfaces."""

import csv
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np


DATASETS = ("kaggle", "pandora")
SPLITS = ("train", "validation", "test")
LEGACY_FEATURE_DIM = 4096
MAX_TOKENS = 2048
LEGACY_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
POOLING = {
    "last-position": "last_hidden_layer_final_padded_position",
    "last-token": "last_hidden_layer_final_unmasked_position",
}
MBTI_TYPES = (
    "ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP",
    "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ",
)
TASKS = {
    "ie": ("I", "E"), "ns": ("N", "S"), "tf": ("T", "F"),
    "pj": ("P", "J"), "type": MBTI_TYPES,
}
LABEL_COLUMNS = tuple(f"label{i}" for i in range(len(TASKS)))
TASK_COLUMN = dict(zip(TASKS, range(len(TASKS))))


def encode_labels(mbti):
    return tuple(TASKS[task].index(mbti[i]) for i, task in enumerate(TASKS) if task != "type") + (
        MBTI_TYPES.index(mbti),
    )


def feature_columns(dim):
    return tuple(str(i) for i in range(dim))


def legacy_feature_definition():
    return {
        "model_id": LEGACY_MODEL_ID,
        "feature_dim": LEGACY_FEATURE_DIM,
        "max_tokens": MAX_TOKENS,
        "padding_side": "right",
        "pad_token": "eos",
        "pooling": POOLING["last-position"],
    }


def encoder_settings(encoder):
    """Select the metadata that affects feature compatibility."""
    if encoder is None:
        return None
    keys = (*legacy_feature_definition(), "dtype", "model_type", "pad_token_id",
            "tokenizer_class", "tokenizer_size")
    settings = {key: encoder[key] for key in keys if key in encoder}
    if (not isinstance(settings.get("model_id"), str) or not settings["model_id"]
            or type(settings.get("feature_dim")) is not int or settings["feature_dim"] < 1
            or settings.get("max_tokens") != MAX_TOKENS or settings.get("padding_side") != "right"
            or settings.get("pooling") not in POOLING.values()
            or settings.get("pad_token") not in ("eos", "native")
            or settings.get("dtype") not in ("bfloat16", "float32")):
        raise ValueError("Unsupported encoder feature definition")
    return settings


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextmanager
def new_directory(destination):
    """Publish a complete output directory after every file has been written."""
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".mbti-") as staging:
        result = Path(staging) / "result"
        result.mkdir()
        yield result
        result.rename(destination)


@contextmanager
def atomic_file(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".mbti-") as staging:
        temporary = Path(staging) / destination.name
        yield temporary
        os.replace(temporary, destination)


def read_manifest(directory, expected_format):
    path = Path(directory) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != expected_format or manifest.get("version") not in (1, 2):
        raise ValueError(f"Unsupported manifest format: {path}")
    if manifest.get("dataset") not in DATASETS or set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError(f"Invalid dataset or splits in {path}")
    return manifest


def read_prepared(directory):
    directory = Path(directory)
    manifest = read_manifest(directory, "mbti-prepared")
    splits = {}
    seen = set()
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        spec = manifest["splits"][split]
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if not rows or len(rows) != spec["rows"]:
            raise ValueError(f"Unexpected row count: {path}")
        for row in rows:
            if (not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen
                    or "\r" in row["id"] or "\n" in row["id"]
                    or row.get("mbti") not in MBTI_TYPES or not isinstance(row.get("text"), str)
                    or not row["text"].strip()):
                raise ValueError(f"Invalid sample or repeated author ID in {path}")
            seen.add(row["id"])
        splits[split] = rows
    return manifest, splits


def read_features(path, *, start=0, count=None, with_ids=None, feature_dim=None):
    """Read float32 vectors and any row IDs from a feature CSV."""
    ids = []

    def feature_values(rows):
        for row in rows:
            if row.startswith(b'"'):
                sample_id, *columns = next(csv.reader([row.decode("utf-8")], strict=True))
                values = ",".join(columns)
            else:
                raw_id, _, values = row.partition(b",")
                sample_id = raw_id.decode("utf-8")
            if not sample_id.strip() or not values:
                raise ValueError("Missing feature sample ID")
            ids.append(sample_id)
            yield values

    with Path(path).open("rb") as stream:
        header = stream.readline().decode("utf-8-sig").rstrip("\r\n")
        columns = tuple(next(csv.reader([header])))
        has_ids = columns[:1] == ("sample_id",)
        columns = columns[1:] if has_ids else columns
        dim = len(columns)
        if not dim or columns != feature_columns(dim) or (with_ids is not None and has_ids != with_ids):
            raise ValueError(f"Expected sequential feature columns starting at 0: {path}")
        if feature_dim is not None and dim != feature_dim:
            raise ValueError(f"Expected {feature_dim} feature columns, got {dim}: {path}")
        selected = islice(stream, start, None if count is None else start + count)
        if has_ids:
            selected = feature_values(selected)
        try:
            features = np.loadtxt(selected, delimiter=",", dtype=np.float32, ndmin=2, comments=None)
        except (ValueError, csv.Error) as exc:
            raise ValueError(f"Invalid feature data in {path}: {exc}") from exc
    if features.shape[1] != dim or not len(features) or not np.isfinite(features).all():
        raise ValueError(f"Expected non-empty, finite {dim}-column features: {path}")
    if count is not None and len(features) != count:
        raise ValueError(f"Expected {count} feature rows, got {len(features)}: {path}")
    if has_ids and (len(ids) != len(features) or len(set(ids)) != len(ids)):
        raise ValueError(f"Expected one unique sample ID per feature row: {path}")
    return features, ids


def read_labels(path, *, with_ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        expected = ("sample_id", *LABEL_COLUMNS) if with_ids else LABEL_COLUMNS
        if tuple(next(reader, ())) != expected:
            raise ValueError(f"Unexpected label columns: {path}")
        rows, ids = [], []
        for row in reader:
            if len(row) != len(expected):
                raise ValueError(f"Invalid label row {reader.line_num}: {path}")
            if with_ids:
                ids.append(row[0])
            try:
                values = tuple(float(value) for value in row[-len(LABEL_COLUMNS):])
                label = int(values[-1])
                if not 0 <= label < len(MBTI_TYPES) or values != encode_labels(MBTI_TYPES[label]):
                    raise ValueError("Label columns disagree")
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid MBTI labels on line {reader.line_num}: {path}") from exc
            rows.append(values)
    if not rows or (with_ids and (any(not value.strip() for value in ids) or len(set(ids)) != len(ids))):
        raise ValueError(f"Expected non-empty labels and unique sample IDs: {path}")
    return np.asarray(rows, dtype=np.int64), ids


def count_rows(path):
    # The supported numeric/label CSV layouts have exactly one physical line per sample.
    with Path(path).open("rb") as stream:
        stream.readline()
        return sum(1 for _ in stream)


@dataclass
class FeatureSplit:
    features: np.ndarray
    labels: np.ndarray
    ids: list[str]


class FeatureCache:
    """Read one split at a time from a versioned cache or the historical cumulative CSV."""

    def __init__(self, directory, dataset):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.directory = Path(directory)
        self.dataset = dataset
        manifest_path = self.directory / "manifest.json"
        self.manifest = None
        self.cache_id = None
        if manifest_path.is_file():
            self.manifest = read_manifest(self.directory, "mbti-features")
            if self.manifest["dataset"] != dataset:
                raise ValueError(f"Cache dataset is {self.manifest['dataset']}, expected {dataset}")
            self.encoder = encoder_settings(self.manifest["encoder"])
            if self.encoder is None:
                raise ValueError("Missing encoder feature definition in cache manifest")
            self.feature_dim = self.encoder["feature_dim"]
            self.split_rows = {split: self.manifest["splits"][split]["rows"] for split in SPLITS}
            if self.manifest["version"] == 2:
                self.cache_id = self.manifest.get("cache_id")
                if not isinstance(self.cache_id, str) or not self.cache_id:
                    raise ValueError("Missing feature cache ID")
        else:
            self.encoder = None
            self.feature_dim = LEGACY_FEATURE_DIM
            self.split_rows = dict(zip(SPLITS, (
                count_rows(self.directory / f"{name}_label.csv") for name in ("train", "eval", "test")
            )))
            total = sum(self.split_rows.values())
            actual = count_rows(self.directory / "eval.csv")
            if min(self.split_rows.values()) < 1 or actual != total:
                raise ValueError(
                    f"Legacy eval.csv must contain train + test + eval ({total} rows, found {actual})"
                )

    def load(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        expected = self.split_rows[split]
        if self.manifest is not None:
            feature_path = self.directory / f"{split}.csv"
            labels, ids = self.load_split_labels(split)
            has_ids = self.manifest["version"] == 2
            features, feature_ids = read_features(
                feature_path, with_ids=has_ids, feature_dim=self.feature_dim,
            )
            if has_ids and feature_ids != ids:
                raise ValueError(f"Feature and label sample IDs do not align in split {split}")
        else:
            legacy_name = "eval" if split == "validation" else split
            start = {
                "train": 0, "test": self.split_rows["train"],
                "validation": self.split_rows["train"] + self.split_rows["test"],
            }[split]
            labels, ids = read_labels(self.directory / f"{legacy_name}_label.csv", with_ids=False)
            features, _ = read_features(
                self.directory / "eval.csv", start=start, count=expected, with_ids=False,
                feature_dim=self.feature_dim,
            )
        if len(features) != expected or len(labels) != expected:
            raise ValueError(f"Feature/label row count mismatch in split {split}")
        return FeatureSplit(features, labels, ids)

    def load_split_labels(self, split):
        path = self.directory / f"{split}_label.csv"
        labels, ids = read_labels(path, with_ids=True)
        if len(labels) != self.split_rows[split]:
            raise ValueError(f"Label row count mismatch in split {split}")
        return labels, ids
