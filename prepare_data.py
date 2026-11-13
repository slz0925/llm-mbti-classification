"""Prepare local Kaggle or PANDORA text for author-level feature extraction."""

import argparse
import csv
import json
import random
import re
import sys
from collections import deque
from pathlib import Path

from data import DATASETS, MBTI_TYPES, SPLITS, new_directory, write_json


URL = re.compile(r"http\S+", flags=re.MULTILINE)
TYPE_MENTION = re.compile(
    r"\b(?:" + "|".join((*MBTI_TYPES, "TYPE_MENTION", "COGFUNC_MENTION")) + r")\b",
    re.IGNORECASE,
)
PANDORA_POST_LIMIT = 2000
PANDORA_MIN_WORDS = 6


def prepare_text(dataset, posts):
    """Apply the dataset policy to an ordered collection of one author's posts."""
    cleaned = []
    for post in posts:
        post = TYPE_MENTION.sub("", URL.sub("", post))
        if not post.strip():
            continue
        if dataset == "pandora" and len(post.split()) < PANDORA_MIN_WORDS:
            continue
        cleaned.append(post)
    if dataset == "pandora":
        cleaned = cleaned[-PANDORA_POST_LIMIT:]
    return "\n".join(cleaned)


def read_csv(path, columns):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected CSV columns {', '.join(columns)} in {path}")
        for row in reader:
            if any(row[column] is None for column in columns):
                raise ValueError(f"Incomplete CSV row {reader.line_num} in {path}")
            yield row


def read_kaggle(path):
    samples = []
    for number, row in enumerate(read_csv(path, ("type", "posts")), start=1):
        label = row["type"].strip().upper()
        if label not in MBTI_TYPES:
            raise ValueError(f"Invalid MBTI type at Kaggle record {number}")
        text = prepare_text("kaggle", row["posts"].split("|||"))
        if text:
            samples.append({"id": str(number), "mbti": label, "text": text})
    return samples


def read_pandora(profiles, comments):
    labels = {}
    for row in read_csv(profiles, ("author", "mbti")):
        author, label = row["author"].strip(), row["mbti"].strip().upper()
        if not author or label not in MBTI_TYPES:
            continue
        if author in labels and labels[author] != label:
            raise ValueError("Conflicting MBTI labels for an author in PANDORA profiles")
        labels[author] = label

    # Retain only the last 2,000 usable comments per author in source order.
    authors = {}
    for row in read_csv(comments, ("author", "body")):
        author = row["author"].strip()
        if author in labels:
            if author not in authors:
                authors[author] = deque(maxlen=PANDORA_POST_LIMIT)
            text = prepare_text("pandora", (row["body"],))
            if text:
                authors[author].append(text)
    return [
        {"id": str(index), "mbti": labels[author], "text": "\n".join(posts)}
        for index, (author, posts) in enumerate(authors.items(), start=1) if posts
    ]


def split_authors(samples, seed):
    shuffled = list(samples)
    if len(shuffled) < 3 or len({row["id"] for row in shuffled}) != len(shuffled):
        raise ValueError("At least three distinct author records are required")
    random.Random(seed).shuffle(shuffled)
    train_end = int(0.6 * len(shuffled))
    validation_end = int(0.8 * len(shuffled))
    return dict(zip(SPLITS, (
        shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:],
    )))


def prepare_dataset(dataset, output, seed, *, csv_path=None, profiles=None, comments=None):
    # A Kaggle row can contain the full posting history for an author.
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    if dataset == "kaggle":
        samples = read_kaggle(csv_path)
    else:
        samples = read_pandora(profiles, comments)
    splits = split_authors(samples, seed)
    manifest = {
        "format": "mbti-prepared", "version": 1, "dataset": dataset,
        "split_seed": seed,
        "splits": {},
    }
    with new_directory(output) as destination:
        for split, rows in splits.items():
            path = destination / f"{split}.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["splits"][split] = {"rows": len(rows)}
        write_json(destination / "manifest.json", manifest)
    return {split: len(rows) for split, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--csv", type=Path, help="Kaggle CSV with type and posts columns")
    parser.add_argument("--profiles", type=Path, help="PANDORA author_profiles.csv")
    parser.add_argument("--comments", type=Path, help="PANDORA all_comments_since_2015.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.dataset == "kaggle" and (not args.csv or args.profiles or args.comments):
        parser.error("Kaggle requires --csv")
    if args.dataset == "pandora" and (args.csv or not args.profiles or not args.comments):
        parser.error("PANDORA requires --profiles and --comments")
    try:
        counts = prepare_dataset(
            args.dataset, args.output, args.seed,
            csv_path=args.csv, profiles=args.profiles, comments=args.comments,
        )
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
