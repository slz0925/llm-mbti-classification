"""Train a classifier on a local feature cache and select it on validation macro-F1."""

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

from data import DATASETS, TASK_COLUMN, TASKS, FeatureCache, atomic_file, encoder_settings
from models import (
    CLASSIFIERS, SKLEARN_CLASSIFIERS, XGBOOST_INSTALL, XGBoostClassifier,
    build_classifier, model_config, sklearn_candidates,
)


def resolve_device(name):
    return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name)


def metrics(targets, predictions, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=range(num_classes), average="macro", zero_division=0,
    )
    return {
        "macro_f1": float(f1), "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision), "macro_recall": float(recall),
        "samples": len(targets),
    }


@torch.inference_mode()
def predict_classes(model, features, device, batch_size, *, feature_dim):
    if features.ndim != 2 or features.shape[1] != feature_dim:
        raise ValueError(f"Classifier requires {feature_dim} feature columns")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if not isinstance(model, nn.Module):
        return model.predict(features).tolist()
    device = resolve_device(device)
    model.eval()
    predicted = []
    for start in range(0, len(features), batch_size):
        inputs = torch.as_tensor(features[start:start + batch_size], device=device).unsqueeze(1)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise ValueError("Classifier returned non-finite logits")
        predicted.extend(logits.argmax(dim=1).tolist())
    return predicted


def train_classifier(cache, classifier, task, checkpoint, device, *, epochs, lr, batch_size, seed):
    classical = classifier in SKLEARN_CLASSIFIERS
    if classifier != "xgboost":
        device = resolve_device(device)
    if (Path(checkpoint).suffix.lower() == ".joblib") != classical:
        raise ValueError("Use .joblib for scikit-learn checkpoints and .pt for PyTorch checkpoints")
    train, validation = cache.load("train"), cache.load("validation")
    if set(train.ids).intersection(validation.ids):
        raise ValueError("Train and validation contain overlapping author IDs")
    column = TASK_COLUMN[task]
    train_labels, validation_labels = train.labels[:, column], validation.labels[:, column]
    num_classes = len(TASKS[task])
    width = train.features.shape[1]
    state = {
        "version": 2, "classifier": classifier, "dataset": cache.dataset, "task": task,
        "classes": list(TASKS[task]), "feature_dim": width, "encoder": cache.encoder,
        "cache_id": cache.cache_id, "split_rows": cache.split_rows,
    }
    training = {
        "seed": seed, "majority_class": int(np.bincount(train_labels, minlength=num_classes).argmax()),
    }
    if classical:
        if np.unique(train_labels).size < 2:
            raise ValueError(f"{classifier} requires at least two classes in the training split")
        best, selected = None, None
        for estimator in sklearn_candidates(classifier, seed, device=device):
            estimator.fit(train.features, train_labels)
            predictions = predict_classes(estimator, validation.features, device, 1, feature_dim=width)
            score = metrics(validation_labels, predictions, num_classes)
            fitted = estimator[-1] if hasattr(estimator, "steps") else estimator
            params = fitted.get_params(deep=False)
            print(json.dumps({"parameters": params, "validation": score}), flush=True)
            if best is None or score["macro_f1"] > best["validation"]["macro_f1"]:
                best, selected = {"validation": score, "parameters": params}, estimator
        with atomic_file(checkpoint) as temporary:
            joblib.dump({
                **state, "model_config": best["parameters"], "training": {**training, **best},
                "estimator": selected, "sklearn_version": sklearn.__version__,
            }, temporary, compress=3)
        return best

    if epochs < 1 or not math.isfinite(lr) or lr <= 0 or batch_size < 0:
        raise ValueError("Use positive epochs and learning rate, and a non-negative batch size")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    config = model_config(classifier, width, num_classes)
    model = build_classifier(classifier, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    step_size = batch_size or len(train.features)
    order = torch.Generator().manual_seed(seed)

    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(train.features), generator=order).numpy() if batch_size else slice(None)
        for start in range(0, len(train.features), step_size):
            batch = indices[start:start + step_size] if batch_size else indices
            inputs = torch.as_tensor(train.features[batch], device=device).unsqueeze(1)
            targets = torch.as_tensor(train_labels[batch], device=device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Training produced a non-finite loss")
            loss.backward()
            optimizer.step()

        predictions = predict_classes(model, validation.features, device, step_size, feature_dim=width)
        score = metrics(validation_labels, predictions, num_classes)
        print(json.dumps({"epoch": epoch, "validation": score}), flush=True)
        if best is None or score["macro_f1"] > best["validation"]["macro_f1"]:
            best = {"epoch": epoch, "validation": score}
            payload = {
                **state, "model_config": config,
                "training": {**training, "learning_rate": lr, "batch_size": batch_size,
                             "requested_epochs": epochs, **best},
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }
            with atomic_file(checkpoint) as temporary:
                torch.save(payload, temporary)
    return best


def load_checkpoint(path, device, *, trust_checkpoint=False):
    classical = Path(path).suffix.lower() == ".joblib"
    if classical and not trust_checkpoint:
        raise ValueError("Only load trusted .joblib checkpoints. Pass --trust-checkpoint for your own files")
    try:
        state = joblib.load(path) if classical else torch.load(path, map_location="cpu", weights_only=True)
    except ModuleNotFoundError as exc:
        if not classical or exc.name != "xgboost":
            raise
        raise ValueError(XGBOOST_INSTALL) from exc
    if (not isinstance(state, dict) or state.get("version") not in (1, 2)
            or state.get("task") not in TASKS or state.get("dataset") not in DATASETS
            or state.get("classifier") not in CLASSIFIERS
            or (state["classifier"] in SKLEARN_CLASSIFIERS) != classical
            or type(state.get("feature_dim")) is not int or state["feature_dim"] < 1):
        raise ValueError("Unsupported classifier checkpoint")
    if state["classes"] != list(TASKS[state["task"]]):
        raise ValueError("Checkpoint class mapping does not match the task")
    config = state["model_config"]
    if state["classifier"] != "xgboost":
        device = resolve_device(device)
    width = state["feature_dim"]
    encoder = encoder_settings(state["encoder"])
    if encoder is not None and encoder["feature_dim"] != width:
        raise ValueError("Checkpoint encoder width does not match classifier input width")
    state["encoder"] = encoder
    if classical:
        model = state["estimator"]
        if isinstance(model, XGBoostClassifier) != (state["classifier"] == "xgboost"):
            raise ValueError("Checkpoint estimator does not match its classifier")
        if state["classifier"] == "xgboost":
            model.validate_label_mapping()
            config = {key: value for key, value in config.items() if key != "device"}
        fitted = model[-1] if hasattr(model, "steps") else model
        if (getattr(model, "n_features_in_", None) != width or fitted.get_params(deep=False) != config
                or not set(model.classes_).issubset(range(len(state["classes"])))):
            raise ValueError("Checkpoint estimator does not match its feature width, configuration or task")
        if state["classifier"] == "xgboost":
            model.set_device(device)
        return model, state
    expected = model_config(state["classifier"], width, len(state["classes"]))
    if config != expected:
        raise ValueError("Checkpoint architecture does not match the classifier and task")
    model = build_classifier(state["classifier"], config)
    model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--features", required=True, type=Path, help="Local feature cache directory")
    parser.add_argument("--classifier", choices=CLASSIFIERS, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100, help="PyTorch training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="PyTorch learning rate")
    parser.add_argument("--batch-size", type=int, default=0, help="PyTorch batch size (0 for full batch)")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="auto", help="auto, cpu or a CUDA device such as cuda:0")
    args = parser.parse_args()
    try:
        cache = FeatureCache(args.features, args.dataset)
        best = train_classifier(
            cache, args.classifier, args.task, args.checkpoint, args.device,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"selected": best, "checkpoint": str(args.checkpoint)}))


if __name__ == "__main__":
    main()
