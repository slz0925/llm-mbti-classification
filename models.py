"""Classifier heads for one cached author representation at a time."""

import json
import math
import warnings

import numpy as np
from torch import device as torch_device, nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        # PyTorch supplies zero initial states on the input device.
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out)[:, -1, :])


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(self.dropout(out)[:, -1, :])


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x[:, -1, :])


class MLPClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                    nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.layers(x[:, -1, :])


class CNNClassifier(nn.Module):
    """Convolve along the coordinates of a single author vector."""

    def __init__(self, input_size, output_size):
        super().__init__()
        pooled_width = min(input_size, 4)
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pooled_width),
            nn.Flatten(start_dim=1),
        )
        self.fc = nn.Linear(32 * pooled_width, output_size)

    def forward(self, x):
        return self.fc(self.features(x))


XGBOOST_INSTALL = "Install XGBoost with: python -m pip install xgboost"


def xgboost_booster_device(booster):
    return json.loads(booster.save_config())["learner"]["generic_param"].get("device")


def xgboost_device(requested):
    automatic = requested == "auto"
    try:
        device = torch_device("cuda" if automatic else requested)
    except RuntimeError as exc:
        raise ValueError("XGBoost --device must be auto, cpu or cuda:N") from exc
    if device.type == "cpu":
        return "cpu"
    if device.type != "cuda":
        raise ValueError("XGBoost --device must be auto, cpu or cuda:N")
    import xgboost

    target = f"cuda:{device.index if device.index is not None else 0}"
    build_info = getattr(xgboost, "build_info", None)
    if build_info is None or not build_info().get("USE_CUDA"):
        if automatic:
            return "cpu"
        raise ValueError("XGBoost CUDA requires a CUDA-enabled XGBoost build with device support (2.0 or later)")
    try:
        # XGBoost can accept a CUDA request while resolving it to CPU.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            booster = xgboost.Booster(params={"device": target, "num_feature": 1})
            actual = xgboost_booster_device(booster)
    except xgboost.core.XGBoostError as exc:
        if automatic:
            return "cpu"
        raise ValueError(f"XGBoost cannot use {target}: {str(exc).splitlines()[0]}") from exc
    if actual == target:
        return target
    if automatic:
        return "cpu"
    raise ValueError(f"XGBoost cannot use {target}: resolved device is {actual or 'unavailable'}")


class XGBoostClassifier:
    """Map the observed task labels to the contiguous IDs required by XGBoost."""

    def __init__(self, max_depth, seed):
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError as exc:
            if exc.name != "xgboost":
                raise
            raise ValueError(XGBOOST_INSTALL) from exc
        self.estimator = XGBClassifier(
            n_estimators=100, max_depth=max_depth, learning_rate=0.1, tree_method="hist", random_state=seed,
        )

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    def fit(self, features, labels):
        self.classes_, encoded = np.unique(labels, return_inverse=True)
        self.estimator.fit(features, encoded)
        self.check_device()
        return self

    def predict(self, features):
        encoded = self.estimator.predict(features)
        self.check_device()
        return self.classes_[encoded]

    def set_device(self, requested):
        target = xgboost_device(requested)
        if target != "cpu" or hasattr(self.estimator, "device"):
            self.estimator.set_params(device=target)

    def check_device(self):
        requested = getattr(self.estimator, "device", None)
        if requested and requested.startswith("cuda"):
            actual = xgboost_booster_device(self.estimator.get_booster())
            if actual != requested:
                raise ValueError(f"XGBoost cannot use {requested}: resolved device is {actual or 'unavailable'}")

    def get_params(self, deep=False):
        params = self.estimator.get_params(deep=deep)
        params.pop("device", None)
        missing = params.get("missing")
        if isinstance(missing, float) and math.isnan(missing):
            # Compare checkpoint metadata reliably after deserialisation.
            params["missing"] = "NaN"
        return params

    def validate_label_mapping(self):
        labels = getattr(self, "classes_", None)
        if (not isinstance(labels, np.ndarray) or labels.ndim != 1 or labels.dtype.kind not in "iu"
                or len(labels) < 2 or not np.all(labels[1:] > labels[:-1])
                or not np.array_equal(self.estimator.classes_, np.arange(len(labels)))):
            raise ValueError("XGBoost checkpoint has an invalid training label mapping")


TORCH_CLASSIFIERS = {
    "linear": (LinearClassifier, {}),
    "mlp": (MLPClassifier, {"hidden_size": 256}),
    "gru": (GRUModel, {"hidden_dim": 512, "n_layers": 2, "drop_prob": 0.2}),
    "rnn": (RNNModel, {"hidden_dim": 512, "n_layers": 2, "drop_prob": 0.2}),
    "bilstm": (LSTMClassifier, {"hidden_size": 512, "num_layers": 1, "bidirectional": True}),
    "cnn": (CNNClassifier, {}),
}

SKLEARN_CLASSIFIERS = ("logistic", "svm-linear", "svm-rbf", "random-forest", "xgboost")
CLASSIFIERS = (*TORCH_CLASSIFIERS, *SKLEARN_CLASSIFIERS)


def model_config(name, feature_dim, num_classes):
    input_name, output_name = ("input_dim", "output_dim") if name in ("gru", "rnn") else (
        "input_size", "output_size"
    )
    return {input_name: feature_dim, output_name: num_classes, **TORCH_CLASSIFIERS[name][1]}


def build_classifier(name, config):
    return TORCH_CLASSIFIERS[name][0](**config)


def sklearn_candidates(name, seed, *, device="auto"):
    if name == "xgboost":
        for depth in (3, 6):
            candidate = XGBoostClassifier(depth, seed)
            candidate.set_device(device)
            yield candidate
        return
    if name == "random-forest":
        for depth in (None, 12):
            yield RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=seed)
        return
    for c in (0.1, 1.0):
        if name == "logistic":
            estimator = LogisticRegression(C=c, max_iter=1000, random_state=seed)
        elif name == "svm-linear":
            estimator = LinearSVC(C=c, max_iter=5000, random_state=seed)
        else:
            estimator = SVC(C=c, kernel="rbf", gamma="scale", cache_size=200)
        yield make_pipeline(StandardScaler(), estimator)
