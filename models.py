"""Classifier heads for one cached author representation at a time."""

from torch import nn
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


TORCH_CLASSIFIERS = {
    "linear": (LinearClassifier, {}),
    "mlp": (MLPClassifier, {"hidden_size": 256}),
    "gru": (GRUModel, {"hidden_dim": 512, "n_layers": 2, "drop_prob": 0.2}),
    "rnn": (RNNModel, {"hidden_dim": 512, "n_layers": 2, "drop_prob": 0.2}),
    "bilstm": (LSTMClassifier, {"hidden_size": 512, "num_layers": 1, "bidirectional": True}),
    "cnn": (CNNClassifier, {}),
}

SKLEARN_CLASSIFIERS = ("logistic", "svm-linear", "svm-rbf", "random-forest")
CLASSIFIERS = (*TORCH_CLASSIFIERS, *SKLEARN_CLASSIFIERS)


def model_config(name, feature_dim, num_classes):
    input_name, output_name = ("input_dim", "output_dim") if name in ("gru", "rnn") else (
        "input_size", "output_size"
    )
    return {input_name: feature_dim, output_name: num_classes, **TORCH_CLASSIFIERS[name][1]}


def build_classifier(name, config):
    return TORCH_CLASSIFIERS[name][0](**config)


def sklearn_candidates(name, seed):
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
