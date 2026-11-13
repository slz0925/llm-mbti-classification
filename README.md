# LLM + Small Models for MBTI Personality Detection

This project presents a simple, efficient approach to author-level MBTI prediction by combining pretrained large language models with lightweight classifiers. It caches one representation per author so that different classifiers can be trained on the same data splits without rerunning the encoder. In our experiments, Llama-3-8B paired with several lightweight classifiers comfortably outperformed pre-2025 state-of-the-art baselines, including [D-DGCN](https://doi.org/10.1609/aaai.v37i11.26627) and [TAE](https://doi.org/10.1609/aaai.v38i16.29782), on most metrics and achieved the highest average macro-F1 among the models compared.

## Architecture

```mermaid
flowchart LR
    A["Author text"] --> B["Selected LLM"]
    B --> C["Final-layer feature extraction"]
    C --> D["Cached representations"]
    D --> E["Selected classifier"]
    E --> F["MBTI dichotomy or 16-way type"]
```

The frozen encoder right-pads or truncates to `--max-tokens` (**2,048** by default) and stores one float32 vector per author. The vector width comes from the model's hidden size. The default `--pooling last-position` takes `hidden_states[-1][:, -1, :]` and uses EOS padding when available. For shorter inputs this selects the final padded position. Choose `--pooling last-token` to select the final unmasked token, using the tokenizer's pad token when available or EOS otherwise.

## Supported classifiers and MBTI tasks

| `--classifier` | Architecture |
| --- | --- |
| `linear` | A single affine layer over the cached vector |
| `mlp` | A 256-unit hidden layer with ReLU and a linear output |
| `gru` | Two GRU layers with hidden width 512, dropout 0.2 and a linear output layer |
| `rnn` | Two vanilla RNN layers with hidden width 512, dropout 0.2 and a linear output layer |
| `bilstm` | One bidirectional LSTM layer with hidden width 512 per direction and a linear output layer |
| `cnn` | A 32-channel 1D convolution (kernel width 3), ReLU, adaptive average pooling and a linear output |
| `logistic` | Logistic regression with training-set standardisation |
| `svm-linear` | Linear SVM with training-set standardisation |
| `svm-rbf` | RBF-kernel SVM with training-set standardisation |
| `random-forest` | Random forest with 100 trees |
| `xgboost` | Gradient-boosted trees fitted directly on cached features |

The recurrent classifiers receive `[batch, 1, hidden_size]` and process one time step per author. Each vector represents the entire author input.

The CNN takes the same input shape and convolves along the coordinates of the hidden vector. It pools into up to four bins before classification.

| `--task` | Classes in checkpoint order |
| --- | --- |
| `ie` | I (Introversion), E (Extraversion) |
| `ns` | N (Intuition), S (Sensing) |
| `tf` | T (Thinking), F (Feeling) |
| `pj` | P (Perceiving), J (Judging) |
| `type` | 16 types in the order defined by `MBTI_TYPES` in `data.py` |

Each run trains a separate classifier for one target. The 16-way task learns directly from full-type labels.

## Installation and local resources

Use Python 3.10 or later and an isolated virtual environment. Install PyTorch 2.6 or later for your Python version and CPU or CUDA setup using the [PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the project dependencies.

```bash
python -m pip install -r requirements.txt
```

Install the encoder dependencies for feature extraction or raw-text prediction. They require Transformers 4.56 or later.

```bash
python -m pip install -r requirements-encoder.txt
```

Choose a decoder-only causal language model that accepts text token IDs and returns token-level hidden states. Use `--model-dir` for an existing local Transformers directory containing the weights, configuration and tokenizer, or `--model-id` for a Hugging Face Hub model. Transformers downloads the Hub files on first use and reuses them from the local [Hugging Face cache](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hf_hub_cache). To relocate the cache, set `HF_HOME` or `HF_HUB_CACHE` to a location outside this repository. For gated models, obtain access and sign in with `hf auth login`. Inference runs locally with either source.

Install additional packages if your model requires them, such as `sentencepiece` and `protobuf` for tokenizers that need SentencePiece conversion. Custom model code is disabled by default. Add `--trust-remote-code` only after reviewing the code you intend to run.

Extraction defaults to BF16 and automatic device placement. Use `--encoder-device cuda:0` to select a GPU explicitly or `--encoder-dtype float32` when needed. Size memory for the chosen model and its activations. Classifier-only commands use the core requirements. PyTorch classifiers support CPU and CUDA, and scikit-learn classifiers run on CPU.

## Data preparation

Obtain the datasets separately and keep all source data, prepared splits and feature caches local. Choose the applicable preparation command from the repository root.

The examples use `data/` for source datasets, `prepared/` for processed splits, `features/` for feature caches, `models/` for local encoders, `checkpoints/` for trained classifiers and `results/` for evaluation output. Pass your own paths as needed. Output commands create their parent directories.

**Kaggle.** Download the [Myers-Briggs Personality Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) and use its `type` and `posts` columns. Each row represents one author and uses `|||` to separate posts.

```bash
python prepare_data.py --dataset kaggle --csv data/kaggle/mbti_1.csv --output prepared/kaggle
```

**PANDORA.** Request access via the [official PANDORA page](https://psy.takelab.fer.hr/datasets/all/pandora/). Provide the author profiles with `author` and `mbti`, and the comments with `author` and `body`.

```bash
python prepare_data.py --dataset pandora --profiles data/pandora/author_profiles.csv --comments data/pandora/all_comments_since_2015.csv --output prepared/pandora
```

Preparation removes URLs and standalone MBTI type labels, groups posts by author and joins them in order with newlines. For PANDORA, it keeps comments with at least six whitespace-separated words after cleaning and takes the last 2,000 qualifying comments per author in source order. It creates deterministic, approximately 60/20/20 train, validation and test partitions using seed 0 by default. The Kaggle split uses the dataset's one-row-per-author convention. Each prepared split is a JSON Lines file with a sample ID, MBTI type and cleaned text.

## Feature extraction

Specify exactly one model source. The examples use a local Meta-Llama-3-8B-Instruct directory and a Hub-hosted Qwen model.

```bash
python extract_features.py --prepared prepared/kaggle --model-dir models/Meta-Llama-3-8B-Instruct --output features/kaggle/llama3-last-position
python extract_features.py --prepared prepared/kaggle --model-id Qwen/Qwen3-0.6B --pooling last-token --max-tokens 8192 --output features/kaggle/qwen3-last-token-8192
```

For PANDORA, use the corresponding prepared and output directories. Each split produces a feature CSV with a sample ID and columns `0` through `hidden_size - 1` (4,096 features for Llama-3-8B), paired with a label CSV in the same sample order. The manifest records the dataset, split sizes, cache ID, model, feature width, input length, dtype and pooling. For Hub models it also records the resolved commit ID. Choose a length within the model's context and use a separate cache for each encoder, input length and pooling combination. Train and evaluate with the same cache, and reuse it across classifiers or tasks.

For a 4,096-column cumulative cache, pass the directory containing `eval.csv` and `train_label.csv`, `test_label.csv` and `eval_label.csv`. The loader aligns labels with the train, test and validation blocks of `eval.csv` by row order.

## Training and evaluation

```bash
python train.py --dataset kaggle --features features/kaggle/llama3-last-position --classifier linear --task ie --epochs 30 --checkpoint checkpoints/kaggle-linear-ie.pt
python evaluate.py --features features/kaggle/llama3-last-position --checkpoint checkpoints/kaggle-linear-ie.pt --output results/kaggle-linear-ie-test.json
```

Train a CNN for the 16-way task on the same cached features.

```bash
python train.py --dataset kaggle --features features/kaggle/llama3-last-position --classifier cnn --task type --epochs 30 --checkpoint checkpoints/kaggle-cnn-type.pt
```

For a classical model on the Qwen cache, train and evaluate a direct 16-way logistic regression classifier.

```bash
python train.py --dataset kaggle --features features/kaggle/qwen3-last-token-8192 --classifier logistic --task type --checkpoint checkpoints/kaggle-logistic-type.joblib
python evaluate.py --features features/kaggle/qwen3-last-token-8192 --checkpoint checkpoints/kaggle-logistic-type.joblib --trust-checkpoint --output results/kaggle-logistic-type-test.json
```

Install XGBoost only if you want to train it or load one of its checkpoints. It uses the same cached features as the other classifiers.

```bash
python -m pip install xgboost
python train.py --dataset kaggle --features features/kaggle/qwen3-last-token-8192 --classifier xgboost --task type --checkpoint checkpoints/kaggle-xgboost-type.joblib
```

PyTorch classifiers use Adam and cross-entropy with full-batch updates by default. Set `--lr`, `--epochs` and `--batch-size` as needed. Validation macro-F1 selects the epoch, and `.pt` stores a state dictionary. Logistic regression and the SVMs compare `C=0.1` and `C=1.0`. Random forest compares unlimited depth and depth 12. XGBoost compares depths 3 and 6 with 100 boosting rounds. Each candidate is fitted on training rows only, including any standardisation, and validation macro-F1 selects the fitted estimator. RBF SVM can take substantially longer on large splits. Load `.joblib` checkpoints only from trusted sources with `--trust-checkpoint`, using the same library versions as for training.

Run the separate test command after model selection. It reports accuracy, macro-F1, macro precision and macro recall, plus a majority-class baseline fitted to the training labels. Macro metrics give every class equal weight and assign zero to undefined per-class scores.

## Inference

Predict from a compatible feature CSV to get one JSON record per input row. The command accepts feature CSVs with or without a sample ID column. For a standalone CSV, use features from the checkpoint's encoder and pooling settings.

```bash
python predict.py --checkpoint checkpoints/kaggle-linear-ie.pt --features features/kaggle/llama3-last-position/test.csv
```

For an author whose text has not been encoded, provide one or more posts and select the encoder used to build the training cache.

```bash
python predict.py --checkpoint checkpoints/kaggle-linear-ie.pt --model-dir models/Meta-Llama-3-8B-Instruct --text "I spent the afternoon sketching plans for a small garden."
python predict.py --checkpoint checkpoints/kaggle-logistic-type.joblib --trust-checkpoint --model-id Qwen/Qwen3-0.6B --text "I enjoy exploring new ideas."
```

Repeat `--text` for multiple posts in order, or use `--text-file` for a UTF-8 file containing one post. With `--model-id`, prediction uses the Hub revision recorded in the checkpoint when available. For a local model or a checkpoint without a recorded revision, supply the same weights and tokenizer used for extraction. Use the same Transformers version if a newer release changes your tokenizer. Prediction uses the checkpoint's dataset, input length, pooling and dtype. An explicit `--max-tokens` must match the checkpoint. For a 4,096-column cumulative cache without encoder metadata, use the corresponding Meta-Llama-3-8B-Instruct resources and the 2,048-token definition. Add `--trust-checkpoint` for a trusted `.joblib` classifier.

## Data access and citation

Cite the dataset and encoder used in your experiments. [Kaggle MBTI dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) · [Gjurković et al. (2021), PANDORA Talks](https://aclanthology.org/2021.socialnlp-1.12/) · [Meta-Llama-3-8B-Instruct model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
