# BioLM 2.0 Framework

A modular PyTorch framework for training language models on biological sequences (RNA/protein). Features a **plugin architecture** where model implementations are separate packages developed and versioned independently.

---

## Table of Contents

- [Installation](#installation)
- [Adding Plugins](#adding-plugins)
- [Data Format](#data-format)
- [Modes Overview](#modes-overview)
- [Usage](#usage)
- [Available Plugins](#available-plugins)
- [Configuration Management](#configuration-management)
- [Output Directory Structure](#output-directory-structure)
- [MLflow Tracking](#mlflow-tracking)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🚀 Installation

**Requirements:**

- Python 3.10+
- Poetry ([install guide](https://python-poetry.org/docs/#installation))

**Framework Installation (no plugins yet):**

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
git checkout biolm-2.0
./install.sh
```

BioLM 2.0 development happens on the `biolm-2.0` branch—`main` is the legacy line, so always install from `biolm-2.0` for the latest features and fixed plugin hooks.

`install.sh` installs only the BioLM framework. Plugins are installed separately (see below).

## 🔌 Adding Plugins

- **Choose one path (most users only need Path A):**

### Path A — Run an existing plugin (recommended)

  ```bash
  # inside the biolm_utils repo
  poetry run biolm install-plugin <git-url>
  poetry run biolm list-plugins
  ```

  For Saluki 2.0 specifically, install from the active branch:

  ```bash
  poetry run biolm install-plugin "https://github.com/dieterich-lab/rna_saluki_cnn.git?ref=saluki-2.0"
  ```

  What `install-plugin` does:
  - Clones the plugin repo into `./plugins/<name>`.
  - Installs it into the active Poetry environment (editable install) so BioLM can load it.

  Use this path when you want to run a plugin and do not plan to modify plugin source code.

  **Plugin discovery:** As long as the plugin is installed in the same Poetry environment (via `install-plugin` or `develop-plugin`), BioLM automatically discovers the entry point—no extra registration steps are needed.

### Path B — Develop a plugin locally

If you are editing plugin code, first clone the plugin repository locally, then point BioLM to that local path.

  ```bash
  # inside the biolm_utils repo
  poetry install --no-interaction
  poetry run biolm develop-plugin /path/to/your/plugin
  ```

  Use `poetry install --no-interaction` only when the framework environment is not set up yet (if you already ran `./install.sh`, you can run only `biolm develop-plugin`).

  This keeps `pyproject.toml` unchanged while wiring editable installs through the CLI. Edits in your plugin repo are picked up immediately.

  Remove a plugin later via `poetry run biolm remove-plugin <plugin-name>` (recommended).

If you previously used `install-plugin` and no longer want the cloned copies, you can safely remove the `./plugins` directory; the CLI will recreate it on demand for future user installs.

---

## 📊 Data Format

Input files must specify the delimiter using the `data_source.columnsep` configuration. By default, the delimiter is set to tab (`\t`). Example (tab-separated columns, raw sequence text):

```tsv
ID	Label	Sequence
seq_001	1.5	AUGCUAGCUAGC
seq_002	2.3	AUGGCUAUGGCU
```

---

## ⚡ Modes Overview

| Mode         | Description                                                                 | Typical Use/Plugin         |
|--------------|-----------------------------------------------------------------------------|---------------------------|
| tokenize     | Build vocabulary/tokenizer from data.                                       | All models                |
| pre-train    | (Optional) Pre-train language model on unlabeled data.                      | Required for LMs          |
| fine-tune    | Train model on labeled data for your task.                                  | All models                |
| predict      | Run inference/prediction on new data.                                       | All models                |
| interpret    | Feature importance/interpretation (e.g., saliency, attention, etc.).        | All models                |

**Notes:**

- Language models (e.g., XLNet) require pre-training before fine-tuning.
- CNN-based models (e.g., Saluki) do **not** require pre-training.

### Mode Quickstart

Below are the canonical commands, vital configuration knobs, and outputs for each mode. Reference paths assume you keep experiment-specific overrides under `./my_experiment` and set `outputpath` inside that config.

**Tokenize**

```bash
poetry run biolm mode=tokenize plugin=<plugin_name> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

- Key config values: `data_source.filepath`, `tokenization.encoding`, `tokenization.vocabsize`.
- Output: tokenizer artifacts in `${outputpath}/tokenize` (e.g., merges.txt, vocab.json).

**Pre-train**

```bash
poetry run biolm mode=pre-train plugin=<plugin_name> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

- Requires a plugin whose config sets `task: pre-train` (see `mode/pre-train.yaml`).
- Important options: `training.nepochs`, `training.batchsize`, `training.scaling`, `settings.mlflow.enabled`.
- Output: pretrained weights and logs in `${outputpath}/pre-train`.

**Fine-tune**

```bash
poetry run biolm mode=fine-tune plugin=<plugin_name> task=<classification|regression> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

- Make sure `plugin` points to the installed model package and `task` matches the plugin expectation (classification/regression).
- Main toggles: `data_source.splitratio`, `training.nepochs`, `training.patience`, `training.gradacc`.
- Output: fine-tuned checkpoints, metrics, and MLflow logs in `${outputpath}/fine-tune`.

**Predict**

```bash
poetry run biolm mode=predict plugin=<plugin_name> task=<classification|regression> data_source.filepath=/path/to/data.tsv inference.pretrainedmodel=/path/to/model.ckpt outputpath=/tmp/biolm_run
```

- Ensure `inference.pretrainedmodel` is set to the checkpoint produced by fine-tuning or pre-training.
- Optional overrides: `inference.looscores.handletokens` (defaults to `mask` here), `debugging.dev` for quick dry-runs.
- Output: `${outputpath}/predict/test_predictions.csv` (IDs plus plugin-specific scores/probabilities) and logs in `${outputpath}/predict/logs/`.

**Interpret**

```bash
poetry run biolm mode=interpret plugin=<plugin_name> task=<classification|regression> data_source.filepath=/path/to/data.tsv inference.pretrainedmodel=/path/to/model.ckpt outputpath=/tmp/biolm_run
```

- Core options under `inference.looscores`:
  - `handletokens`: `mask` (default) or `remove` to control occlusion behaviour.
  - `replacementdict`: dictionary limiting replacements per token; leave `null` for full masking.
  - `replacespecifier`: boolean to include sequence specifier fields in replacements.
- Other useful flags: `debugging.dev` to restrict the number of samples, `training.batchsize` for occlusion batching.
- Output: `${outputpath}/interpret/loo_scores_<handletokens>.csv` and `.pkl` plus logs in `${outputpath}/interpret/logs/`.

---

## 🛠️ Usage

Run any mode with:

```bash
poetry run biolm mode=<tokenize|pre-train|fine-tune|predict|interpret> plugin=<plugin_name> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

Optional equivalent invocation:

```bash
poetry run python -m biolm.runner mode=fine-tune plugin=<plugin_name> task=<classification|regression> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

`--config-path`/`--config-name` are optional. Use them only when you provide a complete custom Hydra config tree.

## 🧭 Execution Flow (at a glance)

1. CLI parses args and Hydra composes configs.
2. `plugin_config` resolves the plugin entry point; plugin config classes are loaded.
3. Data is loaded/prepared (tokenizer built or loaded); datasets are cached under `${outputpath}/{mode}`.
4. Mode dispatcher (`runner`) calls the appropriate trainer/evaluator.
5. Artifacts and logs are written to `${outputpath}/{mode}`; MLflow (if enabled) logs params/metrics/artifacts to `${outputpath}/mlruns`.

## ⚙️ Configuration & Quickstart

BioLM uses Hydra to compose the framework-wide base config ([biolm/conf/config.yaml](biolm/conf/config.yaml#L3-L90)) with
mode-specific overrides ([biolm/conf/mode](biolm/conf/mode)) and any user-provided experiment files.
Organize experiments in dedicated directories so `--config-path`/`--config-name` can find them easily.

### Minimal experiment config

Pick a plugin, output path, and the options that change per run. Here is a minimal `config.yaml` you
can drop into any experiment directory:

```yaml
plugin: <plugin_name>
outputpath: /tmp/biolm_quickstart
task: classification
data_source:
  filepath: examples/data/quickstart_sequences.tsv
  columnsep: "\t"
  idpos: 1
  seqpos: 2
  labelpos: 3
  splitratio: [70, 15, 15]
training:
  nepochs: 3
  batchsize: 4
```

### Hydra composition

The shared base config declares:

```yaml
defaults:
  - mode: ???
  - _self_
```

This means Hydra expects you to resolve a mode file (e.g., the `mode/fine-tune.yaml` bundle) before the CLI can run. You can do this either by adding:

```yaml
defaults:
  - mode: fine-tune
  - _self_
```

inside your experiment config or by passing `mode=fine-tune` on the command line.

Hydra merges the base config, the selected mode, your experiment config, and any runtime overrides
(for example, `training.nepochs=50` or `data_source.filepath=/new/path`). That keeps the common
defaults inside `biolm/conf` untouched while letting you customize only the pieces that change per run.

### Custom experiment directories

Structure each experiment like this:

```
my_experiment/
├── config.yaml
└── mode/
    └── fine-tune.yaml
```

Drop the minimal config above into `config.yaml` and add `mode/fine-tune.yaml` when you need to
override defaults from [biolm/conf/mode/fine-tune.yaml](biolm/conf/mode/fine-tune.yaml#L1-L10)
(different splits, MLflow hooks, debug flags, etc.). Run the CLI with:

```bash
poetry run biolm mode=fine-tune plugin=<plugin_name> task=<classification|regression> data_source.filepath=/path/to/data.tsv outputpath=/tmp/biolm_run
```

If the file does not pin the mode yet, append `mode=fine-tune` to resolve the ??? default.

### Quickstart commands

With a config directory ready, run the modes sequentially as follows (adjust for your plugin if it
does not require pre-training):

**For Saluki (CNN-based, no pre-training needed):**

```bash
# Tokenize data (required for atomic encoding)
poetry run biolm mode=tokenize plugin=saluki data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 outputpath=/tmp/biolm_quickstart
# Fine-tune directly (no pre-training required)
poetry run biolm mode=fine-tune plugin=saluki task=classification data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 outputpath=/tmp/biolm_quickstart
poetry run biolm mode=predict plugin=saluki task=classification data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 inference.pretrainedmodel=/tmp/biolm_quickstart/fine-tune/model.safetensors outputpath=/tmp/biolm_quickstart
poetry run biolm mode=interpret plugin=saluki task=classification data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 inference.pretrainedmodel=/tmp/biolm_quickstart/fine-tune/model.safetensors outputpath=/tmp/biolm_quickstart
```

**For XLNet (transformer-based, requires pre-training):**

```bash
poetry run biolm mode=tokenize plugin=xlnet data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 outputpath=/tmp/biolm_quickstart
poetry run biolm mode=pre-train plugin=xlnet data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 outputpath=/tmp/biolm_quickstart
poetry run biolm mode=fine-tune plugin=xlnet task=classification data_source.filepath=examples/data/quickstart_sequences.tsv data_source.stripheader=true data_source.idpos=1 data_source.seqpos=3 data_source.labelpos=2 outputpath=/tmp/biolm_quickstart
```

Skip the `pre-train` command if your plugin (for example, a CNN) only needs fine-tuning. The
[examples/data/quickstart_sequences.tsv](examples/data/quickstart_sequences.tsv) file includes 100
tab-separated rows (ID, label, sequence) so you can experiment without cloning any plugins.

### Runtime overrides

Pass overrides like `training.batchsize=8`, `data_source.filepath=/new.tsv`, or
`settings.mlflow.enabled=true` after the command to tweak a single value without editing YAML.
Hydra merges these last, so they take precedence over the experiment files and the framework defaults.

---

## 🔌 Available Plugins

| Plugin | Model | Sequences | Pre-training | Use Case |
|--------|-------|-----------|--------------|----------|
| [rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet) | XLNet | RNA/DNA/Protein | Yes | General sequence modeling (pre-train + downstream tasks) |
| [rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn) | CNN | RNA/DNA/Protein | No | Sequence classification/regression without pre-train |

---

## 📂 Output Directory Structure

The framework organizes outputs under the configured `outputpath`:

```plaintext
output/
├── tokenize/
│   ├── merges.txt              # BPE merge rules (if applicable)
│   ├── vocab.json             # Tokenizer vocabulary
│   ├── tokenizer_config.json  # HuggingFace tokenizer configuration
│   └── tokenizer.json         # Serialized tokenizer weights
├── pre-train/
│   ├── checkpoint-XX/         # Checkpoints saved per epoch
│   ├── model.safetensors      # Final model weights
│   ├── config.json            # Model config
│   ├── pre-train_dataset.pkl  # Cached dataset (for reproducibility)
│   ├── logs/<timestamp>.log   # Training logs
│   └── final_model/           # Copy of best checkpoint
├── fine-tune/
│   ├── checkpoint-XX/         # Checkpoints
│   ├── model.safetensors      # Fine-tuned weights
│   ├── fine-tune_dataset.pkl  # Dataset cache
│   ├── all_results.json       # Aggregated metrics (trainer)
│   ├── test_predictions.csv   # Raw predictions on the test split
│   ├── rank_deltas.csv        # Rank delta report (regression)
│   ├── logs/<timestamp>.log   # Training logs
│   └── final_model/           # Best checkpoint copy
├── predict/
│   ├── predict_dataset.pkl    # Cached inference dataset
│   ├── test_predictions.csv   # Model predictions (IDs + outputs)
│   ├── rank_deltas.csv        # Ranking comparison (regression)
│   ├── logs/<timestamp>.log   # Inference logs
│   └── report.csv             # Legacy report file (legacy modes)
├── interpret/
│   ├── interpret_dataset.pkl  # Cached dataset for LOO scoring
│   ├── loo_scores_mask.csv     # Leave-one-out scores (mask policy)
│   ├── loo_scores_mask.pkl     # Serialized SHAP explanations
│   ├── loo_scores_remove.csv   # Leave-one-out scores (remove policy)
│   ├── loo_scores_remove.pkl   # Serialized explanations
│   └── logs/<timestamp>.log   # Interpret logs
└── mlruns/                     # MLflow tracking data
```

Each mode writes `logs/<timestamp>.log` plus the dataset cache (`<mode>_dataset.pkl`) and any ranking/report files so reproducing a run only needs the appropriate slice of the tree.

### Artifact contents (what to expect)

- **Checkpoints**: Saved under `${outputpath}/pre-train` and `${outputpath}/fine-tune` (plugin-specific filenames, e.g., `model.safetensors`). Reuse them by pointing `inference.pretrainedmodel` (for predict/interpret) or `model_load_path` (for continued training).
- **`test_predictions.csv`**: Typically includes sample identifiers plus plugin-specific scores/probabilities; labels may appear if available. Schemas can differ by plugin—consult the plugin README for exact columns.
- **`loo_scores_<handletokens>.csv` / `.pkl`**: Per-position leave-one-out scores; includes sequence IDs, positions, tokens, and plugin-specific score deltas. The `<handletokens>` suffix reflects the occlusion strategy (`mask`/`remove`).
- **MLflow run folders**: Contain `params`, `metrics`, and `artifacts` (including checkpoints and logs). MLflow UI can browse and download these directly.

---

## 📈 MLflow Tracking

BioLM integrates with MLflow for experiment tracking. To enable MLflow:

1. Set `settings.mlflow.enabled: true` in the configuration.
2. Access the MLflow UI:

   ```bash
   poetry run mlflow ui --backend-store-uri output/mlruns
   ```

3. Download artifacts (e.g., models, logs) directly from the UI.

Tracking is scoped to each run’s `outputpath` (default `${outputpath}/mlruns`) rather than a global store; set `mlflow.tracking_uri` if you want a shared backend.

---

## 📜 Plugin Contract (for plugin authors)

See [docs/PLUGIN_CONTRACT.md](docs/PLUGIN_CONTRACT.md) for the required entry point, factory return shape, and dataset/model/tokenizer expectations.

## 🧪 Testing

Run tests with:

```bash
poetry run pytest tests/
```

For specific suites:

```bash
poetry run pytest tests/integration/      # Plugin system tests
poetry run pytest tests/test_*.py         # Unit tests
```

With coverage:

```bash
poetry run pytest --cov=biolm --cov-report=html
```

### Local commit/push hooks (recommended)

This repository uses `pre-commit` for both `pre-commit` and `pre-push` stages.

Install once per clone:

```bash
poetry run pre-commit install
poetry run pre-commit install --hook-type pre-push
```

Hook lifecycle details and stage behavior are documented in [docs/PLUGIN_CONTRACT.md](docs/PLUGIN_CONTRACT.md#9-local-git-hook-lifecycle-pre-commit-vs-pre-push).

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

**Plugin development:** See Plugin Development Guide below

---

## 📝 Citation

```bibtex
@software{biolm2024,
  title = {BioLM 2.0: A Modular Framework for Biological Language Models},
  author = {Philipp Wiesenbach},
  year = {2024},
  url = {https://github.com/dieterich-lab/biolm_utils}
}
```
