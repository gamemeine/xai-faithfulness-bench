# Is the faithfulness metric faithful?

## About

Attribution methods are scored by deletion: remove what the map calls important and watch the prediction fall. The removal creates inputs the model never saw, so the drop may measure the artifact. Vision answered with retraining and with leak-free imputation. Audio has not checked which agree and has no benchmark with a known ground truth. You will build that ground truth from partially spoofed speech (the spliced synthetic segment is what a map should find) and from planted artifacts, score the metrics by whether they rank methods the way the ground truth does, and recommend what to report. A text variant does the same on a detector of machine-generated text.

The desing proposal can be found in [DESIGN_PROPOSAL.md](DESIGN_PROPOSAL.md).

## Setup

Requires Python 3.12 and `uv`

```bash
# install virtual environment and dependencies
uv sync

# authenticate with Hugging Face (required for data/model upload & download)
uv run hf auth login
```

## Data

Original audio recordings come from the [ASVspoof 2019 dataset](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45).
All data is stored on HF at `xai-faithfulness-bench/data`

```bash
# download only processed .pt tensors (for training, evaluation, and XAI)
uv run hf download xai-faithfulness-bench/data --repo-type dataset --include "processed/**" --local-dir ./data

# download everything (raw bonafide audio + processed tensors)
uv run hf download xai-faithfulness-bench/data --repo-type dataset --local-dir ./data

# upload updated data to HF
uv run hf upload xai-faithfulness-bench/data ./data . --repo-type dataset
```

## Scripts

Each stage uses a dedicated YAML config in `configs/`

```bash
# generate log-mel spectrograms, GT masks, and inject planted artifacts from raw audio
uv run python scripts/prepare_data.py --config configs/prepare_data.yaml

# optional: export 3 random WAV pairs (clean vs planted) into samples/ for listening
uv run python scripts/save_samples.py --config configs/prepare_data.yaml

# train ResNet-18 detector (logs in tb_logs/, checkpoints in checkpoints/)
uv run python scripts/train.py --config configs/train.yaml

# inspect training curves in TensorBoard
uv run tensorboard --logdir tb_logs

# evaluate on test set and verify decision flip after artifact removal (output in results/)
uv run python scripts/evaluate.py --config configs/eval.yaml
```

## Checkpoints

Trained model checkpoints are stored on HF at `xai-faithfulness-bench/checkpoints`

```bash
# download trained checkpoints
uv run hf download xai-faithfulness-bench/checkpoints --repo-type model --local-dir ./checkpoints

# upload new checkpoints to HF
uv run hf upload xai-faithfulness-bench/checkpoints ./checkpoints . --repo-type model
```

## Linting & Formatting

```bash
uv run ruff check --fix .
uv run ruff format .
```
