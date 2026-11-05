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

## Run attribution experiments

Run from the repository root. To use the prepared dataset and trained detector:

Training, evaluation and the attribution notebook use `xai_bench.utils.get_device()` to select CUDA, then MPS, then CPU.

```bash
uv run python scripts/evaluate_attributions.py --config configs/attributions.yaml
```

The experiment evaluates methods on 100 test clips with planted artifacts. Set the checkpoint, number of clips, attribution parameters and output directory in `configs/attributions.yaml`. Outputs are in `results/attributions_narrowband_0db/`:

- `metrics.json`: configuration, per-clip scores, curves and method averages
- `attributions.pt`: computed maps

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

## Methods and metrics

### Planned methods

| Attribution method | Status | Module | Short Description | Reference |
| --- | --- | --- | --- | --- |
| Integrated Gradients (IG) | ✅ | `xai_bench.attributions.integrated_gradients` | Integrates gradients along the path from silence to the input. | [Sundararajan et al. (2017)](https://arxiv.org/abs/1703.01365) |
| SmoothGrad-Squared (SG-SQ) | ✅ | `xai_bench.attributions.smoothgrad_squared` | Averages squared gradients over Gaussian-noised copies of the input. | [Hooker et al. (2019)](https://arxiv.org/abs/1806.10758) |
| Grad-CAM | ✅ | `xai_bench.attributions.grad_cam` | Combines convolutional feature maps using target-class gradients. | [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391) |
| Layer-wise Relevance Propagation (LRP) | - | - | Redistributes the output score backward through the network to input features. | [Bach et al. (2015)](https://doi.org/10.1371/journal.pone.0130140) |
| Occlusion (Feature Ablation) | ✅ | `xai_bench.attributions.occlusion` | Masks sliding regions and measures their effect on the model score. | [Zeiler and Fergus (2014)](https://arxiv.org/abs/1311.2901) |
| Random Map Control | ✅ | `scripts/evaluate_attributions.py` | Assigns independent uniform random importance to each input cell. | |

### Planned metrics

| Evaluation metric | Status | Module | Short Description | Reference |
| --- | --- | --- | --- | --- |
| Ground Truth Recovery ↑ | ✅ | `xai_bench.metrics.ground_truth` | Fraction of absolute attribution mass inside the known artifact mask. | [Zhou et al. (2021)](https://arxiv.org/abs/2104.14403) |
| Deletion AUC ↓ (silence, mean, noise) | ✅ | `xai_bench.metrics.deletion` | Replaces the most important cells and integrates artifact probability. | [Petsiuk et al. (2018)](https://arxiv.org/abs/1806.07421) |
| Insertion AUC ↑ (silence, mean, noise) | ✅ | `xai_bench.metrics.insertion` | Restores the most important cells into a filled input and integrates artifact probability. | [Petsiuk et al. (2018)](https://arxiv.org/abs/1806.07421) |
| Leak-free Imputation (ROAD) | - | - | Fills removed cells from their neighbors with added noise to reduce mask leakage. | [Rong et al. (2022)](https://arxiv.org/abs/2202.00449) |
| Head-only Retraining (ROAR-Head) | - | - | Retrains only the classifier head after masking important cells, keeping the backbone frozen. | [Hooker et al. (2019), ROAR](https://arxiv.org/abs/1806.10758) |
| Infidelity ↓ | - | - | Measures the error between attribution-predicted and actual output changes under perturbations. | [Yeh et al. (2019)](https://arxiv.org/abs/1901.09392) |
| Max-Sensitivity ↓ | - | - | Measures the largest attribution change under small input perturbations. | [Yeh et al. (2019)](https://arxiv.org/abs/1901.09392) |
