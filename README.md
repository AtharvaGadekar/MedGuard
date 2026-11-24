# MedGuard

Research prototype for uncertainty-aware classification on PathMNIST. This project is for educational and research use only; it is not a clinical diagnostic tool.

## Environment

This workspace targets Python 3.13 and uses a local `.venv` managed with `uv`.

```bash
uv sync --python 3.13
```

Or activate the already-created environment:

```bash
source .venv/bin/activate
```

## Dataset

Download the official PathMNIST archive and check its predefined split sizes:

```bash
.venv/bin/python scripts/download_data.py
```

The script stores data under `data/raw/` and verifies 89,996 training, 10,004 validation, and 7,180 test images. Data is ignored by Git.

PathMNIST is a nine-class colorectal histology patch dataset distributed by MedMNIST. Images are 28×28 RGB. The official test set comes from a different clinical center than the training/validation data. MedMNIST says the collection is not intended for clinical use. PathMNIST is listed as CC BY 4.0; cite both the MedMNIST paper and the PathMNIST source paper when reporting results.

Sources:

- [MedMNIST v2](https://medmnist.com/v2)
- [Official MedMNIST repository and data links](https://github.com/MedMNIST/MedMNIST)
- PathMNIST source: Kather et al., “Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study,” PLOS Medicine (2019), https://doi.org/10.1371/journal.pmed.1002733

## Project status

- Python environment initialized
- PathMNIST downloaded and split counts audited
- Baseline CNN training script: `scripts/train_baseline.py`

Run the baseline with defaults (10 epochs, seed 42, MPS when available):

```bash
.venv/bin/python scripts/train_baseline.py
```

The best checkpoint is selected by validation macro-F1. Final test metrics, a per-class report, training history, and a confusion matrix are written to `outputs/baseline/`.

## Confidence-based review threshold

Run the selective-prediction evaluation after training:

```bash
.venv/bin/python scripts/evaluate_uncertainty.py --target-coverage 0.80
```

The confidence cutoff is selected using validation predictions to target 80% coverage, then applied unchanged to test predictions. It uses maximum softmax probability as a simple confidence proxy, not calibrated uncertainty. Summary metrics, a risk–coverage curve, and per-image test predictions are written to `outputs/uncertainty/`.

Current preliminary result: the test threshold accepts 82.0% of images and flags 51.3% of test errors for review; accepted-prediction risk is 10.3%. See `outputs/uncertainty/summary.md` for the run details and interpretation limits.

## Architecture comparison and corruption stress

`scripts/train_baseline.py` also supports `--model efficientnet_b0` and `--model vit_tiny`. The original SmallCNN checkpoint is retained as the pilot model. The comparison and stress-test protocol, research question, prior-work boundary, and limitations are documented in `docs/research_positioning.md`.

Reproduce the two architecture runs:

```bash
.venv/bin/python scripts/train_baseline.py --model efficientnet_b0 --epochs 10 --batch-size 256 --seed 42 --output-dir outputs/model_comparison/efficientnet_b0
.venv/bin/python scripts/train_baseline.py --model vit_tiny --epochs 10 --batch-size 256 --seed 42 --output-dir outputs/model_comparison/vit_tiny
```

Run the fixed-threshold blur stress evaluation after the three checkpoints exist:

```bash
.venv/bin/python scripts/evaluate_blur_robustness.py
```

It evaluates clean test images plus five synthetic defocus and motion blur strengths, while keeping each model's review threshold fixed from clean validation data. Results go to `outputs/robustness/`.

Preliminary cross-model and blur findings are summarized in `outputs/robustness/summary.md`; all condition-level metrics are in `outputs/robustness/blur_results.csv`. The research question, prior-work boundary, preliminary hypothesis result, and limitations are in `docs/research_positioning.md`.

## User-facing demo

Launch the local Streamlit interface after the baseline and threshold evaluation have been run:

```bash
.venv/bin/streamlit run app.py
```

The demo uses built-in PathMNIST test examples only. It reports the SmallCNN class prediction, confidence-based review recommendation, and class-probability profile. The blur controls compare the original patch with a synthetic defocus or motion blur. An aggregate stress-test panel also shows the severe-defocus test-set failure separately from the selected image prediction. There is no image-upload control. This is a research demonstration, not a clinical tool.
