# PathMNIST CNN Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a small, reproducible CNN baseline on PathMNIST.

**Architecture:** A single script loads the official splits, trains a compact two-block CNN, selects the checkpoint by validation macro-F1, and evaluates that checkpoint once on the held-out test split. It saves model weights, training history, metrics, and a confusion-matrix visualization.

**Tech Stack:** Python 3.13, PyTorch, torchvision, MedMNIST, scikit-learn, matplotlib.

**Spec:** User request and agreed MedGuard scope in this conversation; no separate written spec.

## Global Constraints

- Keep the official MedMNIST train/validation/test split assignments.
- Use validation data for model selection; reserve test data for final evaluation.
- Keep the prototype educational and non-clinical.
- Reproducibly seed Python, NumPy, and PyTorch.

## Review Focus

- Dataset class IDs and image layout: consume the nine PathMNIST labels and convert RGB HWC images to normalized CHW tensors.
- Split leakage: test data is evaluated only after validation selection and must not affect optimization.
- Shifted class proportions in the official test split: report macro-F1 and per-class metrics in addition to accuracy.
- Apple Silicon: prefer MPS when available and retain a CPU fallback.
- Interrupted or repeated runs: write artifacts under `outputs/baseline/` and keep raw data separate.

---

### Task 1: Baseline training and evaluation script

**Files:**
- Create: `scripts/train_baseline.py`
- Create: `outputs/baseline/` artifacts at runtime
- Modify: `README.md`

**Interfaces:**
- CLI: `python scripts/train_baseline.py --epochs 10 --batch-size 256 --seed 42`
- Produces `outputs/baseline/best_model.pt`, `history.csv`, `metrics.json`, `classification_report.txt`, and `confusion_matrix.png`.

- [x] Load official PathMNIST train/val/test splits with `[0,1]` tensor conversion and fixed-seed shuffling for train only.
- [x] Train the two-convolution CNN with cross-entropy and Adam; select the best checkpoint by validation macro-F1.
- [x] Load the selected checkpoint and calculate final test accuracy, macro-F1, per-class report, and confusion matrix.
- [x] Run the full training command on MPS when available; inspect saved artifacts and record actual results.

**Observed run:** seed 42, 10 epochs, batch size 256, learning rate 0.001, Apple MPS. Best validation macro-F1 was 0.8419; held-out test accuracy was 0.8260 and test macro-F1 was 0.7788. The weakest test-class F1 scores were cancer-associated stroma (0.5397) and debris (0.5711). This is a single-seed preliminary baseline, not a clinical result.
