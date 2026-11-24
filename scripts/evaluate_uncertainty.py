"""Evaluate confidence-based selective prediction on PathMNIST."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from medmnist import INFO, PathMNIST
from train_baseline import DATA_ROOT, NUM_CLASSES, OUTPUT_ROOT as BASELINE_OUTPUT, SmallCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "uncertainty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-coverage", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truths: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in tqdm(loader, leave=False, desc="predict"):
            probabilities = model(images.to(device)).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            truths.append(labels.view(-1).numpy())
            predictions.append(prediction.cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(confidences)


def threshold_for_coverage(confidences: np.ndarray, target_coverage: float) -> float:
    """Return validation confidence cut-off accepting at least the requested fraction."""
    count_to_accept = max(1, int(np.ceil(target_coverage * len(confidences))))
    index = len(confidences) - count_to_accept
    return float(np.partition(confidences, index)[index])


def selective_metrics(
    truths: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    accepted = confidences >= threshold
    correct = predictions == truths
    errors = ~correct
    accepted_count = int(accepted.sum())
    total_errors = int(errors.sum())
    selective_accuracy = float(correct[accepted].mean()) if accepted_count else float("nan")
    error_recall = float((errors & ~accepted).sum() / total_errors) if total_errors else float("nan")
    return {
        "sample_count": int(len(truths)),
        "coverage": float(accepted.mean()),
        "accepted_count": accepted_count,
        "review_count": int((~accepted).sum()),
        "overall_accuracy": float(accuracy_score(truths, predictions)),
        "selective_accuracy": selective_accuracy,
        "selective_risk": float(1.0 - selective_accuracy),
        "errors_flagged_for_review_recall": error_recall,
        "confidence_threshold": float(threshold),
        "uncertainty_threshold": float(1.0 - threshold),
    }


def save_curve(
    val_truth: np.ndarray,
    val_prediction: np.ndarray,
    val_confidence: np.ndarray,
    test_truth: np.ndarray,
    test_prediction: np.ndarray,
    test_confidence: np.ndarray,
    path_csv: Path,
    path_png: Path,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for target in np.linspace(0.2, 1.0, 17):
        threshold = threshold_for_coverage(val_confidence, float(target))
        val_result = selective_metrics(val_truth, val_prediction, val_confidence, threshold)
        accepted = test_confidence >= threshold
        coverage = float(accepted.mean())
        accuracy = float((test_prediction[accepted] == test_truth[accepted]).mean()) if accepted.any() else float("nan")
        rows.append({
            "target_validation_coverage": float(target),
            "validation_coverage": float(val_result["coverage"]),
            "validation_selective_risk": float(val_result["selective_risk"]),
            "confidence_threshold": threshold,
            "test_coverage": coverage,
            "test_selective_risk": float(1.0 - accuracy),
        })
    with path_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    curve_points = [(1.0, float(1.0 - accuracy_score(test_truth, test_prediction)))] + [
        (row["test_coverage"], row["test_selective_risk"]) for row in rows
    ]
    curve_points.sort(key=lambda point: point[0])
    coverage_values = [point[0] for point in curve_points]
    risk_values = [point[1] for point in curve_points]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage_values, risk_values, marker="o", markersize=3)
    ax.set(xlabel="Test coverage (fraction accepted)", ylabel="Test selective risk (error rate accepted)",
           title="PathMNIST risk–coverage curve\nthresholds chosen from validation confidence")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path_png, dpi=160)
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    if not 0 < args.target_coverage <= 1:
        raise ValueError("target-coverage must be greater than 0 and at most 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    checkpoint_path = BASELINE_OUTPUT / "best_model.pt"
    dataset_path = DATA_ROOT / "pathmnist.npz"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Baseline checkpoint missing. Run scripts/train_baseline.py first.")
    if not dataset_path.exists():
        raise FileNotFoundError("PathMNIST missing. Run scripts/download_data.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    val_dataset = PathMNIST(split="val", root=str(DATA_ROOT), download=False, transform=transform)
    test_dataset = PathMNIST(split="test", root=str(DATA_ROOT), download=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    val_truth, val_prediction, val_confidence = predict(model, val_loader, device)
    test_truth, test_prediction, test_confidence = predict(model, test_loader, device)

    threshold = threshold_for_coverage(val_confidence, args.target_coverage)
    validation_metrics = selective_metrics(val_truth, val_prediction, val_confidence, threshold)
    test_metrics = selective_metrics(test_truth, test_prediction, test_confidence, threshold)
    curve_rows = save_curve(
        val_truth, val_prediction, val_confidence, test_truth, test_prediction, test_confidence,
        OUTPUT_DIR / "risk_coverage_curve.csv", OUTPUT_DIR / "risk_coverage_curve.png",
    )

    class_names = [INFO["pathmnist"]["label"][str(i)] for i in range(NUM_CLASSES)]
    accepted_test = test_confidence >= threshold
    with (OUTPUT_DIR / "test_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["index", "true_label_id", "true_label", "predicted_label_id", "predicted_label",
                         "confidence", "uncertainty", "accepted", "correct"])
        for index, (truth, prediction, confidence, accepted) in enumerate(
            zip(test_truth, test_prediction, test_confidence, accepted_test, strict=True)
        ):
            writer.writerow([index, int(truth), class_names[truth], int(prediction), class_names[prediction],
                             f"{confidence:.8f}", f"{1-confidence:.8f}", bool(accepted), bool(truth == prediction)])

    result = {
        "method": "maximum softmax probability confidence threshold",
        "confidence_definition": "max(softmax(logits))",
        "uncertainty_definition": "1 - max(softmax(logits))",
        "threshold_selection": "validation confidence quantile; labels are not used to choose threshold",
        "target_validation_coverage": args.target_coverage,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "validation": validation_metrics,
        "test": test_metrics,
        "full_coverage_test_accuracy": float(accuracy_score(test_truth, test_prediction)),
        "risk_coverage_operating_points": len(curve_rows),
        "limitation": "Softmax confidence is a simple uncertainty proxy and is not guaranteed to be calibrated.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary = f"""# PathMNIST Confidence-Based Abstention

## Method

The model's maximum softmax probability is used as a simple confidence proxy; uncertainty is defined as one minus that value. A confidence threshold was selected from validation predictions to target {args.target_coverage:.0%} validation coverage. The threshold was then applied unchanged to the test split. Validation labels were not used to select the threshold.

## Results

| Metric | Validation | Held-out test |
|---|---:|---:|
| Threshold confidence | {threshold:.4f} | {threshold:.4f} |
| Coverage | {validation_metrics['coverage']:.3f} | {test_metrics['coverage']:.3f} |
| Selective accuracy | {validation_metrics['selective_accuracy']:.4f} | {test_metrics['selective_accuracy']:.4f} |
| Selective risk (accepted error rate) | {validation_metrics['selective_risk']:.4f} | {test_metrics['selective_risk']:.4f} |
| Errors flagged for review (recall) | {validation_metrics['errors_flagged_for_review_recall']:.4f} | {test_metrics['errors_flagged_for_review_recall']:.4f} |

For context, full-coverage test accuracy is {test_metrics['overall_accuracy']:.4f}. See `risk_coverage_curve.png` and `risk_coverage_curve.csv` for the operating-point sweep, and `test_predictions.csv` for per-image outputs.

## Interpretation limit

Maximum softmax probability is a basic confidence score, not a guarantee of calibrated uncertainty. This operating point targets validation coverage; achieved test coverage can differ under the test-set distribution shift. This is a research prototype, not a clinical tool.
"""
    (OUTPUT_DIR / "summary.md").write_text(summary)
    print(f"Device: {device}")
    print(f"Validation-selected confidence threshold: {threshold:.4f} (target coverage {args.target_coverage:.1%})")
    print(f"Validation: coverage={validation_metrics['coverage']:.3f}, risk={validation_metrics['selective_risk']:.4f}")
    print(f"Test: coverage={test_metrics['coverage']:.3f}, selective risk={test_metrics['selective_risk']:.4f}, "
          f"full-coverage accuracy={test_metrics['overall_accuracy']:.4f}")
    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
