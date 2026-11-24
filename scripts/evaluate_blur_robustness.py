"""Evaluate fixed clean-validation review thresholds under synthetic blur."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import PathMNIST
from blur import KERNEL_SIZES, apply_blur
from train_baseline import DATA_ROOT, NUM_CLASSES, PROJECT_ROOT, build_model


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robustness"
MODEL_CHECKPOINTS = {
    "smallcnn": PROJECT_ROOT / "outputs" / "baseline" / "best_model.pt",
    "efficientnet_b0": PROJECT_ROOT / "outputs" / "model_comparison" / "efficientnet_b0" / "best_model.pt",
    "vit_tiny": PROJECT_ROOT / "outputs" / "model_comparison" / "vit_tiny" / "best_model.pt",
}
def threshold_for_coverage(confidences: np.ndarray, target_coverage: float = 0.8) -> float:
    count_to_accept = max(1, int(np.ceil(target_coverage * len(confidences))))
    return float(np.partition(confidences, len(confidences) - count_to_accept)[len(confidences) - count_to_accept])


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device,
            corruption: str | None = None, size: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_truth: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    all_confidence: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            if corruption is not None and size is not None:
                images = apply_blur(images, corruption, size)
            probabilities = model(images).softmax(dim=1)
            confidence, prediction = probabilities.max(dim=1)
            all_truth.append(labels.view(-1).numpy())
            all_prediction.append(prediction.cpu().numpy())
            all_confidence.append(confidence.cpu().numpy())
    return np.concatenate(all_truth), np.concatenate(all_prediction), np.concatenate(all_confidence)


def calculate_metrics(truth: np.ndarray, prediction: np.ndarray, confidence: np.ndarray,
                      threshold: float) -> dict[str, float | int]:
    accepted = confidence >= threshold
    correct = prediction == truth
    errors = ~correct
    coverage = float(accepted.mean())
    accepted_accuracy = float(correct[accepted].mean()) if accepted.any() else float("nan")
    review_rate = 1.0 - coverage
    error_recall = float((errors & ~accepted).sum() / errors.sum()) if errors.any() else float("nan")
    return {
        "samples": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "coverage": coverage,
        "review_rate": review_rate,
        "accepted_accuracy": accepted_accuracy,
        "selective_risk": float(1.0 - accepted_accuracy),
        "errors_flagged_recall": error_recall,
        "random_review_expected_error_recall": review_rate,
        "error_recall_lift_vs_random": float(error_recall / review_rate) if review_rate else float("nan"),
    }


def main() -> None:
    if not (DATA_ROOT / "pathmnist.npz").exists():
        raise FileNotFoundError("PathMNIST missing; run scripts/download_data.py first")
    absent = [name for name, path in MODEL_CHECKPOINTS.items() if not path.exists()]
    if absent:
        raise FileNotFoundError(f"Missing model checkpoints for: {', '.join(absent)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    val_loader = DataLoader(PathMNIST(split="val", root=str(DATA_ROOT), download=False, transform=transform),
                            batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(PathMNIST(split="test", root=str(DATA_ROOT), download=False, transform=transform),
                             batch_size=256, shuffle=False, num_workers=0)
    results: list[dict[str, object]] = []
    model_metadata: dict[str, object] = {}

    for model_name, checkpoint_path in MODEL_CHECKPOINTS.items():
        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        val_truth, val_prediction, val_confidence = predict(model, val_loader, device)
        threshold = threshold_for_coverage(val_confidence)
        val_metrics = calculate_metrics(val_truth, val_prediction, val_confidence, threshold)
        model_metadata[model_name] = {
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "confidence_threshold": threshold,
            "uncertainty_threshold": 1.0 - threshold,
            "clean_validation_metrics": val_metrics,
        }
        print(f"{model_name}: fixed clean-validation confidence threshold={threshold:.4f}", flush=True)

        truth, prediction, confidence = predict(model, test_loader, device)
        clean_metrics = calculate_metrics(truth, prediction, confidence, threshold)
        results.append({"model": model_name, "corruption": "clean", "severity": 0,
                        "kernel_size": 0, "confidence_threshold": threshold, **clean_metrics})
        for corruption in ("defocus", "motion"):
            for severity, kernel_size in enumerate(KERNEL_SIZES, start=1):
                corrupted_truth, corrupted_prediction, corrupted_confidence = predict(
                    model, test_loader, device, corruption=corruption, size=kernel_size
                )
                metrics = calculate_metrics(corrupted_truth, corrupted_prediction, corrupted_confidence, threshold)
                results.append({"model": model_name, "corruption": corruption, "severity": severity,
                                "kernel_size": kernel_size, "confidence_threshold": threshold, **metrics})
                print(
                    f"  {corruption} severity {severity}: acc={metrics['accuracy']:.4f}, "
                    f"coverage={metrics['coverage']:.3f}, risk={metrics['selective_risk']:.4f}, "
                    f"error-recall={metrics['errors_flagged_recall']:.3f}", flush=True,
                )

    fields = list(results[0].keys())
    with (OUTPUT_DIR / "blur_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps({
        "device": str(device),
        "blur_levels": {kind: list(KERNEL_SIZES) for kind in ("defocus", "motion")},
        "severity_definition": "increasing odd square kernel widths 3, 5, 7, 9, 11 on native 28x28 inputs",
        "threshold_selection": "per-model max-softmax confidence quantile on clean validation predictions for 80% target coverage",
        "test_threshold_policy": "fixed from clean validation for clean and every corrupted test condition",
        "random_review_baseline": "expected fraction of errors flagged equals the review rate at each measured coverage",
        "models": model_metadata,
        "results": results,
        "limitation": "Synthetic 28x28 stress tests; not official MedMNIST-C 224x224 corruption data.",
    }, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, corruption in zip(axes, ("defocus", "motion"), strict=True):
        for model_name in MODEL_CHECKPOINTS:
            points = [row for row in results if row["model"] == model_name and row["corruption"] == corruption]
            clean = next(row for row in results if row["model"] == model_name and row["corruption"] == "clean")
            points = [clean, *points]
            axis.plot([row["severity"] for row in points], [row["accuracy"] for row in points],
                      marker="o", label=model_name)
        axis.set(title=f"{corruption.title()} blur", xlabel="Severity")
        axis.set_xticks(range(0, 6), labels=["clean", "1", "2", "3", "4", "5"])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test accuracy")
    axes[1].legend(title="Model", fontsize=8)
    fig.suptitle("PathMNIST blur stress: thresholds fixed from clean validation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "blur_accuracy.png", dpi=160)
    plt.close(fig)

    lines = [
        "# PathMNIST Blur Stress Results", "",
        "All confidence thresholds were chosen on the clean validation split for a target 80% coverage and held fixed for test evaluation.",
        "Blur kernels increase from 3 to 11 pixels on the original 28 × 28 images. This is a controlled synthetic stress test, not official MedMNIST-C data.", "",
        "| Model | Clean accuracy | Clean macro-F1 | Defocus acc. (sev. 5) | Motion acc. (sev. 5) |", "|---|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_CHECKPOINTS:
        clean = next(row for row in results if row["model"] == model_name and row["corruption"] == "clean")
        defocus = next(row for row in results if row["model"] == model_name and row["corruption"] == "defocus" and row["severity"] == 5)
        motion = next(row for row in results if row["model"] == model_name and row["corruption"] == "motion" and row["severity"] == 5)
        lines.append(f"| {model_name} | {clean['accuracy']:.4f} | {clean['macro_f1']:.4f} | {defocus['accuracy']:.4f} | {motion['accuracy']:.4f} |")
    lines.extend([
        "", "## Fixed-threshold selective prediction at highest blur severity", "",
        "| Model | Corruption | Coverage | Accepted error rate | Error recall | Random-review expected recall |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for model_name in MODEL_CHECKPOINTS:
        for corruption in ("defocus", "motion"):
            row = next(row for row in results if row["model"] == model_name and row["corruption"] == corruption and row["severity"] == 5)
            lines.append(f"| {model_name} | {corruption} | {row['coverage']:.3f} | {row['selective_risk']:.4f} | {row['errors_flagged_recall']:.3f} | {row['random_review_expected_error_recall']:.3f} |")
    lines.extend([
        "", "## Interpretation", "",
        "Use `blur_results.csv` to inspect all severities and selective-prediction measures. `errors_flagged_recall` is compared with the expected random-review recall at the same review rate. Coverage can change under corruption because the validation-derived threshold is fixed.",
        "",
        "These results are from one seed and one low-resolution dataset. They are preliminary evidence for the course project, not clinical validation or a new method claim.",
    ])
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Results saved to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
