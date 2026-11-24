"""Train and evaluate a small PathMNIST CNN baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm

from medmnist import INFO, PathMNIST


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "baseline"
NUM_CLASSES = 9


class SmallCNN(nn.Module):
    """Compact CNN for 28x28 RGB images."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class TinyVisionTransformer(nn.Module):
    """Small ViT with 4x4 patches for the native 28x28 PathMNIST images."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        width = 128
        self.patch_embedding = nn.Conv2d(3, width, kernel_size=4, stride=4)
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position_embedding = nn.Parameter(torch.zeros(1, 50, width))  # 49 patches + class token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=256, dropout=0.1,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(images.size(0), -1, -1)
        tokens = torch.cat((class_token, patches), dim=1) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded[:, 0]))


def build_model(model_name: str) -> nn.Module:
    if model_name == "smallcnn":
        return SmallCNN()
    if model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model
    if model_name == "vit_tiny":
        return TinyVisionTransformer()
    raise ValueError(f"Unknown model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smallcnn", "efficientnet_b0", "vit_tiny"), default="smallcnn")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(batch_size: int, seed: int, num_workers: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()  # uint8 RGB HWC -> float32 CHW scaled to [0, 1]
    datasets = {
        split: PathMNIST(split=split, root=str(DATA_ROOT), download=False, transform=transform)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            generator=generator, num_workers=num_workers,
        ),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders["train"], loaders["val"], loaders["test"]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
            images = images.to(device)
            labels = labels.view(-1).long().to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            total_examples += batch_count
            all_targets.append(labels.detach().cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    return total_loss / total_examples, np.concatenate(all_targets), np.concatenate(all_predictions)


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of images")
    ax.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted label", ylabel="True label", title="PathMNIST test confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]}", ha="center", va="center",
                    color="white" if matrix[row, column] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive; num-workers cannot be negative")
    if not (DATA_ROOT / "pathmnist.npz").exists():
        raise FileNotFoundError("PathMNIST not found. Run: .venv/bin/python scripts/download_data.py")

    set_seed(args.seed)
    output_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(args.batch_size, args.seed, args.num_workers)
    class_names = [INFO["pathmnist"]["label"][str(i)] for i in range(NUM_CLASSES)]

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_f1 = -1.0
    history: list[dict[str, float | int]] = []
    best_checkpoint = output_root / "best_model.pt"

    print(f"Device: {device}", flush=True)
    print(f"Splits: train={len(train_loader.dataset):,}, val={len(val_loader.dataset):,}, test={len(test_loader.dataset):,}", flush=True)
    print(f"Model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_targets, val_predictions = run_epoch(model, val_loader, criterion, device)
        train_accuracy = accuracy_score(train_targets, train_predictions)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "val_loss": val_loss, "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
        })
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), best_checkpoint)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_accuracy:.4f} "
            f"| val loss {val_loss:.4f} acc {val_accuracy:.4f} macro-F1 {val_macro_f1:.4f} "
            f"| best val macro-F1 {best_val_f1:.4f}",
            flush=True,
        )

    with (output_root / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
    test_loss, test_targets, test_predictions = run_epoch(model, test_loader, criterion, device)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_macro_f1 = f1_score(test_targets, test_predictions, average="macro", zero_division=0)
    report = classification_report(
        test_targets, test_predictions, labels=np.arange(NUM_CLASSES), target_names=class_names,
        digits=4, zero_division=0,
    )
    matrix = confusion_matrix(test_targets, test_predictions, labels=np.arange(NUM_CLASSES))
    (output_root / "classification_report.txt").write_text(report)
    save_confusion_matrix(matrix, class_names, output_root / "confusion_matrix.png")

    metrics = {
        "dataset": "PathMNIST",
        "model": args.model,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "test_samples": int(len(test_targets)),
        "class_names": class_names,
        "device": str(device),
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "torch_version": torch.__version__,
    }
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("\nFinal held-out test results", flush=True)
    print(f"Accuracy: {test_accuracy:.4f} | Macro-F1: {test_macro_f1:.4f} | Loss: {test_loss:.4f}", flush=True)
    print(f"Artifacts saved to: {output_root}", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
