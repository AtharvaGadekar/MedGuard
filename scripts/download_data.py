"""Download and audit the official PathMNIST splits."""
from pathlib import Path

import medmnist
from medmnist import PathMNIST

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"


def main() -> None:
    print(f"MedMNIST version: {medmnist.__version__}")
    print(f"Downloading PathMNIST into: {DATA_DIR}")
    totals = {}
    for split in ("train", "val", "test"):
        dataset = PathMNIST(split=split, root=str(DATA_DIR), download=True)
        totals[split] = len(dataset)
        image, label = dataset[0]
        print(
            f"{split:5}: {len(dataset):,} images; first image={image.size}, "
            f"label={int(label[0])}"
        )

    expected = {"train": 89_996, "val": 10_004, "test": 7_180}
    if totals != expected:
        raise RuntimeError(f"Unexpected split counts: {totals}; expected {expected}")
    print("Audit passed: official PathMNIST split counts match.")


if __name__ == "__main__":
    main()
