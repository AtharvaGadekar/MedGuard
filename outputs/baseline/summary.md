# PathMNIST SmallCNN Baseline

## Run configuration

- Model: two convolution + max-pooling blocks, 128-unit hidden layer, dropout 0.25, nine output classes
- Optimizer: Adam, learning rate 0.001
- Loss: cross-entropy
- Epochs: 10; batch size: 256; seed: 42
- Device: Apple MPS
- Checkpoint selection: highest validation macro-F1
- Input: 28 × 28 RGB converted to float tensors in [0, 1]

## Results

| Split | Accuracy | Macro-F1 |
|---|---:|---:|
| Best validation checkpoint | — | 0.8419 |
| Held-out test | 0.8260 | 0.7788 |

The test set contains images from a different clinical center and has different class proportions from the training and validation splits. This one-seed run is an initial baseline. Lower test F1 was observed for cancer-associated stroma (0.5397) and debris (0.5711); see `classification_report.txt` and `confusion_matrix.png` for all classes.

## Artifacts

- `best_model.pt`: checkpoint selected on validation macro-F1
- `history.csv`: per-epoch train and validation metrics
- `metrics.json`: configuration and aggregate results
- `classification_report.txt`: test precision, recall, F1, and support by class
- `confusion_matrix.png`: test-set confusion matrix

## Reproduce

From the project root, run:

```bash
.venv/bin/python scripts/train_baseline.py --epochs 10 --batch-size 256 --learning-rate 0.001 --seed 42
```

For interpretation and limits of PathMNIST, see the project README. This research prototype is not intended for clinical use.
