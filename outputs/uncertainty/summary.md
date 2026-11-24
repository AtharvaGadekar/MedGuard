# PathMNIST Confidence-Based Abstention

## Method

The model's maximum softmax probability is used as a simple confidence proxy; uncertainty is defined as one minus that value. A confidence threshold was selected from validation predictions to target 80% validation coverage. The threshold was then applied unchanged to the test split. Validation labels were not used to select the threshold.

## Results

| Metric | Validation | Held-out test |
|---|---:|---:|
| Threshold confidence | 0.6024 | 0.6024 |
| Coverage | 0.800 | 0.820 |
| Selective accuracy | 0.9245 | 0.8968 |
| Selective risk (accepted error rate) | 0.0755 | 0.1032 |
| Errors flagged for review (recall) | 0.6121 | 0.5132 |

For context, full-coverage test accuracy is 0.8260. See `risk_coverage_curve.png` and `risk_coverage_curve.csv` for the operating-point sweep, and `test_predictions.csv` for per-image outputs.

## Interpretation limit

Maximum softmax probability is a basic confidence score, not a guarantee of calibrated uncertainty. This operating point targets validation coverage; achieved test coverage can differ under the test-set distribution shift. This is a research prototype, not a clinical tool.
