# PathMNIST Blur Stress Results

All confidence thresholds were chosen on the clean validation split for a target 80% coverage and held fixed for test evaluation.
Blur kernels increase from 3 to 11 pixels on the original 28 × 28 images. This is a controlled synthetic stress test, not official MedMNIST-C data.

| Model | Clean accuracy | Clean macro-F1 | Defocus acc. (sev. 5) | Motion acc. (sev. 5) |
|---|---:|---:|---:|---:|
| smallcnn | 0.8260 | 0.7788 | 0.2097 | 0.3173 |
| efficientnet_b0 | 0.6664 | 0.6114 | 0.3851 | 0.4627 |
| vit_tiny | 0.7960 | 0.7280 | 0.3297 | 0.5205 |

## Fixed-threshold selective prediction at highest blur severity

| Model | Corruption | Coverage | Accepted error rate | Error recall | Random-review expected recall |
|---|---|---:|---:|---:|---:|
| smallcnn | defocus | 0.953 | 0.7937 | 0.043 | 0.047 |
| smallcnn | motion | 0.811 | 0.6566 | 0.220 | 0.189 |
| efficientnet_b0 | defocus | 0.703 | 0.5457 | 0.376 | 0.297 |
| efficientnet_b0 | motion | 0.722 | 0.4538 | 0.390 | 0.278 |
| vit_tiny | defocus | 0.775 | 0.6548 | 0.243 | 0.225 |
| vit_tiny | motion | 0.730 | 0.3976 | 0.395 | 0.270 |

## Interpretation

Use `blur_results.csv` to inspect all severities and selective-prediction measures. `errors_flagged_recall` is compared with the expected random-review recall at the same review rate. Coverage can change under corruption because the validation-derived threshold is fixed.

These results are from one seed and one low-resolution dataset. They are preliminary evidence for the course project, not clinical validation or a new method claim.
