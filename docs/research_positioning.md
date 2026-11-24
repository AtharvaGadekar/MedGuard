# MedGuard Research Positioning

## Intended workflow

MedGuard is a research prototype for a pathology researcher reviewing image patches. It predicts one of nine tissue labels for a 28 × 28 PathMNIST patch and flags low-confidence predictions for manual review. It does not diagnose a patient or provide clinical advice.

## Prior work and novelty boundary

Failure detection with abstention has already been studied directly on PathMNIST. *Failure Detection in Deep Neural Networks for Medical Imaging* evaluates Bayesian predictive uncertainty and threshold-based abstention on PathMNIST under noisy conditions. It also reports that raw softmax probabilities from deterministic networks can be overconfident and unsuitable as a reliable uncertainty measure: https://pmc.ncbi.nlm.nih.gov/articles/PMC9359318/

Corruption robustness for MedMNIST has also been benchmarked. *MedMNIST-C* defines pathology-relevant corruptions for PathMNIST, including defocus and motion blur, and evaluates five severity levels. That work uses the 224 × 224 MedMNIST+ data, while this project currently uses the original 28 × 28 PathMNIST version: https://arxiv.org/abs/2406.17536

Therefore, MedGuard will not claim to invent uncertainty estimation, abstention, or medical-image blur testing. Its course-level contribution is a controlled, reproducible comparison of simple and suggested classification architectures and validation-only confidence thresholding under a clearly documented 28 × 28 synthetic blur stress test.

## Research question

> At a threshold selected on clean validation data for 80% validation coverage, does maximum-softmax-confidence review identify more errors than random review at the same review rate when PathMNIST test images receive increasing defocus or motion blur?

## Measurable hypothesis

> For each tested blur severity, confidence-based review will flag a larger fraction of misclassified test images than random review at the same review rate, while the accepted-prediction error rate will increase as blur severity increases.

This hypothesis can be rejected. It does not assume confidence scores remain reliable under corruption.

## Experimental protocol

1. Train the pilot SmallCNN, EfficientNet-B0, and a compact ViT on the same official training split. Select each checkpoint using validation macro-F1.
2. For each checkpoint, derive the maximum-softmax-confidence cutoff from clean validation predictions to target 80% validation coverage. Validation labels are not used to choose this cutoff.
3. Keep that cutoff fixed for clean and corrupted test evaluation. Do not tune on test images or labels.
4. Apply five deterministic severity levels each of defocus-style disk blur and horizontal motion blur to test images only. These are controlled synthetic corruptions at 28 × 28, not official MedMNIST-C benchmark images.
5. Report accuracy, macro-F1, coverage, selective risk, error-review recall, and the random-review reference at matched review rate. Show score trends across severity and per-class results.

## Current pilot result

The SmallCNN reached 0.8260 clean test accuracy and 0.7788 macro-F1. At the 80% validation-coverage threshold, clean test coverage was 0.8202, selective risk was 0.1032, and 0.5132 of test errors were flagged for review. These are single-seed preliminary results.

## Preliminary architecture and blur results

All three models were trained from scratch for 10 epochs with seed 42 and the same optimizer settings. Clean test results were: SmallCNN 0.8260 accuracy / 0.7788 macro-F1; EfficientNet-B0 0.6664 / 0.6114; compact ViT 0.7960 / 0.7280. This is a fixed-budget pilot, not an exhaustive or pretrained architecture comparison.

The defocus review hypothesis was not supported for the SmallCNN at severity 5: test accuracy fell to 0.2097, fixed-threshold coverage rose to 0.953, and error-review recall was 0.043 versus a random-review expectation of 0.047. The model became highly accepting while performing poorly. For motion blur severity 5, its error-review recall was 0.220 versus 0.189 expected under random review. Full per-severity and cross-model results are in `outputs/robustness/summary.md` and `outputs/robustness/blur_results.csv`.

## Limitations to report

- PathMNIST is a small-patch educational benchmark, not a clinical validation set.
- The native 28 × 28 images limit the realism of blur simulation and differ from MedMNIST-C's 224 × 224 inputs.
- Maximum softmax probability is a simple confidence proxy and may be overconfident.
- The one-seed architecture comparison trains from scratch for a fixed 10-epoch budget; EfficientNet and ViT may need different optimization, resolution, augmentation, or pretrained weights.
- The initial comparison uses one seed; run repeated seeds if time permits before making comparative claims.
- The test split comes from a different clinical center and has a shifted class distribution.
