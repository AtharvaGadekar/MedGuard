# MedGuard — Class Presentation Script

**Presenters:** Atharva and Kirtiraj
**Target length:** about 8–9 minutes
**Format:** Read the speaker text naturally. The transition line is a cue for the next speaker, not part of the main explanation.

---

## Slide 1 — MedGuard
**Speaker: Atharva | ~20 sec**

“Good morning. We are Atharva and Kirtiraj, and our project is MedGuard. We built a research prototype that classifies small colorectal tissue image patches and flags predictions for human review when the model’s confidence is low. We also tested how the models and review rule behave when the images are blurred.”

**Transition:** “I’ll first explain the problem we chose to study.”

## Slide 2 — Why MedGuard?
**Speaker: Atharva | ~35 sec**

“Our stakeholder is a pathology research team reviewing image patches, not a patient seeking a diagnosis. The input is a 28-by-28 RGB PathMNIST patch. The system returns probabilities across nine tissue classes, its top prediction, and a review recommendation based on a confidence threshold. The intended benefit is to route some uncertain predictions to a human reviewer. This is an educational research prototype, not a clinical tool.”

## Slide 3 — Research question
**Speaker: Atharva | ~40 sec**

“Our question is whether a confidence threshold selected on clean validation images can still identify model errors when test images are blurred. We set the threshold to target 80 percent validation coverage, then kept it fixed during testing. Our hypothesis was that confidence review would catch more errors than random review at the same review rate. Prior work already studies medical-image failure detection and corruption benchmarks, so our contribution is a focused course experiment, not a new uncertainty method.”

**Handoff:** “Kirtiraj will explain the data and how we implemented the experiments.”

## Slide 4 — Dataset: PathMNIST
**Speaker: Kirtiraj | ~35 sec**

“We used PathMNIST from MedMNIST. It contains nine colorectal histology tissue classes, with 89,996 training images, 10,004 validation images, and 7,180 test images. Each image is 28 by 28 pixels and has three color channels. We used the official train, validation, and test splits. The dataset is distributed under CC BY 4.0, and it is intended as a research benchmark rather than clinical evidence.”

## Slide 5 — Implementation pipeline
**Speaker: Kirtiraj | ~35 sec**

“The implementation has four stages. First, we loaded the official PathMNIST splits and converted the images to tensors. Second, we trained each model using the training split and selected its best checkpoint by validation macro-F1. Third, we calibrated a confidence cutoff on validation predictions. Finally, we evaluated clean test images and synthetic blur conditions, saved metrics and plots, and connected the SmallCNN checkpoint to the Streamlit demo.”

## Slide 6 — Models we used
**Speaker: Kirtiraj | ~40 sec**

“We compared three models. SmallCNN is our compact convolutional baseline: two convolution and pooling blocks followed by a small fully connected classifier. EfficientNet-B0 is a standard convolutional architecture, with its final layer changed to output nine classes. Our compact ViT divides the image into four-by-four patches and processes them with transformer layers. All three were trained from scratch, without pretrained weights.”

## Slide 7 — Training and evaluation setup
**Speaker: Kirtiraj | ~35 sec**

“To keep the pilot comparison simple, we used seed 42, ten epochs, batch size 256, Adam with a learning rate of 0.001, and cross-entropy loss. We used the same official splits and selected checkpoints using validation macro-F1. Runs used Apple’s MPS device. This is a fixed-budget, single-seed comparison, so it is useful as a baseline but does not prove that one architecture is generally superior.”

**Handoff:** “Now Atharva will show the measured results and the confidence-review experiment.”

## Slide 8 — Clean test baseline
**Speaker: Atharva | ~35 sec**

“On the clean test set, SmallCNN achieved 82.6 percent accuracy and a macro-F1 of 0.779. The compact ViT reached 79.6 percent accuracy and 0.728 macro-F1. EfficientNet-B0 reached 66.6 percent and 0.611. SmallCNN led under this training budget. Macro-F1 is included because it averages performance across classes, rather than letting the largest classes dominate the summary.”

## Slide 9 — Confidence-based review
**Speaker: Atharva | ~40 sec**

“For the SmallCNN, we used maximum softmax probability as a simple confidence score. We selected a threshold of 0.6024 on validation predictions to accept about 80 percent of cases, without using validation labels to choose the cutoff. On the test set, 82 percent were accepted, and the accepted-case error rate was 10.3 percent. The rule flagged 51.3 percent of test errors for review. This score is a proxy; we do not claim it is calibrated uncertainty.”

## Slide 10 — Blur stress-test protocol
**Speaker: Atharva | ~35 sec**

“We then applied two deterministic synthetic corruptions to the held-out test images: defocus-style blur and horizontal motion blur. Each had five increasing kernel sizes, from 3 to 11 pixels. We kept each model’s threshold fixed at the value selected on clean validation data. We measured accuracy, macro-F1, coverage, selective risk, and the fraction of errors flagged for review, compared with a random-review reference at the same review rate.”

## Slide 11 — Accuracy under blur
**Speaker: Atharva | ~35 sec**

“All three models lost accuracy under severe blur, especially defocus. SmallCNN fell from 82.6 percent clean accuracy to 21.0 percent under severe defocus and 31.7 percent under the strongest motion blur. EfficientNet-B0 had the highest severe-defocus accuracy at 38.5 percent, while ViT had the highest motion-blur accuracy at 52.0 percent. So there is no single winner across both corruption types.”

**Handoff:** “Kirtiraj will explain what happened to the review rule and show the demo.”

## Slide 12 — A failure case for confidence review
**Speaker: Kirtiraj | ~40 sec**

“The most important failure case is SmallCNN under severe defocus. Its accuracy fell to about 21 percent, but the fixed threshold still accepted 95.3 percent of images. The review rule caught only 4.3 percent of errors, slightly below the 4.7 percent expected from random review at that review rate. Under strong motion blur, error review did better than random for all three models, but the result depends on the corruption. This is why we report the negative result.”

## Slide 13 — User-facing demonstration
**Speaker: Kirtiraj | ~35 sec**

“Our Streamlit demo lets a user choose a built-in PathMNIST test example and view the predicted class, confidence, class-probability profile, and review recommendation. The blur controls let us compare the original image with a synthetic stress version. We removed image upload and use built-in examples for this class demonstration. The screen is designed to make the model’s output and review decision easy to inspect.”

## Slide 14 — What we learned
**Speaker: Kirtiraj | ~35 sec**

“Our main finding is that strong clean accuracy does not guarantee robustness, and a fixed confidence threshold can fail under a new corruption. The model comparison also changes depending on whether the stress is defocus or motion blur. Important limitations are the low 28-by-28 resolution, synthetic corruption, one training seed, and the use of maximum softmax probability. These results are preliminary and do not support clinical use.”

“That is what we learned from the implemented prototype and experiments.”

## Slide 15 — Sources
**Speaker: Kirtiraj | ~10 sec**

“Our dataset and prior-work sources are listed here, including the MedMNIST benchmark, the original PathMNIST study, medical-image failure detection work, and the MedMNIST-C corruption benchmark. Thank you. We’re happy to take questions about the implementation, evaluation, or failure cases.”
