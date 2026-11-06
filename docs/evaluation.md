# Evaluation protocol

Synthetic coordinates are quantised to six decimal places before feature extraction, and recorded model weights to twelve decimal places. This removes insignificant platform-dependent final bits of mathematical functions from the reproducibility artifacts; it does not change split membership or round reported classification decisions.

The generator shuffles group IDs once using the fixed seed, assigns 70/15/15 percent to train/validation/test, and only then generates pairs. A building family is never shared between splits. There are 3,360 training rows and 720 each for validation and test.

Baselines: always-positive; a transparent conjunction of area, price, location and bedroom agreement; logistic regression on the nine declared features. Logistic training is batch gradient descent with 400 epochs, learning rate .7, L2 .002 except the intercept, zero initial weights. The decision threshold is selected from .20 through .80 in .01 steps using validation F1. Ties choose the first threshold. The test set is not used to select weights, thresholds or features.

Report confusion matrix, precision, recall, F1, accuracy and row count for each split. A false positive here means merging two different invented units, an error that could erase a real listing in a production catalogue. A false negative retains a duplicate. Their business costs are different and are not estimated by this benchmark.

Identifiers, split, target and generation scenario are prohibited predictors. Group-separated benchmark scores are not interchangeable with random row cross-validation. Benchmark changes after looking at test results require a new version and a fresh held-out evaluation; do not silently reuse a tuned-on test set as independent validation.
