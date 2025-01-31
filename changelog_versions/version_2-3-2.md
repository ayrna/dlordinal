# v2.3.2

January 2025

`dlordinal` `v2.3.2` includes bug fixes in AMAE and MMAE metrics.

# Metrics

## Bug fixes

- [BUG] A bug was detected in the shapes of the confusion matrix and cost matrix when a class is missing in `y_true` but appears in `y_pred`. This issue is related to ([#94](https://github.com/ayrna/dlordinal/issues/94)) ([angelsevillamol](https://github.com/angelsevillamol))
