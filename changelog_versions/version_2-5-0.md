# v2.4.3

**November 2025**

`dlordinal` **v2.5.0** includes the following updates:

---

### New Features
- **Soft-label support for squared EMD loss (PR #128) @stefanahaas41**
  The squared Earth Mover’s Distance loss now supports soft labels (probability distributions over classes) in addition to deterministic integer labels, enabling proper scoring rule applications and calibration tasks.

- **Optional logarithmic formulation for Weighted Kappa Loss (PR #135) @victormvy**
  Added the `use_logarithm` parameter, allowing users to compute the Weighted Kappa Loss using the logarithmic form from Torre et al. A small epsilon is applied to mitigate numerical instabilities.
  Documentation updated accordingly.

- **New `_to_numpy` utility for metrics (PR #134) @MohammadElSakka**
  Introduced `dlordinal.metrics.metrics._to_numpy`, which safely converts PyTorch tensors—including CUDA tensors and tensors with `requires_grad=True`—into NumPy arrays.

- **Unimodal non-parametric output layer (PR #137) @stefanahaas41**
  Added a new method based on the unimodal non-parametric approach presented in *“Conformal Prediction Sets for Ordinal Classification”* (NeurIPS 2023). Implemented as a new output layer.

- **New dataset: Historical Color Images (HCI) (PR #140) @victormvy**
  Added the HCI dataset as a `VisionDataset`, expanding the benchmarking options beyond Adience and FGNet.

- **Extended `_to_numpy` to new metrics (PR #141, PR #142) @MohammadElSakka @franberchez**
  Metrics introduced in #139 now also use the `_to_numpy` utility for consistent handling of PyTorch tensors.

### Bug Fixes
- **Target-shape validation fix in `HybridDropout` (PR #125) @franberchez**
  Corrected dimension checking logic for targets. The expected shape is now explicitly enforced as `[batch_size]`. Tests updated.

- **Corrected scaling factor in `HybridDropout` (PR #138) @franberchez**
  Fixed the dropout scaling factor (now uses `1 / keep_prob`) and removed an unused variable. Added a unit test validating the behaviour.

- **Pre-commit workflow fix (PR #136) @victormvy**
  Disabled auto-fixes on forks to resolve CI issues in the pre-commit action.

### Improvements & Documentation
- Updated WKLoss documentation to clarify the relationship between the logarithmic and non-logarithmic formulations, including stability considerations. *(Related to PR #135)*

---
