# v2.5.0

**November 2025**

`dlordinal` **v2.5.0** includes the following updates:

---

### New Features
- **Soft-label support for squared EMD loss by ([stefanahaas41](https://github.com/stefanahaas41)) in ([#128](https://github.com/ayrna/dlordinal/pull/128))**
  The squared Earth Mover’s Distance loss now supports soft labels (probability distributions over classes) in addition to deterministic integer labels, enabling proper scoring rule applications and calibration tasks.

- **Optional logarithmic formulation for Weighted Kappa Loss by ([victormvy](https://github.com/victormvy)) in ([#135](https://github.com/ayrna/dlordinal/pull/135))**
  Added the `use_logarithm` parameter, allowing users to compute the Weighted Kappa Loss using the logarithmic form from Torre et al. A small epsilon is applied to mitigate numerical instabilities.
  Documentation updated accordingly.

- **New `_to_numpy` utility for metrics by ([MohammadElSakka](https://github.com/MohammadElSakka)) in ([#134](https://github.com/ayrna/dlordinal/pull/134))**
  Introduced `dlordinal.metrics.metrics._to_numpy`, which safely converts PyTorch tensors, including CUDA tensors and tensors with `requires_grad=True`, into NumPy arrays.

- **Unimodal non-parametric output layer by ([stefanahaas41](https://github.com/stefanahaas41)) in ([#137](https://github.com/ayrna/dlordinal/pull/137))**
  Added a new method based on the unimodal non-parametric approach presented in *“Conformal Prediction Sets for Ordinal Classification”* (NeurIPS 2023). Implemented as a new output layer.

- **New dataset: Historical Color Images (HCI) by ([victormvy](https://github.com/victormvy)) in ([#140](https://github.com/ayrna/dlordinal/pull/140))**
  Added the HCI dataset as a `VisionDataset`, expanding the benchmarking options beyond Adience and FGNet.

- **Extended `_to_numpy` to new metrics by ([MohammadElSakka](https://github.com/MohammadElSakka) and [franberchez](https://github.com/franberchez)) in ([#141](https://github.com/ayrna/dlordinal/pull/141) and [#142](https://github.com/ayrna/dlordinal/pull/142))**
  Metrics introduced in `metrics.py` now also use the `_to_numpy` utility for consistent handling of PyTorch tensors.

### Bug Fixes
- **Target-shape validation fix in `HybridDropout` by ([franberchez](https://github.com/franberchez)) in ([#125](https://github.com/ayrna/dlordinal/pull/125))**
  Corrected dimension checking logic for targets. The expected shape is now explicitly enforced as `[batch_size]`. Tests updated.

- **Corrected scaling factor in `HybridDropout` by ([franberchez](https://github.com/franberchez)) in ([#138](https://github.com/ayrna/dlordinal/pull/138))**
  Fixed the dropout scaling factor (now uses `1 / keep_prob`) and removed an unused variable. Added a unit test validating the behaviour.

- **Pre-commit workflow fix by ([victormvy](https://github.com/victormvy)) in ([#136](https://github.com/ayrna/dlordinal/pull/136))**
  Disabled auto-fixes on forks to resolve CI issues in the pre-commit action.

### Improvements & Documentation
- Updated WKLoss documentation to clarify the relationship between the logarithmic and non-logarithmic formulations, including stability considerations. *(Related to [#135](https://github.com/ayrna/dlordinal/pull/135))*

---
