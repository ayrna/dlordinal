# v2.4.0

**April 2025**

`dlordinal` **v2.4.0** includes the following updates:

---

## Loss Functions
- Implemented the **CDWCE** loss function.
- Fixed an issue in **WKLoss** that prevented it from working when `y_true` was one-hot encoded.
- Resolved a device inconsistency in the `WKLoss.forward` method.
- Added **squared Earth Mover’s Distance (EMD)** loss.
- Extended **WKLoss** to support both logits and probabilities (previously supported only probabilities).

---

## Soft Labelling
- Added examples for **geometric soft labels**.
- **Refactored soft labelling losses**:
  These have been reimplemented to support arbitrary loss functions, not just `CrossEntropyLoss`. Additionally, they have been renamed for clarity:
  - `BetaCrossEntropyLoss` → `BetaLoss`
  - `BinomialCrossEntropyLoss` → `BinomialLoss`
  - `CustomTargetsCrossEntropyLoss` → `CustomTargetsLoss`
  - `ExponentialCrossEntropyLoss` → `ExponentialLoss`
  - `GeneralTriangularCrossEntropyLoss` → `GeneralTriangularLoss`
  - `PoissonCrossEntropyLoss` → `PoissonLoss`
  - `TriangularCrossEntropyLoss` → `TriangularLoss`

> ⚠️ The original soft labelling loss classes are now **deprecated** and scheduled for removal in **v3.0.0**.

---

## Maintenance
- Added GPU-enabled versions of all test cases.
