# v2.4.3

**June 2025**

`dlordinal` **v2.4.3** includes the following updates:

---

## Loss functions
- The order of the parameters of the `forward` method in `MCEWKLoss` to match the standard order in pytorch.
- The parameters in the forward methods has been renamed to `input` and `targets` in `MCEWKLoss` and `WKLoss`.
- The tests for `MCEWKLoss` have been updated accordingly.

All this changes are related to pull request ([#120](https://github.com/ayrna/dlordinal/pull/120)) ([franberchez](https://github.com/franberchez))

---
