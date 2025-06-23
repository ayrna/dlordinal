# v2.4.2

**June 2025**

`dlordinal` **v2.4.2** includes the following updates:

---

## Documentation
- Correction of the documentation of **GeneralTriangularLoss** (`alpha` description) related to issue ([#115](https://github.com/ayrna/dlordinal/issues/115)) ([macontrerascordoba](https://github.com/macontrerascordoba))

---

## Output layers
- **Improved numerical stability of the CLM** output layer with logit link by implementing `stable_sigmoid()`. Removed `clip_warning` parameter, as clipping is no longer needed. Related to pull request ([#116](https://github.com/ayrna/dlordinal/pull/116)) ([RafaAyGar](https://github.com/RafaAyGar))

---

## Loss functions
- The `pred_norm` parameter has been removed from **WKLoss**, as predictions are already probabilities and do not need to be normalised. Related pull request ([#117](https://github.com/ayrna/dlordinal/pull/117)) ([victormvy](https://github.com/victormvy))
