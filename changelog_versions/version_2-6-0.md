# v2.6.0

**December 2025**

`dlordinal` **v2.6.0** includes the following updates:

---

### New Features
- **CORN loss function implementation by ([JacksonBurns](https://github.com/JacksonBurns)) in ([#131](https://github.com/ayrna/dlordinal/pull/131))**
  Introduced the CORN loss function.

- **SORD and SLACE loss functions by ([victormvy](https://github.com/victormvy)) in ([#144](https://github.com/ayrna/dlordinal/pull/144))**
  Added SORD and SLACE loss functions.

### Improvements & Maintenance
- **Python 3.14 support by ([victormvy](https://github.com/victormvy)) in ([#145](https://github.com/ayrna/dlordinal/pull/145))**
  Updated the package configuration and CI pipelines to officially support Python 3.14, ensuring compatibility with the latest Python releases and future-proofing the codebase.

- **Improved deprecation handling for soft labelling losses**
  Deprecated losses, such as `BetaCrossEntropyLoss`, now correctly raise a `DeprecationWarning` instead of a `FutureWarning`. Additionally, the test suite has been updated to suppress these warnings for cleaner output during CI.

---

### New Contributors
* **@JacksonBurns** made his first contribution in [#131](https://github.com/ayrna/dlordinal/pull/131)
