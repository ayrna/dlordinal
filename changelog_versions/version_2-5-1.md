# v2.5.1

**December 2025**

`dlordinal` **v2.5.1** includes the following updates:

---

### Python compatibility
- Extended compatibility to Python 3.14.
- Handled deprecated losses properly. Now deprecated soft labelling losses such as `BetaCrossEntropyLoss` show a DeprecationWarning instead of a `FutureWarning`. Also, tests now suppress those warnings.

---
