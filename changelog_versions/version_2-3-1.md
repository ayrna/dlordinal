# v2.3.1

January 2025

`dlordinal` `v2.3.1` includes bug fixes related to soft labelling techniques.

# Highlights

- Implementation of a new soft labelling loss function methodology based on the geometric distribution.
- Enhancement of unit tests to ensure that dlordinal works correctly on Ubuntu, Windows, and macOS.
- Pre-commit configuration updated to work seamlessly across different operating systems.
- Development of a new workflow to run Skorch tutorials independently with their own dependencies.

# Loss functions

## Soft labelling

- [BUG] A problem that prevented the use of the binomial soft labelling methodology with some  numbers of classes has been solved.
