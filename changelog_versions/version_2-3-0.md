# v2.3.0

January 2025

`dlordinal` `v2.3.0` this release includes several corrections and the implementation of a new methodology for loss functions for soft labelling. We would like to thank all contributors who helped make this release possible.

# Highlights

- Implementation of a new soft labelling loss function methodology based on the geometric distribution.
- Enhancement of unit tests to ensure that dlordinal works correctly on Ubuntu, Windows, and macOS.
- Pre-commit configuration updated to work seamlessly across different operating systems.
- Development of a new workflow to run Skorch tutorials independently with their own dependencies.

# General

## Bug fixes

- [BUG] A bug was detected in the pre-commit configuration related to an issue with extending the package to include a new loss function oriented to soft labelling. ([#87](https://github.com/ayrna/dlordinal/issues/87))([stefanahaas41] (https://github.com/stefanahaas41))

## Maintenance

- [MNT] Pre-commit fixes ([#89](https://github.com/ayrna/dlordinal/pull/89)) ([victormvy](https://github.com/victormvy))


# Tutorials

## Maintenance

- [MNT] Skorch tutorials separately with its own skorch dependecy ([#90](https://github.com/ayrna/dlordinal/pull/90)) ([victormvy](https://github.com/victormvy))


# Tests

## Maintenance

- [MNT] Test update with new checks for differents OS ([#92](https://github.com/ayrna/dlordinal/pull/92)) ([franberchez](https://github.com/franberchez))


# Loss functions

## Documentation

- [DOC] Extension of the documentation for loss functions ([#88](https://github.com/ayrna/dlordinal/pull/88)) ([stefanahaas41](https://github.com/stefanahaas41))

## Enhancements

- [ENH] Implementation of a new geometric loss function ([#88](https://github.com/ayrna/dlordinal/pull/88)) ([stefanahaas41](https://github.com/stefanahaas41))


## Soft Labelling

## Documentation

- [DOC] Extension of the documentation for soft labelling ([#88](https://github.com/ayrna/dlordinal/pull/88)) ([stefanahaas41](https://github.com/stefanahaas41))

## Enhancements

- [ENH] Implementation of geometric distribution for soft labelling ([#88](https://github.com/ayrna/dlordinal/pull/88)) ([stefanahaas41](https://github.com/stefanahaas41))
