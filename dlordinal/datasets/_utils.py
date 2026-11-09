"""Shared helpers for the datasets.

Cache markers: a cache step (extraction, preprocessing, splitting) is considered
complete only when its JSON marker exists and its content matches the metadata of
the current configuration. Markers are written atomically and only after the step
finished, so an interrupted run is redone instead of being silently reused.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import StratifiedGroupKFold


def n_splits_from_test_size(test_size: float) -> int:
    """Number of folds whose single fold approximates ``test_size``.

    Parameters
    ----------
    test_size : float
        Desired fraction of samples in the test partition, in (0, 1).

    Returns
    -------
    int
        ``max(2, round(1 / test_size))``.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    return max(2, round(1 / test_size))


def stratified_group_split(
    y: ArrayLike, groups: ArrayLike, test_size: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Subject-disjoint, stratified train/test split.

    The samples are split with a
    :class:`sklearn.model_selection.StratifiedGroupKFold` with
    ``n_splits = max(2, round(1 / test_size))`` folds, so that all the samples of
    a group belong to the same partition. One fold is used as test partition, so
    the test fraction is approximately ``1 / n_splits``. The first fold whose train
    and test partitions both contain every class is used, or the first fold if no
    fold does.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Class labels used for stratification.
    groups : array-like of shape (n_samples,)
        Group (subject) of each sample.
    test_size : float
        Approximate fraction of samples in the test partition, in (0, 1).
    seed : int
        Random state of the fold assignment.

    Returns
    -------
    train_idx : np.ndarray
        Indices of the train partition.
    test_idx : np.ndarray
        Indices of the test partition.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits_from_test_size(test_size), shuffle=True, random_state=seed
    )
    n_classes = len(np.unique(y))
    folds = list(sgkf.split(np.zeros(len(y)), y, groups=groups))
    for train_idx, test_idx in folds:
        if (
            len(np.unique(y[train_idx])) == n_classes
            and len(np.unique(y[test_idx])) == n_classes
        ):
            return train_idx, test_idx
    return folds[0]


def read_marker(path: Union[str, Path]) -> Optional[dict]:
    """Read a JSON marker file.

    Parameters
    ----------
    path : str or Path
        Path of the marker file.

    Returns
    -------
    dict or None
        Content of the marker, or ``None`` if it does not exist or is invalid.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None


def marker_matches(path: Union[str, Path], metadata: dict) -> bool:
    """Check whether a marker exists and stores exactly ``metadata``.

    Parameters
    ----------
    path : str or Path
        Path of the marker file.
    metadata : dict
        Expected (JSON-serialisable) content of the marker.

    Returns
    -------
    bool
        ``True`` if the marker content equals ``metadata``.
    """
    # Round-trip through JSON so that tuples compare equal to lists
    return read_marker(path) == json.loads(json.dumps(metadata))


def write_marker(path: Union[str, Path], metadata: dict) -> None:
    """Atomically write a JSON marker file.

    Parameters
    ----------
    path : str or Path
        Path of the marker file.
    metadata : dict
        JSON-serialisable content of the marker.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def remove_marker(path: Union[str, Path]) -> None:
    """Remove a marker file if it exists.

    Parameters
    ----------
    path : str or Path
        Path of the marker file.
    """
    Path(path).unlink(missing_ok=True)


def rename_with_retry(
    src: Union[str, Path], dst: Union[str, Path], attempts: int = 10, delay: float = 0.2
) -> None:
    """Rename ``src`` to ``dst``, retrying on transient ``PermissionError``.

    On Windows, a directory whose files were just written cannot be renamed while
    another process (e.g. an antivirus or indexing service) still holds a handle on
    one of them, which raises ``PermissionError`` for a short time.

    Parameters
    ----------
    src : str or Path
        Path to rename.
    dst : str or Path
        New path.
    attempts : int, default=10
        Maximum number of attempts.
    delay : float, default=0.2
        Seconds to wait between attempts.
    """
    for attempt in range(attempts):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
