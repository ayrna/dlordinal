import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision.transforms import ToTensor

from dlordinal.datasets import FGNet

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def fake_download(
    url: str,
    download_root: str,
    filename: str,
    md5: str,
) -> None:
    # Copy local test dataset instead of downloading from the internet
    src = url
    dst = Path(download_root) / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    assert check_integrity(str(dst), md5)


@pytest.fixture()
def patched_fgnet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "torchvision.datasets.utils.download_url",
        fake_download,
    )
    monkeypatch.setattr(
        "dlordinal.datasets.FGNet.URL",
        str(TEST_DATA_DIR / "FGNET.zip"),
    )
    monkeypatch.setattr(
        "dlordinal.datasets.FGNet.MD5",
        "f730a9d177bf61c84b46fb4391c4c473",
    )

    def _create(train, root=tmp_path, **kwargs):
        return FGNet(
            root=root,
            train=train,
            download=kwargs.pop("download", True),
            **kwargs,
        )

    return _create


@pytest.fixture
def fgnet_train(patched_fgnet):
    return patched_fgnet(True)


@pytest.fixture
def fgnet_test(patched_fgnet):
    return patched_fgnet(False)


@pytest.mark.no_gpu_ci
def test_download(fgnet_train):
    fgnet_train.download()
    assert fgnet_train._check_integrity_download()


@pytest.mark.no_gpu_ci
def test_process(fgnet_train):
    fgnet_train.process(
        fgnet_train.root / "FGNET/images",
        fgnet_train.root / "FGNET/data_processed",
    )
    assert fgnet_train._check_integrity_process()


@pytest.mark.no_gpu_ci
def test_split(fgnet_train):
    fgnet_train.split(
        fgnet_train.root / "FGNET/data_processed/fgnet.csv",
        fgnet_train.root / "FGNET/data_processed/train.csv",
        fgnet_train.root / "FGNET/data_processed/test.csv",
        fgnet_train.root / "FGNET/data_processed",
        fgnet_train.root / "FGNET/train",
        fgnet_train.root / "FGNET/test",
    )
    assert fgnet_train._check_integrity_split()


@pytest.mark.no_gpu_ci
def test_find_category(fgnet_train):
    assert fgnet_train.find_category(1) == 0
    assert fgnet_train.find_category(9) == 1
    assert fgnet_train.find_category(14) == 2
    assert fgnet_train.find_category(21) == 3
    assert fgnet_train.find_category(33) == 4


@pytest.mark.no_gpu_ci
def test_get_age_from_filename(fgnet_train):
    filename = "001A12X_X.jpg"
    assert fgnet_train.get_age_from_filename(filename) == 12


@pytest.mark.no_gpu_ci
def test_load_data(fgnet_train):
    data = fgnet_train.load_data(fgnet_train.root / "FGNET/images")
    assert len(data) > 0


@pytest.mark.no_gpu_ci
def test_process_images_from_df(fgnet_train):
    data = fgnet_train.load_data(fgnet_train.root / "FGNET/images")
    processed_images = list((fgnet_train.root / "FGNET/data_processed").rglob("*.JPG"))
    assert len(processed_images) == len(data)


@pytest.mark.no_gpu_ci
def test_split_dataframe(fgnet_train):
    csv_path = fgnet_train.root / "FGNET/data_processed/fgnet.csv"
    train_images_path = fgnet_train.root / "FGNET/train"
    original_images_path = fgnet_train.root / "FGNET/images"
    test_images_path = fgnet_train.root / "FGNET/test"
    train_df, test_df = fgnet_train.split_dataframe(
        csv_path, train_images_path, original_images_path, test_images_path
    )
    assert len(train_df) > 0
    assert len(test_df) > 0


@pytest.mark.no_gpu_ci
def test_getitem(fgnet_train, fgnet_test):
    for fgnet in [fgnet_train, fgnet_test]:
        for i in range(len(fgnet)):
            assert isinstance(fgnet[i][0], Image.Image)
            assert isinstance(fgnet[i][1], int)
            assert fgnet[i][1] == fgnet.targets[i]
            assert np.array(fgnet[i][0]).ndim == 3

        fgnet.transform = ToTensor()

        for i in range(len(fgnet)):
            assert isinstance(fgnet[i][0], torch.Tensor)
            assert isinstance(fgnet[i][1], int)
            assert fgnet[i][1] == fgnet.targets[i]
            assert len(fgnet[i][0].shape) == 3

        fgnet.target_transform = lambda target: np.array(target)
        for i in range(len(fgnet)):
            assert isinstance(fgnet[i][0], torch.Tensor)
            assert isinstance(fgnet[i][1], np.ndarray)
            assert np.array_equal(fgnet[i][1], fgnet.targets[i])


@pytest.mark.no_gpu_ci
def test_len(fgnet_train, fgnet_test):
    for fgnet in [fgnet_train, fgnet_test]:
        assert len(fgnet) == len(fgnet.targets)
        assert len(fgnet) == len(fgnet.data)


@pytest.mark.no_gpu_ci
def test_targets(fgnet_train):
    assert len(fgnet_train.targets) > 0
    assert isinstance(fgnet_train.targets, list)
    assert isinstance(fgnet_train.targets[0], int)
    assert np.all(np.array(fgnet_train.targets) >= 0)


@pytest.mark.no_gpu_ci
def test_classes(fgnet_train, fgnet_test):
    assert len(fgnet_train.classes) == 6
    assert isinstance(fgnet_train.classes, list)
    assert fgnet_train.classes == fgnet_test.classes
    assert fgnet_train.classes == np.unique(fgnet_train.targets).tolist()
    assert fgnet_test.classes == np.unique(fgnet_test.targets).tolist()
    assert fgnet_train.classes == [0, 1, 2, 3, 4, 5]


@pytest.mark.no_gpu_ci
def test_categories_count(fgnet_train, fgnet_test):
    train_category_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, label in fgnet_train:
        train_category_counts[label] += 1
    assert all(count > 0 for count in train_category_counts.values())

    test_category_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, label in fgnet_test:
        test_category_counts[label] += 1
    assert all(count > 0 for count in test_category_counts.values())


def _subjects(dataset):
    return {dataset.get_subject_from_filename(p) for p in dataset.data["path"]}


@pytest.mark.no_gpu_ci
def test_split_is_subject_disjoint(fgnet_train, fgnet_test):
    train_subjects, test_subjects = _subjects(fgnet_train), _subjects(fgnet_test)
    assert None not in train_subjects | test_subjects
    assert train_subjects and test_subjects
    assert train_subjects.isdisjoint(test_subjects)
    all_paths = set(fgnet_train.data["path"]) | set(fgnet_test.data["path"])
    assert len(all_paths) == len(fgnet_train) + len(fgnet_test)


@pytest.mark.no_gpu_ci
def test_get_subject_from_filename(fgnet_train):
    assert fgnet_train.get_subject_from_filename("001A12X_X.jpg") == "001"
    assert fgnet_train.get_subject_from_filename("061A26.JPG") == "061"
    assert fgnet_train.get_subject_from_filename("Thumbs.db") is None


@pytest.mark.no_gpu_ci
def test_cache_is_invalidated_by_parameters(patched_fgnet, tmp_path):
    default = patched_fgnet(True)
    assert default.classes == [0, 1, 2, 3, 4, 5]

    custom = patched_fgnet(True, categories=[20, 40], test_size=0.5)
    fresh = patched_fgnet(
        True, root=tmp_path / "fresh", categories=[20, 40], test_size=0.5
    )
    assert custom.classes == [0, 1, 2]
    assert custom.data.equals(fresh.data)

    # Going back to the defaults restores the default split
    again = patched_fgnet(True)
    assert again.data.equals(default.data)


@pytest.mark.no_gpu_ci
def test_target_size_change_reprocesses_images(fgnet_train, patched_fgnet):
    resized = patched_fgnet(True, target_size=(32, 48))
    assert resized[0][0].size == (48, 32)
    assert fgnet_train[0][0].size == (48, 32)  # same root, images were rebuilt


@pytest.mark.no_gpu_ci
def test_interrupted_processing_is_redone(patched_fgnet, monkeypatch):
    import dlordinal.datasets.fgnet as fgnet_module

    original_imread = fgnet_module.imread
    calls = {"n": 0}

    def flaky_imread(path):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("interrupted")
        return original_imread(path)

    monkeypatch.setattr(fgnet_module, "imread", flaky_imread)
    with pytest.raises(RuntimeError, match="interrupted"):
        patched_fgnet(True)
    monkeypatch.setattr(fgnet_module, "imread", original_imread)

    dataset = patched_fgnet(True)
    processed = dataset.root / "FGNET" / "data_processed"
    assert dataset._check_integrity_process()
    assert all((processed / p).exists() for p in dataset.data["path"])
    for i in range(len(dataset)):
        dataset[i]


@pytest.mark.no_gpu_ci
def test_legacy_cache_without_markers_is_rebuilt(fgnet_train, patched_fgnet):
    processed = fgnet_train.root / "FGNET" / "data_processed"
    expected = fgnet_train.data.copy()
    for marker in processed.glob(".dlordinal_*.json"):
        marker.unlink()
    # A stale train.csv from an old version
    expected.iloc[:1].to_csv(processed / "train.csv", index=False)

    dataset = patched_fgnet(True)
    assert dataset.data.equals(expected)


@pytest.mark.no_gpu_ci
def test_split_does_not_depend_on_listing_order(fgnet_train, tmp_path):
    csv_path = fgnet_train.root / "FGNET/data_processed/fgnet.csv"
    shuffled_path = tmp_path / "shuffled.csv"
    pd.read_csv(csv_path).sample(frac=1, random_state=0).to_csv(
        shuffled_path, index=False
    )
    train_a, test_a = fgnet_train.split_dataframe(csv_path)
    train_b, test_b = fgnet_train.split_dataframe(shuffled_path)
    assert train_a.equals(train_b)
    assert test_a.equals(test_b)


@pytest.mark.no_gpu_ci
def test_load_data_ignores_non_image_files(fgnet_train):
    images = fgnet_train.root / "FGNET" / "images"
    n_images = len(fgnet_train.load_data(images))
    (images / "Thumbs.db").write_bytes(b"x")
    (images / "notes.txt").write_text("x")
    (images / "subfolder").mkdir()
    with pytest.warns(UserWarning, match="cover.jpg"):
        Image.new("RGB", (8, 8)).save(images / "cover.jpg")
        data = fgnet_train.load_data(images)
    assert len(data) == n_images
    names = [name for name, _ in data]
    assert names == sorted(names)


@pytest.mark.no_gpu_ci
def test_targets_apply_target_transform_per_element(fgnet_train):
    fgnet_train.target_transform = float
    targets = fgnet_train.targets
    assert isinstance(targets, np.ndarray)
    assert targets.dtype == np.float64
    assert targets.tolist() == [fgnet_train[i][1] for i in range(len(fgnet_train))]

    # Encodings of equal length are stacked into a 2D array
    fgnet_train.target_transform = lambda y: [1] * y + [0] * (5 - y)
    assert fgnet_train.targets.shape == (len(fgnet_train), 5)
    assert fgnet_train.targets[0].tolist() == fgnet_train[0][1]

    # Ragged results cannot be stacked and are returned as a list
    fgnet_train.target_transform = lambda y: [0] * (y + 1)
    assert isinstance(fgnet_train.targets, list)
    assert fgnet_train.targets[0] == fgnet_train[0][1]


@pytest.mark.no_gpu_ci
def test_targets_with_np_array_transform_is_1d(fgnet_train):
    """Usage in the tutorials: target_transform=np.array, then targets is passed to
    scikit-learn splitters and torch.tensor."""
    from sklearn.utils.multiclass import type_of_target

    fgnet_train.target_transform = np.array
    targets = fgnet_train.targets
    assert isinstance(targets, np.ndarray)
    assert targets.ndim == 1
    assert np.issubdtype(targets.dtype, np.integer)
    assert torch.tensor(targets, dtype=torch.long).shape == (len(fgnet_train),)
    # scikit-learn splitters validate y with type_of_target, which failed with
    # "unhashable type: 'numpy.ndarray'" for a list of 0-d arrays
    assert type_of_target(targets) in ("binary", "multiclass")
    assert np.unique(targets).tolist() == sorted(set(targets.tolist()))


@pytest.mark.no_gpu_ci
def test_categories_default_is_not_shared(patched_fgnet, tmp_path):
    first = patched_fgnet(True)
    first.categories.append(99)
    second = patched_fgnet(True, root=tmp_path / "second")
    assert second.categories == [3, 11, 16, 24, 40]


@pytest.mark.no_gpu_ci
def test_validation_size_is_deprecated(patched_fgnet):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        patched_fgnet(True)
    assert not any("validation_size" in str(w.message) for w in record)

    for validation_size in (0.0001, 0.05, 0.5):
        with pytest.warns(DeprecationWarning, match="validation_size"):
            dataset = patched_fgnet(True, validation_size=validation_size)
        assert len(dataset) > 0


@pytest.mark.no_gpu_ci
def test_no_unused_image_copies(fgnet_train):
    assert not (fgnet_train.root / "FGNET" / "train").exists()
    assert not (fgnet_train.root / "FGNET" / "test").exists()


@pytest.mark.no_gpu_ci
def test_legacy_download_without_marker(fgnet_train, patched_fgnet):
    (fgnet_train.root / "FGNET" / ".dlordinal_downloaded.json").unlink()
    assert not fgnet_train._check_integrity_download()
    with pytest.warns(UserWarning, match="older version"):
        dataset = patched_fgnet(True, download=False)
    assert len(dataset) == len(fgnet_train)

    # download=True verifies and re-extracts the archive
    dataset = patched_fgnet(True, download=True)
    assert dataset._check_integrity_download()


@pytest.mark.no_gpu_ci
def test_missing_dataset_raises(tmp_path):
    with pytest.raises(RuntimeError):
        FGNet(root=tmp_path / "missing", download=False)
