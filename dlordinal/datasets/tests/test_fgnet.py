import shutil
from pathlib import Path

import numpy as np
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

    def _create(train):
        return FGNet(
            root=tmp_path,
            train=train,
            download=True,
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
