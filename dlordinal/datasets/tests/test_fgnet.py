import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from dlordinal.datasets import FGNet

TMP_DIR = "./tmp_test_dir_fgnet"


@pytest.fixture
def fgnet_train():
    root = TMP_DIR
    fgnet = FGNet(
        root,
        download=True,
        train=True,
    )
    return fgnet


@pytest.fixture
def fgnet_test():
    root = TMP_DIR
    fgnet = FGNet(
        root,
        download=True,
        train=False,
    )
    return fgnet


def test_download(fgnet_train):
    fgnet_train.download()
    assert fgnet_train._check_integrity_download()


def test_process(fgnet_train):
    fgnet_train.process(
        fgnet_train.root / "FGNET/images",
        fgnet_train.root / "FGNET/data_processed",
    )
    assert fgnet_train._check_integrity_process()


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


def test_find_category(fgnet_train):
    assert fgnet_train.find_category(1) == 0
    assert fgnet_train.find_category(9) == 1
    assert fgnet_train.find_category(14) == 2
    assert fgnet_train.find_category(21) == 3
    assert fgnet_train.find_category(33) == 4


def test_get_age_from_filename(fgnet_train):
    filename = "001A12X_X.jpg"
    assert fgnet_train.get_age_from_filename(filename) == 12


def test_load_data(fgnet_train):
    data = fgnet_train.load_data(fgnet_train.root / "FGNET/images")
    assert len(data) > 0


def test_process_images_from_df(fgnet_train):
    data = fgnet_train.load_data(fgnet_train.root / "FGNET/images")
    processed_images = list((fgnet_train.root / "FGNET/data_processed").rglob("*.JPG"))
    assert len(processed_images) == len(data)


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


def test_len(fgnet_train):
    assert len(fgnet_train) > 0


def test_targets(fgnet_train):
    assert len(fgnet_train.targets) > 0


def test_classes(fgnet_train):
    assert len(fgnet_train.classes) == 6


def test_clean_up():
    path = Path(TMP_DIR)
    if path.exists():
        shutil.rmtree(path)
