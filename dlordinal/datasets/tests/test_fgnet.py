import shutil
from pathlib import Path

import pytest

from dlordinal.datasets import FGNet

TMP_DIR = "./tmp_test_dir_fgnet"


@pytest.fixture
def fgnet_instance():
    root = TMP_DIR
    fgnet = FGNet(root, download=True, process_data=True)
    return fgnet


def test_download(fgnet_instance):
    fgnet_instance.download()
    assert fgnet_instance._check_integrity_download()


def test_process(fgnet_instance):
    fgnet_instance.process(
        fgnet_instance.root / "FGNET/images",
        fgnet_instance.root / "FGNET/data_processed",
    )
    assert fgnet_instance._check_integrity_process()


def test_split(fgnet_instance):
    fgnet_instance.split(
        fgnet_instance.root / "FGNET/data_processed/fgnet.csv",
        fgnet_instance.root / "FGNET/data_processed/train.csv",
        fgnet_instance.root / "FGNET/data_processed/test.csv",
        fgnet_instance.root / "FGNET/data_processed",
        fgnet_instance.root / "FGNET/train",
        fgnet_instance.root / "FGNET/test",
    )
    assert fgnet_instance._check_integrity_split()


def test_find_category(fgnet_instance):
    assert fgnet_instance.find_category(1) == 0
    assert fgnet_instance.find_category(9) == 1
    assert fgnet_instance.find_category(14) == 2
    assert fgnet_instance.find_category(21) == 3
    assert fgnet_instance.find_category(33) == 4


def test_get_age_from_filename(fgnet_instance):
    filename = "001A12X_X.jpg"
    assert fgnet_instance.get_age_from_filename(filename) == 12


def test_load_data(fgnet_instance):
    data = fgnet_instance.load_data(fgnet_instance.root / "FGNET/images")
    assert len(data) > 0


def test_process_images_from_df(fgnet_instance):
    data = fgnet_instance.load_data(fgnet_instance.root / "FGNET/images")
    processed_images = list(
        (fgnet_instance.root / "FGNET/data_processed").rglob("*.JPG")
    )
    assert len(processed_images) == len(data)


def test_split_dataframe(fgnet_instance):
    csv_path = fgnet_instance.root / "FGNET/data_processed/fgnet.csv"
    train_images_path = fgnet_instance.root / "FGNET/train"
    original_images_path = fgnet_instance.root / "FGNET/images"
    test_images_path = fgnet_instance.root / "FGNET/test"
    train_df, test_df = fgnet_instance.split_dataframe(
        csv_path, train_images_path, original_images_path, test_images_path
    )
    assert len(train_df) > 0
    assert len(test_df) > 0


def test_clean_up():
    path = Path(TMP_DIR)
    if path.exists():
        shutil.rmtree(path)
