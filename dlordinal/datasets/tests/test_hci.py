import shutil
from pathlib import Path

import pytest
from torchvision.transforms import ToTensor

from dlordinal.datasets import HCI


@pytest.fixture(scope="session")
def base_path():
    return ".cache/hci_data"


@pytest.fixture(scope="session")
def hci_train(base_path):
    hci_train = HCI(
        root=base_path,
        train=True,
        transform=ToTensor(),
    )
    return hci_train


@pytest.fixture(scope="session")
def hci_test(base_path):
    hci_test = HCI(
        root=base_path,
        train=False,
        transform=ToTensor(),
    )
    return hci_test


def test_hci_basic(hci_train, hci_test):
    assert len(hci_train) > 0
    assert len(hci_test) > 0


def test_hci_categories(hci_train, hci_test):
    train_categories = set()
    for _, label in hci_train:
        train_categories.add(label)
    assert train_categories == {0, 1, 2, 3, 4}

    test_categories = set()
    for _, label in hci_test:
        test_categories.add(label)
    assert test_categories == {0, 1, 2, 3, 4}


def test_hci_categories_count(hci_train, hci_test):
    train_category_counts = {0: 186, 1: 186, 2: 186, 3: 186, 4: 186}
    for _, label in hci_train:
        train_category_counts[label] += 1
    assert all(count > 0 for count in train_category_counts.values())

    test_category_counts = {0: 79, 1: 79, 2: 79, 3: 79, 4: 79}
    for _, label in hci_test:
        test_category_counts[label] += 1
    assert all(count > 0 for count in test_category_counts.values())


def test_hci_image_size(hci_train, hci_test):
    for img, _ in hci_train:
        assert img.shape == (3, 224, 224)
    for img, _ in hci_test:
        assert img.shape == (3, 224, 224)


def test_hci_md5_verification(base_path, tmp_path):
    dst = tmp_path / "hci"
    shutil.copytree(base_path, dst, dirs_exist_ok=True)
    mutable_hci_train = HCI(
        root=dst,
        train=True,
        transform=ToTensor(),
    )

    mutable_hci_test = HCI(
        root=dst,
        train=False,
        transform=ToTensor(),
    )

    # Modify one file to test MD5 verification
    sample_img_path = (
        Path(mutable_hci_train.root) / "0" / next(iter(mutable_hci_train.samples))[0]
    )
    with open(sample_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not mutable_hci_train._verify_md5sums()
    assert not mutable_hci_test._verify_md5sums()


def test_hci_prepare_after_corruption(base_path, tmp_path, hci_train, hci_test):
    dst = tmp_path / "hci"
    shutil.copytree(base_path, dst, dirs_exist_ok=True)
    mutable_hci_train = HCI(
        root=dst,
        train=True,
        transform=ToTensor(),
    )

    mutable_hci_test = HCI(
        root=dst,
        train=False,
        transform=ToTensor(),
    )

    # Modify one file to test re-preparation
    sample_train_img_path = (
        Path(mutable_hci_train.root) / "0" / next(iter(mutable_hci_train.samples))[0]
    )
    with open(sample_train_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not mutable_hci_train._verify_md5sums()
    assert not mutable_hci_test._verify_md5sums()

    sample_test_img_path = (
        Path(mutable_hci_test.root) / "0" / next(iter(mutable_hci_test.samples))[0]
    )
    with open(sample_test_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not mutable_hci_train._verify_md5sums()
    assert not mutable_hci_test._verify_md5sums()

    # Re-prepare dataset
    assert mutable_hci_train._prepare_dataset()
    assert mutable_hci_train._verify_md5sums()
    # Test set is also repaired since they share the same base directory
    assert mutable_hci_test._verify_md5sums()


def test_hci_load_data_with_dataloader(hci_train, hci_test):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(hci_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(hci_test, batch_size=32, shuffle=False)

    for images, labels in train_loader:
        assert images.shape == (32, 3, 224, 224)
        assert labels.shape == (32,)
        break

    for images, labels in test_loader:
        assert images.shape == (32, 3, 224, 224)
        assert labels.shape == (32,)
        break
