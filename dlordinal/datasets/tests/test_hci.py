import shutil
from pathlib import Path

import pytest
from torchvision.datasets.utils import check_integrity
from torchvision.transforms import ToTensor

from dlordinal.datasets import HCI

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
def patched_hci(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "torchvision.datasets.utils.download_url",
        fake_download,
    )
    monkeypatch.setattr(
        "dlordinal.datasets.HCI.URL",
        str(TEST_DATA_DIR / "HistoricalColor-ECCV2012-DecadeDatabase.tar"),
    )
    monkeypatch.setattr(
        "dlordinal.datasets.HCI.MD5",
        "03fd41ecaff6311b751404651a68f626",
    )

    def _create(train):
        return HCI(
            root=tmp_path,
            train=train,
            transform=ToTensor(),
        )

    return _create


@pytest.fixture()
def hci_train(patched_hci):
    return patched_hci(train=True)


@pytest.fixture()
def hci_test(patched_hci):
    return patched_hci(train=False)


@pytest.mark.no_gpu_ci
def test_hci_basic(hci_train, hci_test):
    assert len(hci_train) > 0
    assert len(hci_test) > 0


@pytest.mark.no_gpu_ci
def test_hci_categories(hci_train, hci_test):
    train_categories = set()
    for _, label in hci_train:
        train_categories.add(label)
    assert train_categories == {0, 1, 2, 3, 4}

    test_categories = set()
    for _, label in hci_test:
        test_categories.add(label)
    assert test_categories == {0, 1, 2, 3, 4}


@pytest.mark.no_gpu_ci
def test_hci_categories_count(hci_train, hci_test):
    train_category_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for _, label in hci_train:
        train_category_counts[label] += 1
    assert all(count > 0 for count in train_category_counts.values())

    test_category_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for _, label in hci_test:
        test_category_counts[label] += 1
    assert all(count > 0 for count in test_category_counts.values())


@pytest.mark.no_gpu_ci
def test_hci_image_size(hci_train, hci_test):
    for img, _ in hci_train:
        assert img.shape == (3, 224, 224)
    for img, _ in hci_test:
        assert img.shape == (3, 224, 224)


@pytest.mark.no_gpu_ci
def test_hci_md5_verification(tmp_path, hci_train, hci_test):
    # Modify one file to test MD5 verification
    sample_img_path = Path(hci_train.root) / "0" / next(iter(hci_train.samples))[0]
    with open(sample_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()
    assert not hci_test._verify_md5sums()


@pytest.mark.no_gpu_ci
def test_hci_prepare_after_corruption(tmp_path, hci_train, hci_test):
    # Modify one file to test re-preparation
    sample_train_img_path = (
        Path(hci_train.root) / "0" / next(iter(hci_train.samples))[0]
    )
    with open(sample_train_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()
    assert not hci_test._verify_md5sums()

    sample_test_img_path = Path(hci_test.root) / "0" / next(iter(hci_test.samples))[0]
    with open(sample_test_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()
    assert not hci_test._verify_md5sums()

    # Re-prepare dataset
    assert hci_train._prepare_dataset()
    assert hci_train._verify_md5sums()
    # Test set is also repaired since they share the same base directory
    assert hci_test._verify_md5sums()


@pytest.mark.no_gpu_ci
def test_hci_load_data_with_dataloader(hci_train, hci_test):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(hci_train, batch_size=10, shuffle=True)
    test_loader = DataLoader(hci_test, batch_size=10, shuffle=False)

    for images, labels in train_loader:
        assert images.shape == (10, 3, 224, 224)
        assert labels.shape == (10,)
        break

    for images, labels in test_loader:
        assert images.shape == (10, 3, 224, 224)
        assert labels.shape == (10,)
        break
