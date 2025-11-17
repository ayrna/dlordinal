from torchvision.transforms import ToTensor

from dlordinal.datasets import HCI


def test_hci_basic(tmp_path):
    hci_train = HCI(
        root=tmp_path,
        train=True,
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
    )
    assert len(hci_train) > 0
    assert len(hci_test) > 0


def test_hci_prepare_again(tmp_path):
    hci = HCI(
        root=tmp_path,
        train=True,
    )
    prepared_first = hci._prepare_dataset()
    prepared_second = hci._prepare_dataset()
    assert prepared_first is True
    assert prepared_second is False


def test_hci_categories(tmp_path):
    hci_train = HCI(
        root=tmp_path,
        train=True,
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
    )
    train_categories = set()
    for _, label in hci_train:
        train_categories.add(label)
    assert train_categories == {0, 1, 2, 3, 4}

    test_categories = set()
    for _, label in hci_test:
        test_categories.add(label)
    assert test_categories == {0, 1, 2, 3, 4}


def test_hci_image_size(tmp_path):
    hci_train = HCI(
        root=tmp_path,
        train=True,
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
    )
    for img, _ in hci_train:
        assert img.size == (224, 224)
    for img, _ in hci_test:
        assert img.size == (224, 224)


def test_hci_md5_verification(tmp_path):
    hci_train = HCI(
        root=tmp_path,
        train=True,
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
    )
    # Modify one file to test MD5 verification
    sample_img_path = hci_train.root / "0" / next(iter(hci_train.samples))[0]
    with open(sample_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()

    # HCI test should also fail since it checks the whole dataset
    assert not hci_test._verify_md5sums()


def test_hci_prepare_after_corruption(tmp_path):
    hci_train = HCI(
        root=tmp_path,
        train=True,
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
    )
    # Modify one file to test re-preparation
    sample_train_img_path = hci_train.root / "0" / next(iter(hci_train.samples))[0]
    with open(sample_train_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()
    assert not hci_test._verify_md5sums()

    sample_test_img_path = hci_test.root / "0" / next(iter(hci_test.samples))[0]
    with open(sample_test_img_path, "rb+") as f:
        content = f.read()
        f.seek(0)
        f.write(b"corrupted_data" + content)
    assert not hci_train._verify_md5sums()
    assert not hci_test._verify_md5sums()

    # Re-prepare dataset
    assert hci_train._prepare_dataset()
    assert hci_train._verify_md5sums()
    assert hci_test._prepare_dataset()
    assert hci_test._verify_md5sums()


def test_hci_load_data_with_dataloader(tmp_path):
    from torch.utils.data import DataLoader

    hci_train = HCI(
        root=tmp_path,
        train=True,
        transform=ToTensor(),
    )
    hci_test = HCI(
        root=tmp_path,
        train=False,
        transform=ToTensor(),
    )

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
