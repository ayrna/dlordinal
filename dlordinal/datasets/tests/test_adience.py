import shutil
import tarfile
from pathlib import Path
from typing import Optional

import numpy as np
import PIL.Image as Image
import pytest
import torch
from torchvision.datasets.utils import check_integrity
from torchvision.transforms import ToTensor

from dlordinal.datasets import Adience
from dlordinal.datasets.adience import _image_path_from_row, _track_progress

from .utils.adience import generate_fake_adience, get_adience_md5sums


@pytest.fixture
def adience_root(tmp_path):
    path = generate_fake_adience(tmp_path / "adience")
    return path


@pytest.fixture
def patched_adience(adience_root, monkeypatch):
    md5sums = get_adience_md5sums(adience_root)

    ALIGNED_URL = (
        str(adience_root / "aligned.tar.gz"),
        md5sums["aligned"],
    )
    FOLDS_URLS = tuple(
        [
            (str(adience_root / "folds" / f"fold_{i}_data.txt"), md5sums[f"fold{i}"])
            for i in range(5)
        ]
    )

    monkeypatch.setattr(Adience, "ALIGNED_URL", ALIGNED_URL)
    monkeypatch.setattr(Adience, "FOLDS_URLS", FOLDS_URLS)

    def fake_download_file(
        self,
        url: str,  # it is a local path
        output_path: Path,
        username: str,
        password: str,
        md5: Optional[str] = None,
    ):
        shutil.copy(url, output_path)

        if md5 is not None and not check_integrity(str(output_path), md5):
            raise ValueError(
                f"Downloaded file {output_path} has an invalid MD5 checksum."
            )

    monkeypatch.setattr(Adience, "_download_file", fake_download_file)

    def _create(
        train,
        test_size,
        root=adience_root,
        download=False,
        username=None,
        password=None,
    ):
        return Adience(
            root=root,
            train=train,
            test_size=test_size,
            verbose=False,
            download=download,
            username=username,
            password=password,
        )

    return _create


@pytest.fixture
def adience_train(patched_adience):
    return patched_adience(train=True, test_size=0.2)


@pytest.fixture
def adience_test(patched_adience):
    return patched_adience(train=False, test_size=0.2)


@pytest.mark.no_gpu_ci
@pytest.mark.parametrize("adience", ["adience_train", "adience_test"])
def test_adience_init(request, adience):
    dataset: Adience = request.getfixturevalue(adience)

    assert dataset._check_if_extracted()
    assert dataset._check_if_transformed()
    assert dataset._check_if_partitioned()
    assert dataset._check_input_files()


@pytest.mark.no_gpu_ci
def test_download(patched_adience, tmp_path):
    """Test that verifies download works correctly."""
    # Use patched_adience with an empty download directory
    download_root = tmp_path / "download_root"
    download_root.mkdir(parents=True, exist_ok=True)

    dataset = patched_adience(
        train=True,
        test_size=0.2,
        root=download_root,
        download=True,
        username="fake_user",
        password="fake_pass",
    )

    # Validate that files were downloaded correctly
    assert dataset._check_input_files()
    assert dataset._check_if_extracted()
    assert dataset._check_if_transformed()
    assert dataset._check_if_partitioned()

    # Validate that dataset has data
    assert len(dataset) > 0
    assert len(dataset.data) == len(dataset.targets)
    assert len(dataset.classes) > 0


def test_download_without_credentials(patched_adience, tmp_path):
    download_root = tmp_path / "download_root"
    download_root.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        patched_adience(
            train=True,
            test_size=0.2,
            root=download_root,
            download=True,
        )


@pytest.mark.parametrize(
    "corrupted_file_path",
    [
        "aligned.tar.gz",
        "fold_0_data.txt",
        "fold_1_data.txt",
        "fold_2_data.txt",
        "fold_3_data.txt",
        "fold_4_data.txt",
    ],
)
def test_download_corrupted_file(
    patched_adience, tmp_path, monkeypatch, corrupted_file_path
):
    download_root = tmp_path / "download_root"
    download_root.mkdir(parents=True, exist_ok=True)

    def corrupt_download_file(
        self,
        url: str,  # it is a local path
        output_path: Path,
        username: str,
        password: str,
        md5: Optional[str] = None,
    ):
        shutil.copy(url, output_path)

        if output_path.name == corrupted_file_path:
            with open(output_path, "ab") as f:
                f.write(b"corrupted data")

        if md5 is not None and not check_integrity(str(output_path), md5):
            raise ValueError(
                f"Downloaded file {output_path} has an invalid MD5 checksum."
            )

    monkeypatch.setattr(Adience, "_download_file", corrupt_download_file)

    with pytest.raises(ValueError):
        patched_adience(
            train=True,
            test_size=0.2,
            root=download_root,
            download=True,
            username="fake_user",
            password="fake_pass",
        )


@pytest.mark.no_gpu_ci
def test_adience_len(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert len(adience) == len(adience.targets)
        assert len(adience) == len(adience.data)

        adience.targets.append(0)

        with pytest.raises(ValueError):
            len(adience)


@pytest.mark.no_gpu_ci
def test_adience_getitem(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        for i in range(len(adience)):
            assert isinstance(adience[i][0], Image.Image)
            assert isinstance(adience[i][1], int)
            assert adience[i][1] == adience.targets[i]
            assert np.array(adience[i][0]).ndim == 3

        adience.transform = ToTensor()

        for i in range(len(adience)):
            assert isinstance(adience[i][0], torch.Tensor)
            assert isinstance(adience[i][1], int)
            assert adience[i][1] == adience.targets[i]
            assert len(adience[i][0].shape) == 3

        adience.target_transform = lambda target: np.array(target)
        for i in range(len(adience)):
            assert isinstance(adience[i][0], torch.Tensor)
            assert isinstance(adience[i][1], np.ndarray)
            assert np.array_equal(adience[i][1], adience.targets[i])


@pytest.mark.no_gpu_ci
def test_assign_range_integers(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("1") == 0
        assert adience._assign_range("5") == 1
        assert adience._assign_range("10") == 2
        assert adience._assign_range("18") == 3
        assert adience._assign_range("30") == 4
        assert adience._assign_range("41") == 5
        assert adience._assign_range("50") == 6
        assert adience._assign_range("70") == 7
        assert adience._assign_range("101") is None


@pytest.mark.no_gpu_ci
def test_assing_range_tuples(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("(0, 2)") == 0
        assert adience._assign_range("(4, 6)") == 1
        assert adience._assign_range("(8, 13)") == 2
        assert adience._assign_range("(15, 20)") == 3
        assert adience._assign_range("(25, 32)") == 4
        assert adience._assign_range("(38, 43)") == 5
        assert adience._assign_range("(48, 53)") == 6
        assert adience._assign_range("(60, 100)") == 7


@pytest.mark.no_gpu_ci
def test_assign_range_none(adience_train, adience_test):
    for adience in [adience_train, adience_test]:
        assert adience._assign_range("None") is None


@pytest.mark.no_gpu_ci
def test_adience_train_test(adience_train, adience_test):
    assert len(adience_train) != len(adience_test)

    train_labels = [label for _, label in adience_train]
    test_labels = [label for _, label in adience_test]

    assert train_labels != test_labels


@pytest.mark.no_gpu_ci
def test_image_path_from_row():
    row = {"user_id": "123", "face_id": "456", "original_image": "image.jpg"}
    path = _image_path_from_row(row)
    assert path == "123/landmark_aligned_face.456.image.jpg"


@pytest.mark.no_gpu_ci
def test_track_progress(adience_root):
    """Test that verifies _track_progress correctly iterates tar members."""
    tar_file_path = adience_root / "aligned.tar.gz"

    # Read the existing tar.gz with real data
    with tarfile.open(tar_file_path, "r:gz") as file:
        member_count = 0
        for member in _track_progress(file):
            assert isinstance(member, tarfile.TarInfo)
            member_count += 1

        # Must have at least one member (aligned folder or images inside)
        assert member_count > 0


@pytest.mark.no_gpu_ci
def test_adience_classes(adience_train, adience_test):
    assert adience_train.classes == adience_test.classes
    assert adience_train.classes == np.unique(adience_train.targets).tolist()
    assert adience_test.classes == np.unique(adience_test.targets).tolist()


@pytest.mark.no_gpu_ci
def test_check_input_files(adience_train, adience_test, tmp_path):
    assert adience_train._check_input_files()
    assert adience_test._check_input_files()

    adience_train._data_file_path.unlink()
    assert not adience_train._check_input_files()

    (adience_test._folds_path / "fold_0_data.txt").unlink()
    assert not adience_test._check_input_files()

    with pytest.raises(FileNotFoundError):
        Adience(root=Path(tmp_path) / "test", train=True)


@pytest.mark.no_gpu_ci
def test_check_if_extracted(adience_train, adience_test):
    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    shutil.rmtree(adience_train._images_path)
    assert not adience_train._check_if_extracted()
    assert not adience_test._check_if_extracted()


@pytest.mark.no_gpu_ci
def test_check_if_transformed(adience_train, adience_test):
    assert adience_train._check_if_transformed()
    assert adience_test._check_if_transformed()

    shutil.rmtree(adience_train._transformed_images_path)
    assert not adience_train._check_if_transformed()
    assert not adience_test._check_if_transformed()


@pytest.mark.no_gpu_ci
def test_check_if_partitioned(adience_train, adience_test):
    assert adience_train._check_if_partitioned()
    assert adience_test._check_if_partitioned()

    split_file = (
        adience_train.root
        / "cache"
        / f"splits_{adience_train._cache_key}"
        / "train.csv"
    )
    split_file.unlink()
    assert not adience_train._check_if_partitioned()
    split_file = (
        adience_test.root / "cache" / f"splits_{adience_train._cache_key}" / "test.csv"
    )
    split_file.unlink()
    assert not adience_test._check_if_partitioned()


@pytest.mark.no_gpu_ci
def test_extract_data(adience_train, adience_test):
    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    adience_train._extract_data()
    adience_test._extract_data()

    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()

    shutil.rmtree(adience_train._images_path)

    assert not adience_train._check_if_extracted()
    assert not adience_test._check_if_extracted()

    adience_train._extract_data()

    assert adience_train._check_if_extracted()
    assert adience_test._check_if_extracted()


@pytest.mark.no_gpu_ci
def test_assign_range_malformed_and_edge_cases(adience_train):
    # Malformed and edge cases
    malformed = [
        "",
        "10abc",
        "(1,2,3)",
        "  ",
        None,
        "(0,2",
        "0,2)",
        "(a,b)",
        "(0,2,4)",
        "(0,2,)",
        "(,2)",
        "(0,)",
        "(0,2)extra",
    ]
    for val in malformed:
        assert adience_train._assign_range(val) is None, f"Failed for input: {val}"

    # Valid edge cases
    assert adience_train._assign_range("(0, 2)") == 0
    assert adience_train._assign_range("    (0, 2)") == 0
    assert adience_train._assign_range("(0, 2)   ") == 0
    assert adience_train._assign_range("(4,6)") == 1
    assert adience_train._assign_range("(4,   6)") == 1
    assert adience_train._assign_range("(4  ,6)") == 1
    assert adience_train._assign_range("8") == 2
    assert adience_train._assign_range("100") == 7
    assert adience_train._assign_range("101") is None


@pytest.mark.no_gpu_ci
def test_change_test_size(adience_root, patched_adience):
    adience1 = patched_adience(train=True, test_size=0.2, root=adience_root)
    adience2 = patched_adience(train=True, test_size=0.5, root=adience_root)

    assert len(adience1) != len(adience2)
    assert len(adience1.targets) != len(adience2.targets)

    assert adience1._cache_key != adience2._cache_key

    cache_dir = adience_root / "cache"
    split_dirs = list(cache_dir.glob("splits_*"))
    assert len(split_dirs) >= 2

    train1 = (cache_dir / f"splits_{adience1._cache_key}" / "train.csv").read_bytes()
    train2 = (cache_dir / f"splits_{adience2._cache_key}" / "train.csv").read_bytes()
    assert train1 != train2
