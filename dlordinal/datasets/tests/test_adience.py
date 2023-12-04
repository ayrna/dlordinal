import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from PIL import Image

from ..adience import Adience

temp_dir = None


@pytest.fixture
def adience_instance():
    global temp_dir
    temp_dir = tempfile.TemporaryDirectory(prefix="tmp_", suffix="_adience", dir="./")
    temp_path = Path(temp_dir.name)

    folds_path = temp_path / "folds"
    images_path = temp_path / "images"
    transformed_images_path = temp_path / "transformed_images"
    partition_path = temp_path / "partitions"

    folds_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    transformed_images_path.mkdir(parents=True, exist_ok=True)
    partition_path.mkdir(parents=True, exist_ok=True)

    (images_path / "30601258@N03").mkdir(parents=True, exist_ok=True)
    (images_path / "7153718@N04").mkdir(parents=True, exist_ok=True)
    (images_path / "7285955@N06").mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (816, 816)).save(
        images_path
        / "30601258@N03"
        / "landmark_aligned_face.2049.7486613949_909254ccf9_o.jpg"
    )

    Image.new("RGB", (816, 816)).save(
        images_path
        / "30601258@N03"
        / "landmark_aligned_face.1.9904044896_cb797f78d2_o.jpg"
    )

    Image.new("RGB", (816, 816)).save(
        images_path
        / "7153718@N04"
        / "landmark_aligned_face.2050.9486613949_909254ccf9_o.jpg"
    )

    Image.new("RGB", (816, 816)).save(
        images_path
        / "7153718@N04"
        / "landmark_aligned_face.2282.11597935265_29bcdfa4a5_o.jpg"
    )

    Image.new("RGB", (816, 816)).save(
        images_path
        / "7285955@N06"
        / "landmark_aligned_face.2052.10524078416_6a401de320_o.jpg"
    )

    Image.new("RGB", (816, 816)).save(
        images_path
        / "7285955@N06"
        / "landmark_aligned_face.2050.6486613949_909254ccf9_o.jpg"
    )

    with tarfile.open(temp_path / "fake_data.tar.gz", "w:gz") as tar:
        pass

    list_folds_files = [
        "fold_0_data.txt",
        "fold_1_data.txt",
        "fold_2_data.txt",
        "fold_3_data.txt",
        "fold_4_data.txt",
    ]

    tabulador = "\t"

    for file in list_folds_files:
        with open(folds_path / file, "w") as f:
            f.write(
                "user_id"
                + tabulador
                + "original_image"
                + tabulador
                + "face_id"
                + tabulador
                + "age"
                + tabulador
                + "gender"
                + tabulador
                + "x"
                + tabulador
                + "y"
                + tabulador
                + "dx"
                + tabulador
                + "dy"
                + tabulador
                + "tilt_ang"
                + tabulador
                + "fiducial_yaw_angle"
                + tabulador
                + "fiducial_score"
                + "\n"
            )
            f.write(
                "30601258@N03"
                + tabulador
                + "10399646885_67c7d20df9_o.jpg"
                + tabulador
                + "1"
                + tabulador
                + "(25, 32)"
                + tabulador
                + "f"
                + tabulador
                + "0"
                + tabulador
                + "414"
                + tabulador
                + "1086"
                + tabulador
                + "1383"
                + tabulador
                + "-115"
                + tabulador
                + "30"
                + tabulador
                + "17"
                + "\n"
            )
            f.write(
                "7153718@N04"
                + tabulador
                + "10424815813_e94629b1ec_o.jpg"
                + tabulador
                + "2"
                + tabulador
                + "(25, 32)"
                + tabulador
                + "m"
                + tabulador
                + "301"
                + tabulador
                + "105"
                + tabulador
                + "640"
                + tabulador
                + "641"
                + tabulador
                + "0"
                + tabulador
                + "0"
                + tabulador
                + "94"
                + "\n"
            )

    adience_instance = Adience(
        extract_file_path=temp_path / "fake_data.tar.gz",
        folds_path=folds_path,
        images_path=images_path,
        transformed_images_path=transformed_images_path,
        partition_path=partition_path,
        number_partitions=20,
        ranges=[
            (0, 2),
            (4, 6),
            (8, 13),
            (15, 20),
            (25, 32),
            (38, 43),
            (48, 53),
            (60, 100),
        ],
        test_size=0.4,
        extract=True,
        transfrom=True,
    )

    return adience_instance


def test_adience_instance(adience_instance):
    assert isinstance(adience_instance, Adience)


def test_assign_range_integers(adience_instance):
    assert adience_instance.assign_range("1") == 0
    assert adience_instance.assign_range("5") == 1
    assert adience_instance.assign_range("10") == 2
    assert adience_instance.assign_range("18") == 3
    assert adience_instance.assign_range("30") == 4
    assert adience_instance.assign_range("41") == 5
    assert adience_instance.assign_range("50") == 6
    assert adience_instance.assign_range("70") == 7
    assert adience_instance.assign_range("101") is None


def test_assing_range_tuples(adience_instance):
    assert adience_instance.assign_range("(0, 2)") == 0
    assert adience_instance.assign_range("(4, 6)") == 1
    assert adience_instance.assign_range("(8, 13)") == 2
    assert adience_instance.assign_range("(15, 20)") == 3
    assert adience_instance.assign_range("(25, 32)") == 4
    assert adience_instance.assign_range("(38, 43)") == 5
    assert adience_instance.assign_range("(48, 53)") == 6
    assert adience_instance.assign_range("(60, 100)") == 7


def test_assign_range_none(adience_instance):
    assert adience_instance.assign_range("None") is None


def test_image_path_from_row(adience_instance):
    row = {"user_id": "123", "face_id": "456", "original_image": "image.jpg"}
    path = adience_instance.image_path_from_row(row)
    assert path == "123/landmark_aligned_face.456.image.jpg"


def test_track_progress(adience_instance):
    tar_file_path = "fake.tar.gz"

    try:
        with tarfile.open(tar_file_path, "w:gz") as file:
            for member in adience_instance.track_progress(file):
                assert isinstance(member, tarfile.TarInfo)

    finally:
        os.remove(tar_file_path)


def test_process_and_split(adience_instance, monkeypatch):
    global temp_dir

    assert isinstance(temp_dir, tempfile.TemporaryDirectory)

    df1 = pd.DataFrame.from_dict(
        {
            "user_id": ["30601258@N03", "30601258@N03"],
            "original_image": [
                "7486613949_909254ccf9_o.jpg",
                "9904044896_cb797f78d2_o.jpg",
            ],
            "face_id": ["2049", "1"],
            "age": ["(25, 32)", "(25, 32)"],
        }
    )

    df2 = pd.DataFrame.from_dict(
        {
            "user_id": ["7153718@N04", "7153718@N04"],
            "original_image": [
                "9486613949_909254ccf9_o.jpg",
                "11597935265_29bcdfa4a5_o.jpg",
            ],
            "face_id": ["2050", "2282"],
            "age": ["(8, 13)", "(8, 13)"],
        }
    )

    df3 = pd.DataFrame.from_dict(
        {
            "user_id": ["7285955@N06", "7285955@N06"],
            "original_image": [
                "10524078416_6a401de320_o.jpg",
                "6486613949_909254ccf9_o.jpg",
            ],
            "face_id": ["2052", "2050"],
            "age": ["(60, 100)", "(60, 100)"],
        }
    )

    folds = [df1, df2, df3]

    mock_open = Mock(side_effect=lambda _: Image.new("RGB", (128, 128)))
    mock_symlink_to = Mock()

    monkeypatch.setattr(adience_instance, "_check_if_transformed", lambda: False)
    monkeypatch.setattr("PIL.Image.open", mock_open)
    monkeypatch.setattr("pathlib.Path.symlink_to", mock_symlink_to)
    monkeypatch.setattr("sklearn.model_selection.StratifiedShuffleSplit", None)

    adience_instance.transformed_images_path = (
        Path(temp_dir.name) / "transformed_images"
    )

    adience_instance.process_and_split(folds)

    files_transformed = (Path(temp_dir.name) / "transformed_images").iterdir()
    files_partition = list((Path(temp_dir.name) / "partitions").iterdir())

    assert mock_open.call_count == 6
    assert len(list(files_transformed)) == 3

    for dir in files_transformed:
        dir.iterdir()
        assert len(list(dir.iterdir())) == 2

        for file in dir.iterdir():
            assert file.is_file()
            assert file.suffix == ".jpg"

    assert len(files_partition) == adience_instance.number_partitions
