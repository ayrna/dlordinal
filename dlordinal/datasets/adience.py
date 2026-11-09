import base64
import hashlib
import json
import re
import tarfile
import warnings
from contextlib import closing
from pathlib import Path
from typing import Callable, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from ._utils import (
    marker_matches,
    n_splits_from_test_size,
    remove_marker,
    stratified_group_split,
    write_marker,
)


class Adience(VisionDataset):
    """
    PyTorch dataset for the Adience age classification benchmark
    :footcite:t:`eidinger2014age`.

    The Adience dataset contains unfiltered face images collected from Flickr
    albums and is commonly used for age and gender classification benchmarks.

    Parameters
    ----------
    root : Union[str, Path]
        Root directory where the dataset will be stored.

        If ``download=False`` (default), the following files are expected
        to already exist inside the ``adience`` directory:

        1. ``aligned.tar.gz``:
        tar.gz archive containing the aligned face images.

        2. ``folds``:
        directory containing the official Adience fold files:
        ``fold_0_data.txt`` through ``fold_4_data.txt``.

        If ``download=True``, these files are downloaded automatically
        from the official Adience website.
    ranges : list, optional
        List of age ranges to use, by default [(0, 2), (4, 6), (8, 13),
        (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)].
    test_size : float, optional, default = 0.2
        Approximate fraction of the images used for the test partition. The
        partition is subject-disjoint: all the images of a subject (``user_id``)
        belong to the same partition, so that no identity is shared between train
        and test. It is obtained as one fold of a
        :class:`sklearn.model_selection.StratifiedGroupKFold` with
        ``n_splits = max(2, round(1 / test_size))`` folds, so the actual test
        fraction is approximately ``1 / n_splits``. The first fold whose train and
        test partitions both contain every class is used.
    transform : Callable, optional
        A callable that takes in an PIL image and returns a transformed version.
    target_transform : Callable, optional
        A callable that takes in the target and transforms it.
    verbose : bool, optional, default = False
        Whether to print progress messages.
    download : bool, optional, default = False
        Whether to download the dataset automatically.

        Downloading requires valid username and password credentials
        provided by the Adience dataset authors.
    username : str, optional
        Username to download the dataset. If not provided, the dataset will not be
        downloaded and the files are expected to be already present in the root
        directory.
    password : str, optional
        Password to download the dataset. If not provided, the dataset will not be
        downloaded and the files are expected to be already present in the root
        directory.

    Attributes
    ----------
    root : Path
        Root directory where the Adience dataset is stored.
    train : bool
        Whether to use the training or test partition.
    transform : Callable
        A callable that takes in an PIL image and returns a transformed version.
    target_transform : Callable
        A callable that takes in the target and transforms it.
    verbose : bool
        Whether to print progress messages.
    data : list
        List of image paths.
    targets : list
        Contains the target of each sampel contained in the dataset.
    classes : list
        Unique classes in the dataset.
    download : bool
        Whether to download the dataset if it is not already present in the root
        directory. If False, the files are expected to be already present in the root
        directory.

    Notes
    -----
    The extracted images, the resized images and the train/test split are cached
    inside the dataset directory. Each step writes a completion marker with its
    metadata once it has finished, and is redone when the marker is missing or
    does not match the current configuration. The MD5 checksum of
    ``aligned.tar.gz`` is only verified when the archive has to be downloaded or
    extracted.
    """

    ALIGNED_URL = (
        "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/aligned.tar.gz",
        "bf8336d576433f0143828925eadbe23f",
    )
    FOLDS_URLS = [
        (
            "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_0_data.txt",
            "dda2131b5a4934a67f0acfda8b50a65b",
        ),
        (
            "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_1_data.txt",
            "bb558fff6aba953b5b05403d74dfd8a8",
        ),
        (
            "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_2_data.txt",
            "a156e37bf4292a61ee5e11a06cfc6c5f",
        ),
        (
            "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_3_data.txt",
            "7c9f7dab8fb034affe8a08e97da24266",
        ),
        (
            "http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/fold_4_data.txt",
            "68ebc064a70274551a565fdd5235f0cc",
        ),
    ]
    DEFAULT_RANGES = (
        (0, 2),
        (4, 6),
        (8, 13),
        (15, 20),
        (25, 32),
        (38, 43),
        (48, 53),
        (60, 100),
    )
    # Version of the resized images. Bump it when the resize policy changes.
    _TRANSFORM_VERSION = 1
    _TRANSFORM_HEIGHT = 128
    _SPLIT_SEED = 0

    root: Path
    train: bool
    _ranges: list
    _test_size: float
    transform: Optional[Callable]
    target_transform: Optional[Callable]
    verbose: bool
    data: list
    targets: list
    classes: list
    download: bool

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        ranges: Union[list, tuple] = DEFAULT_RANGES,
        test_size: float = 0.2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        verbose: bool = False,
        download: bool = False,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        super().__init__(
            root=str(root),
            transform=transform,
            target_transform=target_transform,
        )

        self.root = Path(root)
        self.train = train
        self._ranges = tuple([tuple(r) for r in ranges])
        if self._ranges != self.DEFAULT_RANGES:
            warnings.warn(
                "Custom age ranges differ from the official Adience benchmark "
                "and may make results non-comparable.",
                UserWarning,
            )

        self._test_size = test_size
        self._n_splits = n_splits_from_test_size(test_size)
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.download = download

        # Version 2.0: subject-disjoint split (StratifiedGroupKFold by user_id)
        self._version = "2.0"
        self._config = self._get_config_dict()
        self._cache_key = self._make_cache_key(self._config)

        self.data = []
        self.targets = []
        self.classes = []

        # TODO: remove in 3.0.0
        if (self.root / "adience").exists():
            warnings.warn(
                (
                    "Workspace-style root is deprecated and will be removed in v3.0.0. "
                    "Pass root='.../adience' directly instead. "
                    "Current behavior will be removed."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            self._root_adience = self.root / "adience"
        else:
            self._root_adience = self.root

        self._data_file_path = self._root_adience / "aligned.tar.gz"
        self._folds_path = self._root_adience / "folds"
        self._images_path = self._root_adience / "aligned"
        self._transformed_images_path = self._root_adience / "transformed"
        self._extracted_marker = self._images_path / ".dlordinal_extracted.json"
        self._transformed_marker = (
            self._transformed_images_path / ".dlordinal_transformed.json"
        )
        self._archive_verified = False

        if self.download and (username is None or password is None):
            raise ValueError("username and password are required when download=True")

        if self.download and username is not None and password is not None:
            self._download(username, password)

        if not self._check_folds():
            raise FileNotFoundError(self._missing_files_message())

        self.folds_ = [
            pd.read_csv(self._folds_path / f"fold_{f}_data.txt", sep="\t")
            for f in range(5)
        ]

        # The archive is only needed (and verified) when the resized images are not
        # available yet.
        if not self._check_if_transformed():
            self._extract_data()
            self._build_transformed()
        self._df = self._build_dataframe(self.folds_)
        self._build_splits()

    @staticmethod
    def _missing_files_message() -> str:
        return (
            "Some input files are missing. Please, check the documentation of the"
            " root parameter to see the expected directory structure or provide the"
            " username and password to download the files automatically."
        )

    def _check_archive(self) -> bool:
        """
        Check if the images archive exists and has the expected MD5 checksum.
        The (expensive) checksum is computed at most once per instance.
        """
        if not self._data_file_path.exists():
            self._archive_verified = False
            return False
        if not self._archive_verified:
            self._archive_verified = check_integrity(
                str(self._data_file_path), self.ALIGNED_URL[1]
            )
        return self._archive_verified

    def _check_folds(self) -> bool:
        """
        Check if the fold files exist and have the expected MD5 checksums.
        """
        for i in range(5):
            fold_path = self._folds_path / f"fold_{i}_data.txt"
            if not fold_path.exists() or not check_integrity(
                str(fold_path), self.FOLDS_URLS[i][1]
            ):
                return False
        return True

    def _check_input_files(self) -> bool:
        """
        Check if the input files (archive and fold files) are present and valid.
        """
        return self._check_archive() and self._check_folds()

    def _extraction_metadata(self) -> dict:
        return {"archive_md5": self.ALIGNED_URL[1]}

    def _transform_metadata(self) -> dict:
        return {
            "archive_md5": self.ALIGNED_URL[1],
            "height": self._TRANSFORM_HEIGHT,
            "version": self._TRANSFORM_VERSION,
        }

    def _check_if_extracted(self) -> bool:
        """
        Check if the tar.gz file has been completely extracted.
        """
        return marker_matches(self._extracted_marker, self._extraction_metadata())

    def _check_if_transformed(self) -> bool:
        """
        Check if all the images have been transformed.
        """
        return marker_matches(self._transformed_marker, self._transform_metadata())

    def _check_if_partitioned(self) -> bool:
        """
        Check if a valid cached split exists for the current configuration.
        """

        split_dir = self._root_adience / "cache" / f"splits_{self._cache_key}"
        config_path = split_dir / "config.json"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"

        return (
            train_path.exists()
            and test_path.exists()
            and marker_matches(config_path, self._config)
        )

    def _download_file(
        self,
        url: str,
        output_path: Path,
        username: str,
        password: str,
        md5: Optional[str] = None,
    ):
        credentials = f"{username}:{password}".encode("utf-8")
        encoded_credentials = base64.b64encode(credentials).decode("utf-8")

        req = Request(url)
        req.add_header("Authorization", f"Basic {encoded_credentials}")

        try:
            with closing(urlopen(req, timeout=300)) as response:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size is not None else None

                with (
                    open(output_path, "wb") as f,
                    tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=output_path.name,
                        disable=not self.verbose,
                    ) as pbar,
                ):
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        except HTTPError as e:
            raise RuntimeError(f"HTTP error while downloading {url}: {e}") from e
        except URLError as e:
            raise RuntimeError(f"URL error while downloading {url}: {e}") from e

        if md5 is not None and not check_integrity(str(output_path), md5):
            raise ValueError(
                f"Downloaded file {output_path} has an invalid MD5 checksum."
            )

    def _download(self, username: str, password: str, force: bool = False):
        self._root_adience.mkdir(exist_ok=True, parents=True)

        if username is None or password is None:
            raise ValueError(
                "Username and password must be provided to download the dataset."
            )

        # The archive is not needed once its content has been extracted or resized
        archive_needed = force or not (
            self._check_if_transformed() or self._check_if_extracted()
        )

        if archive_needed and (force or not self._check_archive()):
            aligned_name = "aligned.tar.gz"

            if self.verbose:
                print(f"{aligned_name} is missing or corrupted. Downloading...")

            self._download_file(
                url=self.ALIGNED_URL[0],
                output_path=self._root_adience / aligned_name,
                username=username,
                password=password,
                md5=self.ALIGNED_URL[1],
            )
            # _download_file raises if the checksum of the downloaded file is wrong
            self._archive_verified = True

        self._folds_path.mkdir(exist_ok=True, parents=True)

        for i, (url, md5) in enumerate(self.FOLDS_URLS):
            filename = f"fold_{i}_data.txt"

            if force or (
                not (self._folds_path / filename).exists()
                or not check_integrity(str(self._folds_path / filename), md5)
            ):

                if self.verbose:
                    print(f"{filename} is missing or corrupted. Downloading...")

                self._download_file(
                    url=url,
                    output_path=self._folds_path / filename,
                    username=username,
                    password=password,
                    md5=md5,
                )

    def _extract_data(self):
        """
        Extract the data tar.gz file.
        """
        if self._check_if_extracted():
            if self.verbose:
                print("File already extracted.")
            return

        if not self._check_archive():
            raise FileNotFoundError(self._missing_files_message())

        if self.verbose:
            print("Extracting file...")

        remove_marker(self._extracted_marker)
        with tarfile.open(self._data_file_path, "r:gz") as file:
            path = self._data_file_path.parent
            path.mkdir(exist_ok=True, parents=True)
            kwargs = {}
            if hasattr(tarfile, "data_filter"):
                # Reject absolute paths, path traversal and special files
                kwargs["filter"] = "data"
            file.extractall(path, members=_track_progress(file, self.verbose), **kwargs)

        write_marker(self._extracted_marker, self._extraction_metadata())

    def _build_transformed(self) -> None:
        """
        Create a transformed (resized) version of all images.

        This step is independent of:
        - train/test split
        - age ranges
        - dataset partitioning

        It depends only on:
        - raw images
        - resize policy (fixed here: 128px height)
        """

        if self._check_if_transformed():
            if self.verbose:
                print("Transformed images already exist.")
            return

        remove_marker(self._transformed_marker)
        self._transformed_images_path.mkdir(exist_ok=True, parents=True)

        if self.verbose:
            print("Creating transformed images...")

        image_paths = sorted(self._images_path.rglob("*.jpg"))
        height = self._TRANSFORM_HEIGHT

        for src_image in tqdm(
            image_paths,
            total=len(image_paths),
            disable=not self.verbose,
            desc="transforming",
        ):
            # Preserve relative structure. Existing files are overwritten because
            # they may come from an interrupted run.
            rel_path = src_image.relative_to(self._images_path)
            dst_image = self._transformed_images_path / rel_path
            dst_image.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src_image) as img:
                img = img.convert("RGB")

                width_percent = height / float(img.size[1])
                new_width = int(img.size[0] * width_percent)

                resized = img.resize(
                    (new_width, height),
                    Image.Resampling.BILINEAR,
                )

                resized.save(dst_image)

        write_marker(self._transformed_marker, self._transform_metadata())

    def _build_dataframe(self, folds: list) -> pd.DataFrame:
        """
        Build the internal dataframe from raw fold files.

        This includes:
        - Filtering invalid age entries
        - Mapping ages to class ranges
        - Constructing relative image paths
        - Merging all folds into a single dataframe
        """

        fold_dfs = []

        for f, fold in enumerate(folds):
            valid = fold["age"].notna()
            fold = fold.loc[valid]

            fold = fold.assign(age=fold["age"].map(self._assign_range))
            fold = fold.dropna(subset=["age"])
            fold = fold.assign(age=fold["age"].astype(int))

            df = pd.DataFrame(
                {
                    "path": fold.apply(_image_path_from_row, axis="columns"),
                    "age": fold["age"],
                    "user_id": fold["user_id"].astype(str),
                }
            )

            fold_dfs.append(df)

        return pd.concat(fold_dfs, ignore_index=True)

    def _build_splits(self):
        """
        Create subject-disjoint train/test splits and persist them as CSV files.

        All the images of a subject (``user_id``) are assigned to the same
        partition. The test partition is one fold of a
        :class:`sklearn.model_selection.StratifiedGroupKFold` with
        ``max(2, round(1 / test_size))`` folds, stratified by age range (see
        :func:`dlordinal.datasets._utils.stratified_group_split`).

        The split is cached using a hash of the dataset configuration
        (e.g. test_size, ranges, split method).
        """

        split_dir = self._root_adience / "cache" / f"splits_{self._cache_key}"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"
        config_path = split_dir / "config.json"
        split_path = train_path if self.train else test_path

        if self._check_if_partitioned():
            if self.verbose:
                print("Splits already exist. Loading from cache.")
            self._load_split_from_csv(split_path)
            return

        split_dir.mkdir(parents=True, exist_ok=True)
        # config.json marks a complete split: remove it while the split is rebuilt
        remove_marker(config_path)

        train_idx, test_idx = stratified_group_split(
            self._df["age"],
            groups=self._df["user_id"],
            test_size=self._test_size,
            seed=self._SPLIT_SEED,
        )

        columns = ["path", "age", "user_id"]
        train_df = self._df.iloc[train_idx][columns]
        test_df = self._df.iloc[test_idx][columns]

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        write_marker(config_path, self._config)

        self._load_split_from_csv(split_path)

    def _load_split_from_csv(self, split_path: Path):
        df = pd.read_csv(split_path, dtype={"path": str, "user_id": str})

        self.data = [str(self._transformed_images_path / p) for p in df["path"]]
        self.targets = [int(t) for t in df["age"]]
        self.classes = np.unique(self.targets).tolist()

    def _get_config_dict(self):
        return {
            "group": "user_id",
            "n_splits": self._n_splits,
            "ranges": self._ranges,
            "seed": self._SPLIT_SEED,
            "split_method": "StratifiedGroupKFold",
            "test_size": self._test_size,
            "version": self._version,
        }

    def _make_cache_key(self, config):
        s = json.dumps(config, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _assign_range(self, age: str):
        """
        Assign an age range to an age.

        Parameters
        ----------
        age : str
            Age to assign a range to.
        """
        if age is None:
            return None
        if not isinstance(age, str):
            age = str(age)
        age = age.strip()

        m = re.match(r"^\((\d+) *, *(\d+)\)$", age)
        if m:
            age = (int(m.group(1)), int(m.group(2)))
        else:
            m = re.match(r"^(\d+)$", age)
            if m:
                age = int(m.group(0))
            else:
                return None

        if age in self._ranges:
            return self._ranges.index(age)

        if isinstance(age, tuple):
            age_minimum, age_maximum = age
            for i, (range_minimum, range_maximum) in enumerate(self._ranges):
                if (age_minimum >= range_minimum) and (age_maximum <= range_maximum):
                    return i
            return None

        if isinstance(age, int):
            for i, (range_minimum, range_maximum) in enumerate(self._ranges):
                if (age >= range_minimum) and (age <= range_maximum):
                    return i
            return None

        return None

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.

        Raises
        ------
        ValueError
            If the data and targets have different lengths.
        """

        if len(self.data) != len(self.targets):
            raise ValueError("Data and targets have different lengths.")

        return len(self.data)

    def __getitem__(self, index):
        """Returns the image and the target associated with the sample at the given
        index. If a transform is provided, the image is transformed. If a target
        transform is provided, the target is transformed.

        Parameters
        ----------
        index : int
            Index of the item to return.

        Returns
        -------
        tuple
            Tuple containing the image and the target.
        """

        image_path = self.data[index]
        target = self.targets[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


def _image_path_from_row(row):
    """
    Get the image path from a row.

    Parameters
    ----------
    row : pd.Series
        Row to get the image path from.
    """
    return f'{row["user_id"]}/landmark_aligned_face.{row["face_id"]}.{row["original_image"]}'


def _track_progress(file, verbose: bool = False):
    """
    Track the progress of the extraction.

    The members are streamed while they are extracted. The total number of members
    is not computed beforehand because that would require decompressing the whole
    archive one extra time.

    Parameters
    ----------
    file : tarfile.TarFile
        File to track the progress of.
    """
    for member in tqdm(file, unit=" files", desc="extracting", disable=not verbose):
        # this will be the current file being extracted
        yield member
