import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import requests
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


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

        The dataset is stored under the ``adience`` directory inside ``root``.

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
        Test size.
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
        Root directory where the datasets are stored.
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
        ranges: list = [
            (0, 2),
            (4, 6),
            (8, 13),
            (15, 20),
            (25, 32),
            (38, 43),
            (48, 53),
            (60, 100),
        ],
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
        self._ranges = ranges
        self._test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.download = download

        self._version = "1.0"
        self._config = self._get_config_dict()
        self._cache_key = self._make_cache_key(self._config)

        self.data = []
        self.targets = []
        self.classes = []

        self.root_adience_ = self.root / "adience"
        self.data_file_path_ = self.root_adience_ / "aligned.tar.gz"
        self.folds_path_ = self.root_adience_ / "folds"
        self.images_path_ = self.root_adience_ / "aligned"
        self.transformed_images_path_ = self.root_adience_ / "transformed"

        if self.download and (username is None or password is None):
            raise ValueError("username and password are required when download=True")

        if self.download and username is not None and password is not None:
            self._download(username, password)

        if not self._check_input_files():
            raise FileNotFoundError(
                "Some input files are missing. Please, check the documentation of the"
                " root parameter to see the expected directory structure or provide the"
                " username and password to download the files automatically."
            )

        self.folds_ = [
            pd.read_csv(self.folds_path_ / f"fold_{f}_data.txt", sep="\t")
            for f in range(5)
        ]

        self._extract_data()
        self._build_transformed()
        self._df = self._build_dataframe(self.folds_)
        self._build_splits()

    def _check_input_files(self) -> bool:
        """
        Check if the input files are present.
        """

        result = self.data_file_path_.exists() and self.folds_path_.exists()
        result = result and check_integrity(
            str(self.data_file_path_), self.ALIGNED_URL[1]
        )
        for i in range(5):
            result = result and (self.folds_path_ / f"fold_{i}_data.txt").exists()
            result = result and check_integrity(
                str(self.folds_path_ / f"fold_{i}_data.txt"), self.FOLDS_URLS[i][1]
            )
        return result

    def _check_if_extracted(self) -> bool:
        """
        Check if the tar.gz file has been extracted.
        """
        path = self.data_file_path_.parent
        path = path / "aligned"
        return any(path.rglob("*.jpg"))

    def _check_if_transformed(self) -> bool:
        """
        Check if the images have been transformed.
        """
        return self.transformed_images_path_.exists()

    def _check_if_partitioned(self) -> bool:
        """
        Check if a valid cached split exists for the current configuration.
        """

        split_dir = self.root_adience_ / "cache" / f"splits_{self._cache_key}"
        config_path = split_dir / "config.json"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"

        if not (config_path.exists() and train_path.exists() and test_path.exists()):
            return False

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception:
            return False

        ranges = [tuple(r) for r in config.get("ranges")]

        return (
            config.get("version") == self._version
            and config.get("test_size") == self._test_size
            and ranges == self._ranges
        )

    def _download_file(
        self,
        url: str,
        output_path: Path,
        username: str,
        password: str,
        md5: Optional[str] = None,
    ):
        response = requests.get(
            url,
            auth=(username, password),
            stream=True,
            timeout=(10, 300),
        )

        response.raise_for_status()

        total_size = response.headers.get("content-length")
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

            for chunk in response.iter_content(chunk_size=1024 * 1024):

                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        if md5 is not None and not check_integrity(str(output_path), md5):
            raise ValueError(
                f"Downloaded file {output_path} has an invalid MD5 checksum."
            )

    def _download(self, username: str, password: str, force: bool = False):
        self.root_adience_.mkdir(exist_ok=True, parents=True)

        if username is None or password is None:
            raise ValueError(
                "Username and password must be provided to download the dataset."
            )

        if (
            force
            or not self.data_file_path_.exists()
            or not check_integrity(str(self.data_file_path_), self.ALIGNED_URL[1])
        ):
            aligned_name = "aligned.tar.gz"

            if self.verbose:
                print(f"{aligned_name} is missing or corrupted. Downloading...")

            self._download_file(
                url=self.ALIGNED_URL[0],
                output_path=self.root_adience_ / aligned_name,
                username=username,
                password=password,
                md5=self.ALIGNED_URL[1],
            )

        self.folds_path_.mkdir(exist_ok=True, parents=True)

        for url, md5 in self.FOLDS_URLS:
            filename = url.rsplit("/", 1)[-1]

            if force or (
                not (self.folds_path_ / filename).exists()
                or not check_integrity(str(self.folds_path_ / filename), md5)
            ):

                if self.verbose:
                    print(f"{filename} is missing or corrupted. Downloading...")

                self._download_file(
                    url=url,
                    output_path=self.folds_path_ / filename,
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

        if self.verbose:
            print("Extracting file...")

        with tarfile.open(self.data_file_path_, "r:gz") as file:
            path = self.data_file_path_.parent
            path.mkdir(exist_ok=True, parents=True)
            if sys.version_info >= (3, 12):
                file.extractall(
                    path, members=_track_progress(file, self.verbose), filter="data"
                )
            else:
                file.extractall(path, members=_track_progress(file, self.verbose))

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

        self.transformed_images_path_.mkdir(exist_ok=True, parents=True)

        if self.verbose:
            print("Creating transformed images...")

        # We need the full image list from the extracted dataset
        image_paths = list(self.images_path_.rglob("*"))
        image_paths = [p for p in image_paths if p.is_file()]

        for src_image in tqdm(
            image_paths,
            total=len(image_paths),
            disable=not self.verbose,
            desc="transforming",
        ):
            # Preserve relative structure
            rel_path = src_image.relative_to(self.images_path_)
            dst_image = self.transformed_images_path_ / rel_path

            if dst_image.exists():
                continue

            dst_image.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src_image) as img:
                img = img.convert("RGB")

                width_percent = 128 / float(img.size[1])
                new_width = int(img.size[0] * width_percent)

                resized = img.resize(
                    (new_width, 128),
                    Image.Resampling.BILINEAR,
                )

                resized.save(dst_image)

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
                }
            )

            fold_dfs.append(df)

        return pd.concat(fold_dfs, ignore_index=True)

    def _build_splits(self):
        """
        Create train/test splits and persist them as CSV files.

        The split is cached using a hash of the dataset configuration
        (e.g. test_size, ranges).
        """

        split_dir = self.root_adience_ / "cache" / f"splits_{self._cache_key}"
        train_path = split_dir / "train.csv"
        test_path = split_dir / "test.csv"
        config_path = split_dir / "config.json"

        if self._check_if_partitioned():
            if self.verbose:
                print("Splits already exist. Loading from cache.")
            self._load_split_from_csv(train_path, test_path)
            return

        split_dir.mkdir(parents=True, exist_ok=True)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self._test_size,
            random_state=0,
        )

        train_idx, test_idx = next(sss.split(self._df, self._df["age"]))

        train_df = self._df.iloc[train_idx][["path", "age"]]
        test_df = self._df.iloc[test_idx][["path", "age"]]

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        with open(config_path, "w") as f:
            json.dump(self._config, f, indent=2)

        self._load_split_from_csv(train_path, test_path)

    def _load_split_from_csv(self, train_path, test_path):
        df = pd.read_csv(train_path if self.train else test_path)

        self.data = [str(self.transformed_images_path_ / p) for p in df["path"]]
        self.targets = df["age"].tolist()
        self.classes = np.unique(self.targets).tolist()

    def _get_config_dict(self):
        return {
            "ranges": self._ranges,
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
        m = re.match(r"\((\d+), *(\d+)\)", age)
        if m:
            age = (int(m.group(1)), int(m.group(2)))
        else:
            m = re.match(r"(\d+)", age)
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

    Parameters
    ----------
    file : tarfile.TarFile
        File to track the progress of.
    """
    for member in tqdm(file, total=len(file.getmembers()), disable=not verbose):
        # this will be the current file being extracted
        # Go over each member
        yield member
