import re
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm

from ._utils import (
    marker_matches,
    n_splits_from_test_size,
    remove_marker,
    stratified_group_split,
    write_marker,
)

_DEFAULT_CATEGORIES = (3, 11, 16, 24, 40)
_DEFAULT_VALIDATION_SIZE = 0.15

_DOWNLOAD_MARKER = ".dlordinal_downloaded.json"
_PROCESS_MARKER = ".dlordinal_processed.json"
_SPLIT_MARKER = ".dlordinal_split.json"


class FGNet(VisionDataset):
    """
    Base class for FGNet dataset.

    Attributes
    ----------
    root : Path
        Root directory of the dataset.
    target_size : tuple
        Size of the images after resizing.
    categories : list
        List of categories to be used.
    test_size : float
        Size of the test set.
    validation_size : float
        Deprecated, it has no effect.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    data : pd.DataFrame
        Dataframe containing the dataset.

    Parameters
    ----------
    root : str or Path
        Root directory of the dataset.
    download : bool, optional, default = True
        If True, downloads the dataset from the internet and puts it in the root directory.
        If the dataset is already downloaded, it is not downloaded again.
    target_size : tuple, optional
        Size of the images after resizing. Default is (128, 128).
    categories : list, optional
        List of categories to be used. Default is [3, 11, 16, 24, 40].
    test_size : float, optional
        Approximate fraction of the images used for the test set. Default is 0.2.
        The split is subject-disjoint: all the images of a subject (the numeric
        prefix of the file name, e.g. ``001`` in ``001A02.JPG``) belong to the
        same partition. It is obtained as one fold of a
        :class:`sklearn.model_selection.StratifiedGroupKFold` with
        ``n_splits = max(2, round(1 / test_size))`` folds, so the actual test
        fraction is approximately ``1 / n_splits``. The first fold whose train and
        test partitions both contain every class is used.
    validation_size : float, optional
        Deprecated, it has no effect and will be removed in a future version. The
        train set always contains all the samples that are not in the test set.
    train : bool, optional
        If True, returns the training dataset, otherwise returns the test dataset. Default is True.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.

    Notes
    -----
    The processed images and the train/test split are cached inside ``root``. Each
    step writes a completion marker with its configuration (``target_size``,
    ``categories``, ``test_size``, ...) once it has finished, and is redone when the
    marker is missing or does not match the current configuration.
    """

    URL = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"
    MD5 = "1206978cac3626321b84c22b24cc8d19"

    # Bump these versions when the processing or the split procedure changes
    _PROCESS_VERSION = 2
    _SPLIT_VERSION = 2
    _SPLIT_SEED = 1

    # Attributes
    root: Path
    target_size: tuple
    categories: list
    test_size: float
    validation_size: float
    transform: Optional[Callable]
    target_transform: Optional[Callable]
    data: pd.DataFrame

    def __init__(
        self,
        root: Union[str, Path],
        download: bool = True,
        target_size: tuple = (128, 128),
        categories: Sequence[int] = _DEFAULT_CATEGORIES,
        test_size: float = 0.2,
        validation_size: float = _DEFAULT_VALIDATION_SIZE,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(FGNet, self).__init__(
            str(root), transform=transform, target_transform=target_transform
        )

        n_splits_from_test_size(test_size)  # validates test_size

        if validation_size != _DEFAULT_VALIDATION_SIZE:
            warnings.warn(
                "validation_size is deprecated and has no effect: the train set"
                " contains all the samples that are not in the test set. It will be"
                " removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.root = Path(root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.target_size = tuple(target_size)
        self.categories = list(categories)
        self.test_size = test_size
        self.validation_size = validation_size
        self.transform = transform
        self.target_transform = target_transform

        original_path = self.root / "FGNET/images"
        processed_path = self.root / "FGNET/data_processed"

        original_csv_path = self.root / "FGNET/data_processed/fgnet.csv"
        train_csv_path = self.root / "FGNET/data_processed/train.csv"
        test_csv_path = self.root / "FGNET/data_processed/test.csv"

        original_images_path = self.root / "FGNET/data_processed"
        train_images_path = self.root / "FGNET/train"
        test_images_path = self.root / "FGNET/test"

        if download:
            self.download()
        if not self._check_integrity_download():
            if self._has_legacy_download():
                warnings.warn(
                    "The FGNet files in root were extracted by an older version and"
                    " their integrity cannot be verified. Use download=True to verify"
                    " and re-extract them.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    "Dataset not found or corrupted. You can use download=True to"
                    " download it"
                )

        self.process(original_path, processed_path)
        self.split(
            original_csv_path,
            train_csv_path,
            test_csv_path,
            original_images_path,
            train_images_path,
            test_images_path,
        )

        # Load train and test dataframes
        csv_path = train_csv_path if train else test_csv_path
        self.data = pd.read_csv(csv_path, dtype={"path": str})

    def __str__(self) -> str:
        return "FGNet"

    def __len__(self) -> int:
        """
        Obtain the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """

        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to get.

        Returns
        -------
        tuple
            (image, target) where target is the class index of the target class.
        """
        img_path = self.root / "FGNET" / "data_processed" / self.data["path"].iat[index]

        # Load the image as PIL.Image.Image
        with Image.open(img_path) as image:
            image = image.convert("RGB")

        # Apply the transform if defined
        if self.transform:
            image = self.transform(image)

        target = int(self.data["category"].iat[index])
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    @property
    def targets(self) -> Union[List[int], np.ndarray, List[Any]]:
        """
        Return the targets of the dataset. If ``target_transform`` is defined, it is
        applied to each target, as in ``__getitem__``, and the results are stacked
        into a NumPy array (e.g. ``target_transform=np.array`` gives a 1D integer
        array). Results that cannot be stacked are returned as a list.

        Returns
        -------
        list or np.ndarray
            List of targets, or array of transformed targets.
        """

        targets = [int(t) for t in self.data["category"]]
        if self.target_transform:
            transformed = [self.target_transform(t) for t in targets]
            try:
                return np.asarray(transformed)
            except (ValueError, TypeError):
                return transformed
        return targets

    @property
    def classes(self) -> List[int]:
        """
        Return the unique classes in the dataset.

        Returns
        -------
        list
            List of unique classes.
        """
        return np.unique(self.data["category"]).tolist()

    def download(self) -> None:
        """
        Download the FGNet dataset and extract it.
        """
        if self._check_integrity_download():
            print("Files already downloaded and verified")
            return

        # Reuses root/fgnet.zip without downloading it again if its MD5 matches
        download_and_extract_archive(
            self.URL,
            str(self.root),
            filename="fgnet.zip",
            md5=self.MD5,
        )
        write_marker(self.root / "FGNET" / _DOWNLOAD_MARKER, {"md5": self.MD5})

    def process(self, original_path, processed_path):
        """
        Process the FGNet dataset and save it in the processed_path.

        Parameters
        ----------
        original_path : Path
            Path to the original dataset.
        processed_path : Path
            Path to save the processed dataset.
        """
        if self._check_integrity_process():
            print("Files already processed and verified")
            return

        processed_path = Path(processed_path)
        # The split depends on the processed data: invalidate both
        remove_marker(processed_path / _PROCESS_MARKER)
        remove_marker(processed_path / _SPLIT_MARKER)

        data = self.load_data(Path(original_path))
        df = pd.DataFrame(data, columns=["path", "category"])
        processed_path.mkdir(parents=True, exist_ok=True)
        self.process_images_from_df(df, Path(original_path), processed_path)
        df.to_csv(processed_path / "fgnet.csv", index=False)
        write_marker(processed_path / _PROCESS_MARKER, self._process_metadata())
        return df

    def split(
        self,
        original_csv_path: Path,
        train_csv_path: Path,
        test_csv_path: Path,
        original_images_path: Path,
        train_images_path: Path,
        test_images_path: Path,
    ):
        """
        Split the FGNet dataset into subject-disjoint train and test sets.

        Parameters
        ----------
        original_csv_path : Path
            Path to the original csv file.
        train_csv_path : Path
            Path to save the train csv file.
        test_csv_path : Path
            Path to save the test csv file.
        original_images_path : Path
            Unused, kept for backward compatibility.
        train_images_path : Path
            Unused, kept for backward compatibility.
        test_images_path : Path
            Unused, kept for backward compatibility.
        """
        if self._check_integrity_split():
            print("Files already split and verified")
            return

        split_marker = Path(train_csv_path).parent / _SPLIT_MARKER
        remove_marker(split_marker)

        train, test = self.split_dataframe(
            original_csv_path, train_images_path, original_images_path, test_images_path
        )

        test.to_csv(test_csv_path, index=False)
        train.to_csv(train_csv_path, index=False)
        write_marker(split_marker, self._split_metadata())

    def _process_metadata(self) -> dict:
        return {
            "categories": list(self.categories),
            "target_size": list(self.target_size),
            "version": self._PROCESS_VERSION,
        }

    def _split_metadata(self) -> dict:
        return {
            "group": "subject",
            "n_splits": self._n_splits,
            "process": self._process_metadata(),
            "seed": self._SPLIT_SEED,
            "split_method": "StratifiedGroupKFold",
            "test_size": self.test_size,
            "version": self._SPLIT_VERSION,
        }

    @property
    def _n_splits(self) -> int:
        return n_splits_from_test_size(self.test_size)

    def _has_legacy_download(self) -> bool:
        """
        Check for files extracted by a previous version, which did not write a
        download marker.
        """
        images = self.root / "FGNET" / "images"
        return images.is_dir() and any(images.iterdir())

    def _check_integrity_download(self) -> bool:
        """
        Check if the FGNet dataset has been completely downloaded and extracted.
        """
        return marker_matches(self.root / "FGNET" / _DOWNLOAD_MARKER, {"md5": self.MD5})

    def _check_integrity_process_split(self) -> bool:
        """
        Check if the FGNet dataset is processed and split.
        """
        return self._check_integrity_process() and self._check_integrity_split()

    def _check_integrity_process(self) -> bool:
        """
        Check if the FGNet dataset has been completely processed with the current
        configuration.
        """
        return marker_matches(
            self.root / "FGNET/data_processed" / _PROCESS_MARKER,
            self._process_metadata(),
        )

    def _check_integrity_split(self) -> bool:
        """
        Check if the FGNet dataset has been split with the current configuration.
        """
        processed = self.root / "FGNET/data_processed"
        return (
            (processed / "train.csv").exists()
            and (processed / "test.csv").exists()
            and marker_matches(processed / _SPLIT_MARKER, self._split_metadata())
        )

    def get_age_from_filename(self, filename):
        """
        Get the age from the filename.

        Parameters
        ----------
        filename : str
            Filename of the image.
        """
        m = re.match("[0-9]+A([0-9]+).*", filename)
        if m:
            return int(m.groups()[0])
        return None

    def get_subject_from_filename(self, filename):
        """
        Get the subject identifier from the filename (e.g. ``"001"`` for
        ``"001A02.JPG"``).

        Parameters
        ----------
        filename : str
            Filename of the image.
        """
        m = re.match("([0-9]+)A[0-9]+", filename)
        if m:
            return m.groups()[0]
        return None

    def find_category(self, real_age):
        """
        Find the category of the real age.

        Parameters
        ----------
        real_age : int
            Real age of the image.
        """
        for i, age in enumerate(self.categories):
            if real_age < age:
                return i
        return len(self.categories)

    def load_data(self, original_path: Path):
        """
        Load the data from the original_path. Only image files whose name contains
        the age are used. Files are sorted by name, so that the result does not
        depend on the order in which the file system lists them.

        Parameters
        ----------
        original_path : Path
            Path to the original dataset.
        """
        data = []
        for img in sorted(Path(original_path).iterdir(), key=lambda p: p.name):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTENSIONS:
                continue
            age = self.get_age_from_filename(img.name)
            if age is None:
                warnings.warn(
                    f"Skipping {img.name}: the age cannot be parsed from its name.",
                    UserWarning,
                )
                continue
            category = self.find_category(age)
            data.append([img.name, category])

        return data

    def process_images_from_df(
        self, df: pd.DataFrame, original_path: Path, processed_path: Path
    ):
        """
        Process the images from the dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with the images.
        original_path : Path
            Path to the original dataset.
        processed_path : Path
            Path to save the processed dataset.
        """
        for idx, row in tqdm(
            df.iterrows(), total=df.shape[0], desc="Processing images", unit="image"
        ):
            path = original_path / Path(row["path"])
            processed_path_images = processed_path / Path(row["path"])
            img = imread(path)
            img = img_as_ubyte(resize(img, self.target_size, anti_aliasing=True))

            processed_path_images.parent.mkdir(parents=True, exist_ok=True)
            imsave(processed_path_images, img, check_contrast=False)

    def split_dataframe(
        self,
        csv_path: Path,
        train_images_path: Optional[Path] = None,
        original_images_path: Optional[Path] = None,
        test_images_path: Optional[Path] = None,
    ):
        """
        Split the dataframe into subject-disjoint train and test sets.

        The test set is one fold of a
        :class:`sklearn.model_selection.StratifiedGroupKFold` with
        ``max(2, round(1 / test_size))`` folds, stratified by category and grouped
        by subject (see :func:`dlordinal.datasets._utils.stratified_group_split`).

        Parameters
        ----------
        csv_path : Path
            Path to the csv file.
        train_images_path : Path, optional
            Unused, kept for backward compatibility.
        original_images_path : Path, optional
            Unused, kept for backward compatibility.
        test_images_path : Path, optional
            Unused, kept for backward compatibility.

        Returns
        -------
        train_df : pd.DataFrame
            Train set with columns ``path`` and ``category``.
        test_df : pd.DataFrame
            Test set with columns ``path`` and ``category``.
        """
        df = pd.read_csv(csv_path, dtype={"path": str})
        df = df.sort_values("path", ignore_index=True)[["path", "category"]]
        groups = df["path"].map(self.get_subject_from_filename)

        train_idx, test_idx = stratified_group_split(
            df["category"],
            groups=groups,
            test_size=self.test_size,
            seed=self._SPLIT_SEED,
        )

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, test_df
