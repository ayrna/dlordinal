import re
import sys
import tarfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


class Adience(VisionDataset):
    """
    Base class for the Adience dataset.

    Parameters
    ----------
    root : Union[str, Path]
        Root directory where the datasets are stored. The Adience dataset is expected
        to be located under the `adience` directory inside the root directory. In the
        `adience` directory, the following files are expected:
        1) `aligned.tar.gz`: a tar.gz file containing the images;
        2) `folds`: a directory containing the folds. Each fold is expected to be
        a file named `fold_{f}_data.txt`, where `f` is the fold number starting from 0.
        These files can be downloaded from the Adience website
        (https://talhassner.github.io/home/projects/Adience/Adience-data.html)
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

    Attributes
    ----------
    root : Path
        Root directory where the datasets are stored.
    train : bool
        Whether to use the training or test partition.
    ranges : list
        List of age ranges to use to define the categories.
    test_size : float
        Percentage of the dataset to use for testing.
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
    """

    root: Path
    train: bool
    ranges: list
    test_size: float
    transform: Optional[Callable]
    target_transform: Optional[Callable]
    verbose: bool
    data: list
    targets: list
    classes: list

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
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.train = train
        self.ranges = ranges
        self.test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose
        self.data = []
        self.targets = []
        self.classes = []

        self.root_adience_ = self.root / "adience"
        self.data_file_path_ = self.root_adience_ / "aligned.tar.gz"
        self.folds_path_ = self.root_adience_ / "folds"
        self.images_path_ = self.root_adience_ / "aligned"
        self.transformed_images_path_ = self.root_adience_ / "transformed"

        if self.train:
            self.partition_path_ = self.root_adience_ / "train"
        else:
            self.partition_path_ = self.root_adience_ / "test"

        if not self._check_input_files():
            raise FileNotFoundError(
                "Some input files are missing. Please, check the documentation of the"
                " root parameter to see the expected directory structure."
            )

        self.folds_ = [
            pd.read_csv(self.folds_path_ / f"fold_{f}_data.txt", sep="\t")
            for f in range(5)
        ]

        self._extract_data()
        self._process_and_split(self.folds_)
        self._load_data()

    def _check_input_files(self) -> bool:
        """
        Check if the input files are present.
        """

        result = self.data_file_path_.exists() and self.folds_path_.exists()
        for i in range(5):
            result = result and (self.folds_path_ / f"fold_{i}_data.txt").exists()
        return result

    def _check_if_extracted(self) -> bool:
        """
        Check if the tar.gz file has been extracted.
        """
        path = self.data_file_path_.parent
        path = path / "aligned"
        return path.exists()

    def _check_if_transformed(self) -> bool:
        """
        Check if the images have been transformed.
        """
        return self.transformed_images_path_.exists()

    def _check_if_partitioned(self) -> bool:
        """
        Check if the images have been partitioned.
        """
        return self.partition_path_.exists()

    def _extract_data(self):
        """
        Extract the data tar.gz file.
        """
        if self._check_if_extracted():
            if self.verbose:
                print("File already extracted.")
            return

        print("Extracting file...")
        with tarfile.open(self.data_file_path_, "r:gz") as file:
            path = self.data_file_path_.parent
            path.mkdir(exist_ok=True, parents=True)
            if sys.version_info >= (3, 12):
                file.extractall(path, members=_track_progress(file), filter="data")
            else:
                file.extractall(path, members=_track_progress(file))

    def _process_and_split(self, folds: list) -> None:
        """
        Process the folds and split the images into partitions.

        Parameters
        ----------
        folds : list
            List of folds.
        """

        is_transformed = self._check_if_transformed()
        is_partitioned = self._check_if_partitioned()

        if is_transformed and is_partitioned:
            if self.verbose:
                print("Files already transformed and partitioned.")
            return

        fold_dfs = list()
        for f, fold in enumerate(folds):
            notna = fold["age"].notna()
            n_discarded = (~notna).sum()
            if self.verbose:
                print(
                    f"Fold {f}: discarding {n_discarded} entries"
                    f" ({(n_discarded / len(fold)) * 100:.1f}%)"
                )
            fold = fold.loc[notna]
            fold = fold.assign(age=fold["age"].map(self._assign_range))
            fold = fold.dropna(subset=["age"])
            fold = fold.assign(age=fold["age"].astype(int))

            fold_dfs.append(
                pd.DataFrame(
                    dict(
                        path=fold.apply(_image_path_from_row, axis="columns"),
                        age=fold["age"],
                    )
                )
            )
        self.df_: pd.DataFrame = pd.concat(fold_dfs, ignore_index=True)

        if is_transformed:
            if self.verbose:
                print("File already transformed.")
        else:
            self.transformed_images_path_.mkdir(exist_ok=True)
            if self.verbose:
                print("Resizing images...")
            for row in tqdm(self.df_.itertuples(), total=len(self.df_)):
                dst_image = self.transformed_images_path_ / row.path
                if dst_image.is_file():
                    continue
                src_image = self.images_path_ / row.path
                dst_image.parent.mkdir(exist_ok=True, parents=True)

                # open the source image
                with Image.open(src_image) as img:
                    # calculate the new width that maintains the aspect ratio
                    width_percent = 128 / float(img.size[1])
                    new_width = int((float(img.size[0]) * float(width_percent)))

                    # resize the image using the calculated width and 128 height
                    resized_img = img.resize((new_width, 128))

                    # save the resized image to the destination path
                    resized_img.save(dst_image)

        if is_partitioned:
            if self.verbose:
                print("File already partitioned.")
        else:
            for c in range(len(self.ranges)):
                (self.partition_path_ / f"{c}").mkdir(parents=True, exist_ok=True)

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_size, random_state=0
            )

            train_index, test_index = next(sss.split(self.df_, self.df_["age"]))
            if self.train:
                name = "train"
                partition_df: pd.DataFrame = self.df_.iloc[train_index]
            else:
                name = "test"
                partition_df: pd.DataFrame = self.df_.iloc[test_index]

            for row in tqdm(
                partition_df.itertuples(),
                total=len(partition_df),
                leave=False,
                desc=name,
            ):
                image_path = self.transformed_images_path_ / row.path
                assert image_path.is_file()
                new_path = self.root_adience_ / f"{name}/{row.age}/{image_path.name}"
                if not new_path.exists():
                    new_path.symlink_to(image_path.resolve())

    def _load_data(self):
        for cls in range(len(self.ranges)):
            path = self.partition_path_ / f"{cls}"
            for image_path in path.iterdir():
                self.data.append(str(image_path))
                self.targets.append(cls)

        self.classes = np.unique(self.targets).tolist()

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

        if age in self.ranges:
            return self.ranges.index(age)

        if isinstance(age, tuple):
            age_minimum, age_maximum = age
            for i, (range_minimum, range_maximum) in enumerate(self.ranges):
                if (age_minimum >= range_minimum) and (age_maximum <= range_maximum):
                    return i
            return None

        if isinstance(age, int):
            for i, (range_minimum, range_maximum) in enumerate(self.ranges):
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

        image = Image.open(image_path)

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


def _track_progress(file):
    """
    Track the progress of the extraction.

    Parameters
    ----------
    file : tarfile.TarFile
        File to track the progress of.
    """
    for member in tqdm(file, total=len(file.getmembers())):
        # this will be the current file being extracted
        # Go over each member
        yield member
