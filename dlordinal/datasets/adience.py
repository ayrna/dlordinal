import tarfile
from pathlib import Path
from typing import Union

import pandas as pd

# import subprocess
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


class Adience:
    """
    Base class for the Adience dataset.

    Parameters
    ----------
    extract_file_path : Union[str, Path]
        Path to the tar.gz file containing the dataset.
    folds_path : Union[str, Path]
        Path to the folder containing the folds.
    images_path : Union[str, Path]
        Path to the folder containing the images.
    transformed_images_path : Union[str, Path]
        Path to the folder containing the transformed images.
    partition_path : Union[str, Path]
        Path to the folder containing the partitions.
    number_partitions : int, optional
        Number of partitions to create, by default 20.
    ranges : list, optional
        List of age ranges to use, by default [(0, 2), (4, 6), (8, 13),
        (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)].
    test_size : float, optional
        Test size, by default 0.2.
    extract : bool, optional
        Boolean indicating if the tar.gz file should be extracted, by default True.
    transfrom : bool, optional
        Boolean indicating if the images should be transformed and the partitions
        created, by default True.
    """

    def __init__(
        self,
        extract_file_path: Union[str, Path],
        folds_path: Union[str, Path],
        images_path: Union[str, Path],
        transformed_images_path: Union[str, Path],
        partition_path: Union[str, Path],
        number_partitions: int = 20,
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
        extract: bool = True,
        transfrom: bool = True,
    ) -> None:
        super().__init__()

        self.extract_file_path = Path(extract_file_path)
        self.folds_path = Path(folds_path)
        self.images_path = Path(images_path)
        self.transformed_images_path = Path(transformed_images_path)
        self.partition_path = Path(partition_path)

        self.number_partitions = number_partitions
        self.ranges = ranges
        self.test_size = test_size

        folds = [
            pd.read_csv(self.folds_path / f"fold_{f}_data.txt", sep="\t")
            for f in range(5)
        ]

        if extract:
            self.extract_tar_gz()

        if transfrom:
            self.process_and_split(folds)

    def _check_if_extracted(self) -> bool:
        """
        Check if the tar.gz file has been extracted.
        """
        path = self.extract_file_path.parent
        path = path / "aligned"
        return path.exists()

    def _check_if_transformed(self) -> bool:
        """
        Check if the images have been transformed and the partitions created.
        """
        return self.transformed_images_path.exists() or self.partition_path.exists()

    def extract_tar_gz(self):
        """
        Extract the tar.gz file.
        """
        if self._check_if_extracted():
            print("File already extracted.")
            return

        print("Extracting file...")
        with tarfile.open(self.extract_file_path, "r:gz") as file:
            path = self.extract_file_path.parent
            path.mkdir(exist_ok=True, parents=True)
            file.extractall(path, members=self.track_progress(file))
            file.close()

    def process_and_split(self, folds: list) -> None:
        """
        Process the folds and split the images into partitions.

        Parameters
        ----------
        folds : list
            List of folds.
        """
        if self._check_if_transformed():
            print("File already transformed.")
            return

        fold_dfs = list()
        for f, fold in enumerate(folds):
            fold = fold.assign(age=fold["age"].map(self.assign_range))
            notna = fold["age"].notna()
            n_discarded = (~notna).sum()
            print(
                f"Fold {f}: discarding {n_discarded} entries"
                f" ({(n_discarded / len(fold)) * 100:.1f}%)"
            )
            fold = fold.loc[notna]
            fold = fold.assign(age=fold["age"].astype(int))

            fold_dfs.append(
                pd.DataFrame(
                    dict(
                        path=fold.apply(self.image_path_from_row, axis="columns"),
                        age=fold["age"],
                    )
                )
            )
        df: pd.DataFrame = pd.concat(fold_dfs, ignore_index=True)  # type: ignore

        self.transformed_images_path.mkdir(exist_ok=True)
        print("Resizing images...")
        for row in tqdm(df.itertuples(), total=len(df)):
            dst_image = self.transformed_images_path / row.path
            if dst_image.is_file():
                continue
            src_image = self.images_path / row.path
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

        self.partition_path.mkdir(exist_ok=True)
        for partition in range(self.number_partitions):
            for c in range(len(self.ranges)):
                (self.partition_path / f"{partition}/train/{c}").mkdir(
                    parents=True, exist_ok=True
                )
                (self.partition_path / f"{partition}/test/{c}").mkdir(
                    parents=True, exist_ok=True
                )

        sss = StratifiedShuffleSplit(
            self.number_partitions, test_size=self.test_size, random_state=0
        )
        for partition, (train_index, test_index) in tqdm(
            enumerate(sss.split(df, df["age"]))
        ):
            train_df: pd.DataFrame = df.iloc[train_index]  # type: ignore
            test_df: pd.DataFrame = df.iloc[test_index]  # type: ignore
            for name, partition_df in zip(("train", "test"), (train_df, test_df)):
                for row in tqdm(
                    partition_df.itertuples(),
                    total=len(partition_df),
                    leave=False,
                    desc=name,
                ):
                    image_path = self.transformed_images_path / row.path
                    assert image_path.is_file()
                    new_path = (
                        self.partition_path
                        / f"{partition}/{name}/{row.age}/{image_path.name}"
                    )
                    if not new_path.exists():
                        new_path.symlink_to(image_path.resolve())

    def assign_range(self, age: str):
        """
        Assign an age range to an age.

        Parameters
        ----------
        age : str
            Age to assign a range to.
        """
        age = eval(age)

        if age is None:
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

    def image_path_from_row(self, row):
        """
        Get the image path from a row.

        Parameters
        ----------
        row : pd.Series
            Row to get the image path from.
        """
        return f'{row["user_id"]}/landmark_aligned_face.{row["face_id"]}.{row["original_image"]}'

    def track_progress(self, file):
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
