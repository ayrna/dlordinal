import re
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte
from sklearn.model_selection import train_test_split
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from tqdm import tqdm


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
        Size of the validation set.
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
        Size of the test set. Default is 0.2.
    validation_size : float, optional
        Size of the validation set. Default is 0.15.
    train : bool, optional
        If True, returns the training dataset, otherwise returns the test dataset. Default is True.
    transform : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    target_transform : callable, optional
        A function/transform that takes in the target and transforms it.
    """

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
        categories: list = [3, 11, 16, 24, 40],
        test_size: float = 0.2,
        validation_size: float = 0.15,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super(FGNet, self).__init__(
            str(root), transform=transform, target_transform=target_transform
        )

        self.root = Path(root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.categories = categories
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
        if train:
            self.data = pd.read_csv(train_csv_path)
        else:
            self.data = pd.read_csv(test_csv_path)

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
        img_path = (
            self.root / "FGNET" / "data_processed" / self.data.iloc[index]["path"]
        )

        # Cargar la imagen como PIL.Image.Image
        image = Image.open(img_path)
        image = image.convert("RGB")

        # Aplicar transformación si está definida
        if self.transform:
            image = self.transform(image)

        target = int(self.data.iloc[index]["category"])
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    @property
    def targets(self) -> List[int]:
        """
        Return the targets of the dataset.

        Returns
        -------
        list
            List of targets.
        """

        if self.target_transform:
            return self.target_transform(self.data["category"])
        else:
            return self.data["category"].tolist()

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

        download_and_extract_archive(
            "http://yanweifu.github.io/FG_NET_data/FGNET.zip",
            str(self.root),
            filename="fgnet.zip",
            md5="1206978cac3626321b84c22b24cc8d19",
        )

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

        data = self.load_data(original_path)
        df = pd.DataFrame(data, columns=["path", "category"])
        processed_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path / "fgnet.csv", index=False)
        self.process_images_from_df(df, original_path, processed_path)
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
        Split the FGNet dataset into train and test sets.

        Parameters
        ----------
        original_csv_path : Path
            Path to the original csv file.
        train_csv_path : Path
            Path to save the train csv file.
        test_csv_path : Path
            Path to save the test csv file.
        original_images_path : Path
            Path to the original images.
        train_images_path : Path
            Path to save the train images.
        test_images_path : Path
            Path to save the test images.
        """
        if self._check_integrity_split():
            print("Files already split and verified")
            return

        train, test = self.split_dataframe(
            original_csv_path, train_images_path, original_images_path, test_images_path
        )

        test.to_csv(test_csv_path, index=False)
        train.to_csv(train_csv_path, index=False)

    def _check_integrity_download(self) -> bool:
        """
        Check if the FGNet dataset is downloaded and extracted.
        """
        return (self.root / "FGNET").exists()

    def _check_integrity_process_split(self) -> bool:
        """
        Check if the FGNet dataset is processed and split.
        """
        return (
            (self.root / "FGNET/data_processed").exists()
            and (self.root / "FGNET/trainval").exists()
            and (self.root / "FGNET/test").exists()
        )

    def _check_integrity_process(self) -> bool:
        """
        Check if the FGNet dataset is processed.
        """
        return (self.root / "FGNET/data_processed").exists()

    def _check_integrity_split(self) -> bool:
        """
        Check if the FGNet dataset is split.
        """
        return (self.root / "FGNET/train").exists() and (
            self.root / "FGNET/test"
        ).exists()

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
        Load the data from the original_path.

        Parameters
        ----------
        original_path : Path
            Path to the original dataset.
        """
        data = []
        for img in original_path.iterdir():
            age = self.get_age_from_filename(img.name)
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
        train_images_path: Path,
        original_images_path: Path,
        test_images_path: Path,
    ):
        """
        Split the dataframe into train and test sets.

        Parameters
        ----------
        csv_path : Path
            Path to the csv file.
        train_images_path : Path
            Path to save the train images.
        original_images_path : Path
            Path to the original images.
        test_images_path : Path
            Path to save the test images.
        """
        df = pd.read_csv(csv_path)
        x = np.array(df["path"])
        y = np.array(df["category"])
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=1,
            stratify=y,
        )

        for path, label in zip(x_train, y_train):
            train_path = train_images_path / str(label)
            train_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(original_images_path / path, train_path / path)

        for path, label in zip(x_test, y_test):
            test_path = test_images_path / str(label)
            test_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(original_images_path / path, test_path / path)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.validation_size,
            random_state=1,
            stratify=y_train,
        )

        train = np.hstack((x_train[:, np.newaxis], y_train[:, np.newaxis]))
        val = np.hstack((x_val[:, np.newaxis], y_val[:, np.newaxis]))
        test = np.hstack((x_test[:, np.newaxis], y_test[:, np.newaxis]))
        trainval = np.vstack((train, val))

        test_df = pd.DataFrame(data=test, columns=["path", "category"])
        train_df = pd.DataFrame(data=trainval, columns=["path", "category"])

        return train_df, test_df
