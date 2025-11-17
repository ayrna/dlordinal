from hashlib import md5
from pathlib import Path
from shutil import move, rmtree
from typing import Callable, Optional, Union

import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torchvision.datasets.utils import download_and_extract_archive


class HCI(ImageFolder):
    """
    Historical Color Images (HCI) Decade Database dataset :footcite:t:`palermo2012dating`.

    This dataset contains colour photographs from five decades (1930s-1970s),
    organised for decade classification. Upon first use, the dataset is
    automatically downloaded, verified, preprocessed, and split into training
    and test subsets.

    The preprocessing pipeline includes:
    - verifying and downloading the dataset archive if necessary;
    - extracting and normalising directory names according to class labels;
    - resizing all images to 224x224 pixels;
    - creating a stratified 70/30 train/test split;
    - generating an MD5 checksum file for future integrity checks.

    Parameters
    ----------
    root : str or Path
        Root directory where the dataset will be stored and processed.
    transform : callable, optional
        A function/transform applied to each loaded PIL image.
    target_transform : callable, optional
        A function/transform applied to the target label.
    is_valid_file : callable, optional
        A function that takes a file path and returns ``True`` if the file
        should be included.
    train : bool, default=True
        If ``True``, loads the training split; otherwise, loads the test split.

    Attributes
    ----------
    URL : str
        Download URL for the dataset archive.
    MD5 : str
        MD5 checksum used to verify the downloaded archive.
    CATEGORIES : dict
        Mapping from decade names to numeric class labels (as strings).

    Example
    -----
    >>> from dlordinal.datasets.hci import HCI
    >>> dataset = HCI(root="data", train=True)
    >>> img, label = dataset[0]

    Notes
    -----
    The train/test split is stratified by decade, with 70% of the images in the
    training set and 30% in the test set. Preprocessing is only performed the
    first time the dataset is initialised.

    """

    URL = "http://graphics.cs.cmu.edu/projects/historicalColor/HistoricalColor-ECCV2012-DecadeDatabase.tar"
    MD5 = "afb4c47b7da105c4afd1f27e06bea171"
    CATEGORIES = {"1930s": "0", "1940s": "1", "1950s": "2", "1960s": "3", "1970s": "4"}

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train: bool = True,
    ):
        self.root = Path(root)
        self._prepare_dataset()
        super().__init__(
            root=self.root / "HCI" / ("train" if train else "test"),
            loader=default_loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def _prepare_dataset(self) -> bool:
        target_folder = self.root / "HCI"
        if not target_folder.exists() or not self._verify_md5sums():
            if target_folder.exists():
                rmtree(target_folder, ignore_errors=True)
            # Download and extract
            download_and_extract_archive(self.URL, self.root, md5=self.MD5)
            extracted_folder = (
                self.root
                / "HistoricalColor-ECCV2012"
                / "data"
                / "imgs"
                / "decade_database"
            )
            extracted_folder.rename(target_folder)
            rmtree(self.root / "HistoricalColor-ECCV2012", ignore_errors=True)

            # Rename categories
            for old_name, new_name in self.CATEGORIES.items():
                (target_folder / old_name).rename(target_folder / new_name)

            # Rescale images to 224x224
            for cat in self.CATEGORIES.values():
                cat_folder = target_folder / cat
                for img_path in cat_folder.glob("*"):
                    if img_path.suffix.lower() in IMG_EXTENSIONS:
                        try:
                            with Image.open(img_path) as img:
                                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                                img.save(img_path)
                        except UnidentifiedImageError as e:
                            print(f"Removing corrupted image: {img_path} ({e})")
                            img_path.unlink()

            # Train/test split
            self._split_train_test(target_folder)

            # Create md5sums file
            self._create_md5sums_file()

            return True
        return False

    def _split_train_test(self, folder: Path, train_frac: float = 0.7, seed: int = 0):
        # Gather all images and labels
        records = [
            {"path": str(p.resolve()), "label": int(cat)}
            for cat in self.CATEGORIES.values()
            for p in (folder / cat).glob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        ]
        df = pd.DataFrame(records)
        train_df = df.groupby("label", group_keys=False).sample(
            frac=train_frac, random_state=seed
        )
        test_df = df.drop(train_df.index)

        # Move files
        for df_subset, subset_name in [(train_df, "train"), (test_df, "test")]:
            subset_folder = folder / subset_name
            for row in df_subset.itertuples(index=False):
                dest_folder = subset_folder / str(row.label)
                dest_folder.mkdir(parents=True, exist_ok=True)
                move(row.path, dest_folder / Path(row.path).name)

        # Remove empty original category folders
        for cat in self.CATEGORIES.values():
            cat_folder = folder / cat
            if cat_folder.exists() and not any(cat_folder.iterdir()):
                cat_folder.rmdir()

    def _create_md5sums_file(self):
        md5sum_path = self.root / "HCI" / "md5sums.txt"
        with open(md5sum_path, "w") as f:
            for img_path in (self.root / "HCI").rglob("*"):
                if img_path.suffix.lower() in IMG_EXTENSIONS:
                    with open(img_path, "rb") as img_file:
                        file_hash = md5(img_file.read()).hexdigest()
                    relative_path = img_path.relative_to(self.root / "HCI")
                    f.write(f"{file_hash} {relative_path}\n")

    def _verify_md5sums(self) -> bool:
        md5sum_path = self.root / "HCI" / "md5sums.txt"
        if not md5sum_path.exists():
            return False
        with open(md5sum_path, "r") as f:
            for line in f:
                expected_hash, relative_path = line.strip().split(" ", 1)
                img_path = self.root / "HCI" / relative_path
                if not img_path.exists():
                    return False
                with open(img_path, "rb") as img_file:
                    actual_hash = md5(img_file.read()).hexdigest()
                if actual_hash != expected_hash:
                    return False
        return True
