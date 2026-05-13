import hashlib
import random
import shutil
import tarfile
from pathlib import Path

from PIL import Image

RANGES = [
    "(0, 2)",
    "(4, 6)",
    "(8, 13)",
    "(15, 20)",
    "(25, 32)",
    "(38, 43)",
    "(48, 53)",
    "(60, 100)",
]


def generate_fake_adience(root: Path, n_users: int = 12, imgs_per_user: int = 2):
    root = Path(root)
    images_path = root / "aligned"
    images_tar_path = root / "aligned.tar.gz"
    folds_path = root / "folds"

    images_path.mkdir(parents=True, exist_ok=True)
    folds_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)

    for fold in range(5):
        all_rows = []

        for u in range(n_users):
            user_id = f"{u}@{u*3}"

            for i in range(imgs_per_user):
                age = rng.choice(RANGES)
                face_id = rng.randint(1, 9999)
                image_name = (
                    f"{rng.randint(1000000,9999999)}"
                    f"_{rng.getrandbits(32):08x}_o.jpg"
                )

                # folder structure like real dataset
                user_dir = images_path / user_id
                user_dir.mkdir(parents=True, exist_ok=True)

                img_path = user_dir / f"landmark_aligned_face.{face_id}.{image_name}"

                # create dummy image
                img = Image.new("RGB", (816, 816))
                img.save(img_path)

                all_rows.append(
                    [
                        user_id,
                        image_name,
                        face_id,
                        age,
                        rng.choice(["m", "f"]),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                        rng.randint(0, 1000),
                    ]
                )

        fold_path = folds_path / f"fold_{fold}_data.txt"

        header = (
            "user_id\toriginal_image\tface_id\tage\tgender\tx\ty\tdx\tdy\t"
            "tilt_ang\tfiducial_yaw_angle\tfiducial_score"
        )

        with open(fold_path, "w") as f:
            f.write(header + "\n")
            for row in all_rows:
                f.write("\t".join(map(str, row)) + "\n")

    # Create targz file with aligned images
    with tarfile.open(images_tar_path, "w:gz") as tar:
        tar.add(images_path, arcname="aligned")

    # Remove aligned directory after creating tar.gz
    shutil.rmtree(images_path)

    return root


def compute_md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_adience_md5sums(root: Path):
    md5sums = {
        "aligned": compute_md5sum(root / "aligned.tar.gz"),
        "fold0": compute_md5sum(root / "folds/fold_0_data.txt"),
        "fold1": compute_md5sum(root / "folds/fold_1_data.txt"),
        "fold2": compute_md5sum(root / "folds/fold_2_data.txt"),
        "fold3": compute_md5sum(root / "folds/fold_3_data.txt"),
        "fold4": compute_md5sum(root / "folds/fold_4_data.txt"),
    }
    return md5sums
