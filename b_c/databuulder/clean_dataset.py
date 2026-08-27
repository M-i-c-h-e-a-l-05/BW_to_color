"""
clean_dataset.py

Non-destructive image cleaning pipeline
for the Historical Image Dating Dataset.

Author: Micheal Leveiro
"""

from pathlib import Path
from PIL import Image
import hashlib
import csv
import shutil
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

REJECTED_DIR = Path("rejected")

REPORT_DIR = Path("metadata")

REPORT_FILE = REPORT_DIR / "cleaning_report.csv"

MIN_WIDTH = 224
MIN_HEIGHT = 224

BLUR_THRESHOLD = 50.0

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# STATISTICS
# ============================================================

stats = {

    "total": 0,

    "valid": 0,

    "corrupted": 0,

    "too_small": 0,

    "blurry": 0,

    "duplicates": 0

}


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REJECTED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for folder in [

        "corrupted",

        "too_small",

        "blurry",

        "duplicate"

    ]:

        (REJECTED_DIR / folder).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# IMAGE HASH
# ============================================================

def calculate_hash(file_path):

    hasher = hashlib.sha256()

    try:

        with open(file_path, "rb") as f:

            while True:

                chunk = f.read(1024 * 1024)

                if not chunk:
                    break

                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception:

        return None


# ============================================================
# BLUR DETECTION
# ============================================================

def calculate_blur(file_path):

    try:

        image = cv2.imread(
            str(file_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            return None

        variance = cv2.Laplacian(
            image,
            cv2.CV_64F
        ).var()

        return float(variance)

    except Exception:

        return None


# ============================================================
# MOVE REJECTED FILE
# ============================================================

def move_to_rejected(
    file_path,
    category
):

    destination_dir = (
        REJECTED_DIR / category
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        destination_dir /
        file_path.name
    )

    # Prevent overwriting

    counter = 1

    while destination.exists():

        destination = (
            destination_dir /
            f"{file_path.stem}_{counter}"
            f"{file_path.suffix}"
        )

        counter += 1

    shutil.copy2(
        file_path,
        destination
    )


# ============================================================
# CHECK IMAGE
# ============================================================

def inspect_image(file_path):

    record = {

        "file": str(file_path),

        "width": "",

        "height": "",

        "blur_score": "",

        "status": "VALID",

        "reason": ""

    }

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        with Image.open(file_path) as image:

            image.verify()

        with Image.open(file_path) as image:

            width, height = image.size

    except Exception as e:

        stats["corrupted"] += 1

        record["status"] = "REJECTED"

        record["reason"] = "CORRUPTED"

        move_to_rejected(
            file_path,
            "corrupted"
        )

        return record


    record["width"] = width
    record["height"] = height


    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    if (

        width < MIN_WIDTH

        or

        height < MIN_HEIGHT

    ):

        stats["too_small"] += 1

        record["status"] = "REJECTED"

        record["reason"] = "TOO_SMALL"

        move_to_rejected(
            file_path,
            "too_small"
        )

        return record


    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blur_score = calculate_blur(
        file_path
    )

    record["blur_score"] = blur_score

    if (

        blur_score is not None

        and

        blur_score < BLUR_THRESHOLD

    ):

        stats["blurry"] += 1

        record["status"] = "FLAGGED"

        record["reason"] = "BLURRY"

        move_to_rejected(
            file_path,
            "blurry"
        )

        return record


    stats["valid"] += 1

    return record


# ============================================================
# MAIN CLEANING
# ============================================================

def clean_dataset():

    create_directories()

    print()
    print("=" * 60)
    print("HISTORICAL DATASET CLEANER")
    print("=" * 60)
    print()

    image_files = [

        path

        for path in DATASET_DIR.rglob("*")

        if (

            path.is_file()

            and

            path.suffix.lower()
            in IMAGE_EXTENSIONS

        )

    ]

    stats["total"] = len(
        image_files
    )

    print(
        f"Images found: {stats['total']}"
    )

    print()

    records = []

    seen_hashes = set()

    for index, file_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"[{index}/{stats['total']}] "
            f"{file_path.name}"
        )

        # ----------------------------------------------------
        # Exact duplicate check
        # ----------------------------------------------------

        file_hash = calculate_hash(
            file_path
        )

        if file_hash is not None:

            if file_hash in seen_hashes:

                stats["duplicates"] += 1

                record = {

                    "file": str(file_path),

                    "width": "",

                    "height": "",

                    "blur_score": "",

                    "status": "REJECTED",

                    "reason": "DUPLICATE"

                }

                move_to_rejected(
                    file_path,
                    "duplicate"
                )

                records.append(record)

                continue

            seen_hashes.add(
                file_hash
            )

        # ----------------------------------------------------
        # Image inspection
        # ----------------------------------------------------

        record = inspect_image(
            file_path
        )

        records.append(record)


    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "file",

                "width",

                "height",

                "blur_score",

                "status",

                "reason"

            ]

        )

        writer.writeheader()

        writer.writerows(records)


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(
        f"Total images : {stats['total']}"
    )

    print(
        f"Valid        : {stats['valid']}"
    )

    print(
        f"Corrupted    : {stats['corrupted']}"
    )

    print(
        f"Too small    : {stats['too_small']}"
    )

    print(
        f"Blurry       : {stats['blurry']}"
    )

    print(
        f"Duplicates   : {stats['duplicates']}"
    )

    print()

    print(
        f"Report saved to: {REPORT_FILE}"
    )

    print(
        f"Rejected copies: {REJECTED_DIR}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    clean_dataset()
    