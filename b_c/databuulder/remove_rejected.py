"""
remove_rejected.py

Historical Dataset Cleaning

1. Removes images that were already rejected during
   the first cleaning stage.

2. Detects near-duplicate images using perceptual hashing.

3. Uses hash bucketing to avoid comparing every image
   against every other image.

4. Moves detected near-duplicates to:
   rejected/near_duplicate/

IMPORTANT:
- Original accepted images are not modified.
- Near-duplicates are moved, not permanently deleted.
- The first encountered version is kept.

Author: Micheal Leveiro
"""

from pathlib import Path
import shutil
import csv

from PIL import Image
import imagehash


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

REJECTED_DIR = Path("rejected")

NEAR_DUPLICATE_DIR = (
    REJECTED_DIR / "near_duplicate"
)

REPORT_DIR = Path("metadata")

REPORT_FILE = (
    REPORT_DIR /
    "near_duplicate_report.csv"
)


# Previously rejected categories

CATEGORIES = [
    "corrupted",
    "too_small",
    "blurry",
    "duplicate"
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# pHASH SETTINGS
# ============================================================

# pHash is normally 64 bits.

HASH_BITS = 64


# Maximum Hamming distance considered
# a near duplicate.

PHASH_THRESHOLD = 6


# Number of bits used for each bucket.

BUCKET_BITS = 16


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    NEAR_DUPLICATE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REMOVE PREVIOUSLY REJECTED IMAGES
# ============================================================

def remove_rejected_images():

    total_removed = 0

    print("=" * 60)
    print("REMOVING PREVIOUSLY REJECTED IMAGES")
    print("=" * 60)

    # --------------------------------------------------------
    # Build a filename -> [paths] index of the dataset ONCE,
    # instead of doing a full rglob() per rejected file.
    # --------------------------------------------------------

    print("Indexing dataset files...")

    dataset_index = {}

    for file_path in DATASET_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        dataset_index.setdefault(
            file_path.name,
            []
        ).append(file_path)

    print(
        f"Indexed {sum(len(v) for v in dataset_index.values())} "
        f"dataset files"
    )

    print()

    for category in CATEGORIES:

        rejected_folder = (
            REJECTED_DIR / category
        )

        if not rejected_folder.exists():
            continue

        for rejected_file in rejected_folder.iterdir():

            if not rejected_file.is_file():
                continue

            filename = rejected_file.name

            matches = dataset_index.get(
                filename,
                []
            )

            if not matches:
                continue

            for original_file in matches:

                try:

                    original_file.unlink()

                    total_removed += 1

                except Exception as e:

                    print(
                        f"ERROR removing "
                        f"{original_file}: {e}"
                    )

    print()

    print(
        f"Previously rejected images removed: "
        f"{total_removed}"
    )

    print()


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_image_files():

    files = []

    for file_path in DATASET_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if (
            file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            files.append(file_path)

    return sorted(files)


# ============================================================
# CALCULATE pHASH
# ============================================================

def calculate_phash(file_path):

    try:

        with Image.open(file_path) as image:

            image = image.convert("RGB")

            return imagehash.phash(
                image
            )

    except Exception as e:

        print(
            f"PHASH ERROR: {file_path}"
        )

        print(
            f"Reason: {e}"
        )

        return None


# ============================================================
# CONVERT HASH TO INTEGER
# ============================================================

def hash_to_integer(image_hash):

    return int(
        str(image_hash),
        16
    )


# ============================================================
# CREATE BUCKET KEYS
# ============================================================

def get_bucket_keys(
    hash_value
):
    """
    Creates multiple bucket keys.

    Instead of comparing a hash against
    every other hash, only hashes sharing
    at least one bucket are compared.

    With 64-bit pHash and 16-bit buckets,
    we divide the hash into four segments.

    """

    keys = []

    for offset in range(
        0,
        HASH_BITS,
        BUCKET_BITS
    ):

        mask = (
            (1 << BUCKET_BITS) - 1
        )

        bucket = (
            hash_value >>
            offset
        ) & mask

        keys.append(
            (
                offset,
                bucket
            )
        )

    return keys


# ============================================================
# MOVE NEAR DUPLICATE
# ============================================================

def move_near_duplicate(
    file_path
):

    destination = (
        NEAR_DUPLICATE_DIR /
        file_path.name
    )

    counter = 1

    while destination.exists():

        destination = (
            NEAR_DUPLICATE_DIR /
            f"{file_path.stem}_"
            f"{counter}"
            f"{file_path.suffix}"
        )

        counter += 1

    try:

        shutil.move(
            str(file_path),
            str(destination)
        )

        return destination

    except Exception as e:

        print(
            f"MOVE ERROR: {file_path}"
        )

        print(
            f"Reason: {e}"
        )

        return None


# ============================================================
# PERCEPTUAL DUPLICATE DETECTION
# ============================================================

def find_near_duplicates():

    print("=" * 60)
    print("OPTIMIZED NEAR-DUPLICATE DETECTION")
    print("=" * 60)

    image_files = get_image_files()

    total_images = len(
        image_files
    )

    print(
        f"Images found: {total_images}"
    )

    print(
        f"pHash threshold: "
        f"{PHASH_THRESHOLD}"
    )

    print(
        f"Bucket size: "
        f"{BUCKET_BITS} bits"
    )

    print()

    # --------------------------------------------------------
    # Buckets
    #
    # Each bucket contains:
    #
    # {
    #     (offset, bucket_value):
    #         [(hash_integer, image_path), ...]
    # }
    #
    # --------------------------------------------------------

    buckets = {}

    # --------------------------------------------------------
    # Hash information
    # --------------------------------------------------------

    kept_hashes = []

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    near_duplicates = 0

    hash_errors = 0

    comparisons = 0

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_rows = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, file_path in enumerate(
        image_files,
        start=1
    ):

        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{total_images} "
                f"| duplicates: "
                f"{near_duplicates} "
                f"| comparisons: "
                f"{comparisons}"
            )

        current_hash = calculate_phash(
            file_path
        )

        if current_hash is None:

            hash_errors += 1

            continue

        current_integer = (
            hash_to_integer(
                current_hash
            )
        )

        # ----------------------------------------------------
        # Find candidate buckets
        # ----------------------------------------------------

        candidate_entries = []

        candidate_ids = set()

        bucket_keys = get_bucket_keys(
            current_integer
        )

        for bucket_key in bucket_keys:

            entries = buckets.get(
                bucket_key,
                []
            )

            for entry in entries:

                entry_path = entry[1]

                if entry_path not in candidate_ids:

                    candidate_ids.add(
                        entry_path
                    )

                    candidate_entries.append(
                        entry
                    )

        # ----------------------------------------------------
        # Compare candidates
        # ----------------------------------------------------

        is_duplicate = False

        duplicate_path = None

        duplicate_distance = None

        for (
            stored_hash,
            stored_path
        ) in candidate_entries:

            comparisons += 1

            distance = (
                current_hash -
                stored_hash
            )

            if (
                distance
                <= PHASH_THRESHOLD
            ):

                is_duplicate = True

                duplicate_path = (
                    stored_path
                )

                duplicate_distance = (
                    distance
                )

                break

        # ----------------------------------------------------
        # Duplicate found
        # ----------------------------------------------------

        if is_duplicate:

            destination = (
                move_near_duplicate(
                    file_path
                )
            )

            if destination:

                near_duplicates += 1

                report_rows.append({

                    "duplicate":

                        str(destination),

                    "kept_image":

                        str(duplicate_path),

                    "phash_distance":

                        duplicate_distance

                })

            continue

        # ----------------------------------------------------
        # Keep image
        # ----------------------------------------------------

        kept_hashes.append(
            (
                current_hash,
                file_path
            )
        )

        # ----------------------------------------------------
        # Add to buckets
        # ----------------------------------------------------

        entry = (current_hash, file_path)

        for bucket_key in bucket_keys:

            if bucket_key not in buckets:

                buckets[bucket_key] = []

            buckets[bucket_key].append(
                entry
            )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=[
                "duplicate",
                "kept_image",
                "phash_distance"
            ]

        )

        writer.writeheader()

        writer.writerows(
            report_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    remaining = (
        total_images -
        near_duplicates
    )

    print()

    print("=" * 60)
    print("NEAR-DUPLICATE DETECTION COMPLETE")
    print("=" * 60)

    print(
        f"Images analyzed : "
        f"{total_images}"
    )

    print(
        f"Near duplicates  : "
        f"{near_duplicates}"
    )

    print(
        f"Hash errors      : "
        f"{hash_errors}"
    )

    print(
        f"Hash comparisons : "
        f"{comparisons}"
    )

    print(
        f"Images remaining : "
        f"{remaining}"
    )

    print()

    print(
        "Near-duplicates:"
    )

    print(
        f"  {NEAR_DUPLICATE_DIR}"
    )

    print()

    print(
        "Report:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 60)
    print("HISTORICAL DATASET CLEANING")
    print("=" * 60)

    print()

    # STEP 1
    create_directories()

    # STEP 2
    remove_rejected_images()

    # STEP 3
    find_near_duplicates()

    print()

    print("=" * 60)
    print("CLEANING FINISHED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()