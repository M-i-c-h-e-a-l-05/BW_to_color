"""
restructure_dataset.py

Restructures the cleaned historical image dataset
into the final 7 classification classes:

1900
1920
WWII
1950
1960
1970
Modern

Existing folders:

1900_1919 -> 1900
1920_1939 -> 1920
WWII      -> WWII
1950s     -> 1950
1960s     -> 1960
1970s     -> 1970
Modern    -> Modern

Unknown images are NOT included.

Author: Micheal Leveiro
"""

from pathlib import Path
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".tif",
    ".tiff"
}


# ============================================================
# FOLDER MAPPING
# ============================================================

FOLDER_MAPPING = {

    "1900_1919": "1900",

    "1920_1939": "1920",

    "WWII": "WWII",

    "1950s": "1950",

    "1960s": "1960",

    "1970s": "1970",

    "Modern": "Modern"
}


# ============================================================
# CREATE FINAL DIRECTORIES
# ============================================================

def create_target_directories():

    print("=" * 60)
    print("CREATING FINAL DATASET DIRECTORIES")
    print("=" * 60)

    for target_folder in set(
        FOLDER_MAPPING.values()
    ):

        target_path = (
            DATASET_DIR /
            target_folder
        )

        target_path.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Created/verified: "
            f"{target_path}"
        )

    print()


# ============================================================
# GET IMAGE FILES
# ============================================================

def get_images(folder):

    if not folder.exists():
        return []

    return [
        file_path
        for file_path in folder.iterdir()
        if (
            file_path.is_file()
            and
            file_path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]


# ============================================================
# GENERATE UNIQUE FILENAME
# ============================================================

def get_unique_destination(
    destination_folder,
    filename
):

    destination = (
        destination_folder /
        filename
    )

    counter = 1

    while destination.exists():

        stem = destination.stem

        suffix = destination.suffix

        destination = (
            destination_folder /
            f"{stem}_{counter}{suffix}"
        )

        counter += 1

    return destination


# ============================================================
# MOVE IMAGES
# ============================================================

def merge_folders():

    total_moved = 0

    print("=" * 60)
    print("MERGING HISTORICAL DATASET")
    print("=" * 60)

    for source_folder, target_folder in (
        FOLDER_MAPPING.items()
    ):

        source_path = (
            DATASET_DIR /
            source_folder
        )

        target_path = (
            DATASET_DIR /
            target_folder
        )

        if not source_path.exists():

            print(
                f"\nSource not found: "
                f"{source_folder}"
            )

            continue

        images = get_images(
            source_path
        )

        print(
            f"\n{source_folder}"
            f" -> "
            f"{target_folder}"
        )

        print(
            f"Images: {len(images)}"
        )

        moved = 0

        for image_path in images:

            destination = (
                get_unique_destination(
                    target_path,
                    image_path.name
                )
            )

            try:

                shutil.move(
                    str(image_path),
                    str(destination)
                )

                moved += 1
                total_moved += 1

            except Exception as e:

                print(
                    f"ERROR: "
                    f"{image_path}"
                )

                print(
                    f"Reason: {e}"
                )

        print(
            f"Moved: {moved}"
        )

    print()

    print("=" * 60)
    print("MERGING COMPLETE")
    print("=" * 60)

    print(
        f"Total images moved: "
        f"{total_moved}"
    )

    print()


# ============================================================
# REMOVE EMPTY OLD DIRECTORIES
# ============================================================

def remove_empty_directories():

    print("=" * 60)
    print("REMOVING EMPTY OLD DIRECTORIES")
    print("=" * 60)

    for source_folder in FOLDER_MAPPING:

        source_path = (
            DATASET_DIR /
            source_folder
        )

        if not source_path.exists():
            continue

        try:

            # Only remove if completely empty.

            if not any(
                source_path.iterdir()
            ):

                source_path.rmdir()

                print(
                    f"Removed: "
                    f"{source_path}"
                )

            else:

                print(
                    f"Not empty, kept: "
                    f"{source_path}"
                )

        except Exception as e:

            print(
                f"ERROR: "
                f"{source_path} | {e}"
            )

    print()


# ============================================================
# SHOW FINAL DISTRIBUTION
# ============================================================

def show_final_distribution():

    print("=" * 60)
    print("FINAL DATASET DISTRIBUTION")
    print("=" * 60)

    total = 0

    final_classes = [
        "1900",
        "1920",
        "WWII",
        "1950",
        "1960",
        "1970",
        "Modern"
    ]

    for class_name in final_classes:

        folder = (
            DATASET_DIR /
            class_name
        )

        count = len(
            get_images(folder)
        )

        total += count

        print(
            f"{class_name:<10} : "
            f"{count}"
        )

    print("-" * 30)

    print(
        f"{'TOTAL':<10} : "
        f"{total}"
    )

    print()

    unknown_folder = (
        DATASET_DIR /
        "Unknown"
    )

    unknown_count = len(
        get_images(
            unknown_folder
        )
    )

    if unknown_count:

        print(
            f"Unknown images retained: "
            f"{unknown_count}"
        )

        print(
            "Unknown was NOT merged into "
            "any class."
        )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("HISTORICAL IMAGE DATASET RESTRUCTURING")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Create final folders
    # --------------------------------------------------------

    create_target_directories()

    # --------------------------------------------------------
    # Move images into final classes
    # --------------------------------------------------------

    merge_folders()

    # --------------------------------------------------------
    # Remove old folders if empty
    # --------------------------------------------------------

    remove_empty_directories()

    # --------------------------------------------------------
    # Display final dataset
    # --------------------------------------------------------

    show_final_distribution()

    print("=" * 60)
    print("DATASET RESTRUCTURING FINISHED")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

