import random
import shutil
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path(
    "/home/micheal-leveiro/Projects/BW_to_col/HistoricalDatasetBuilder/dataset"
)

DEST = Path(
    "/home/micheal-leveiro/Projects/BW_to_col/historical/dataset"
)

# How many images you want from each era
IMAGES_PER_ERA = {
    "1900": 1000,
    "1920": 1000,
    "WWII": 790,
    "1950": 1000,
    "1960": 1000,
    "1970": 1000,
    "Modern": 1000,
}

# 80% train / 20% validation
TRAIN_RATIO = 0.8

# Reproducible random selection
RANDOM_SEED = 42

# ============================================================
# SETUP
# ============================================================

random.seed(RANDOM_SEED)

train_dir = DEST / "train"
val_dir = DEST / "val"

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# PROCESS EACH ERA
# ============================================================

for era, desired_count in IMAGES_PER_ERA.items():

    source_dir = SOURCE / era

    if not source_dir.exists():
        print(f"\nWARNING: {source_dir} does not exist")
        continue

    # Get image files
    images = [
        f for f in source_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".tif",
            ".tiff"
        }
    ]

    print(f"\n{era}")
    print(f"Available : {len(images)}")
    print(f"Requested : {desired_count}")

    # --------------------------------------------------------
    # Check whether enough images exist
    # --------------------------------------------------------

    if len(images) < desired_count:

        print(
            f"WARNING: Only {len(images)} available. "
            f"Using all available images."
        )

        selected = images

    else:

        selected = random.sample(
            images,
            desired_count
        )

    # --------------------------------------------------------
    # Shuffle selected images
    # --------------------------------------------------------

    random.shuffle(selected)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_count = int(
        len(selected) * TRAIN_RATIO
    )

    train_images = selected[:train_count]
    val_images = selected[train_count:]

    # --------------------------------------------------------
    # Create era folders
    # --------------------------------------------------------

    era_train_dir = train_dir / era
    era_val_dir = val_dir / era

    era_train_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    era_val_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy training images
    # --------------------------------------------------------

    for image in train_images:

        shutil.copy2(
            image,
            era_train_dir / image.name
        )

    # --------------------------------------------------------
    # Copy validation images
    # --------------------------------------------------------

    for image in val_images:

        shutil.copy2(
            image,
            era_val_dir / image.name
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        f"Selected  : {len(selected)}"
    )

    print(
        f"Train     : {len(train_images)}"
    )

    print(
        f"Validation: {len(val_images)}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"\nTrain directory:")
print(train_dir)

print(f"\nValidation directory:")
print(val_dir)

print("\nDone.")