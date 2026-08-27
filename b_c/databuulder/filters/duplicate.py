"""
filters/duplicate.py

Duplicate Image Removal using Perceptual Hash (pHash)

Author: Micheal Leveiro
"""

from pathlib import Path

from PIL import Image
import imagehash


class DuplicateFilter:

    def __init__(

        self,

        hash_size=16,

        threshold=5

    ):

        self.hash_size = hash_size

        self.threshold = threshold

        self.hashes = {}

        self.checked = 0

        self.removed = 0

    #########################################################

    def compute_hash(self, image_path):

        try:

            img = Image.open(image_path)

            return imagehash.phash(

                img,

                hash_size=self.hash_size

            )

        except Exception:

            return None

    #########################################################

    def run(self, dataset_folder):

        dataset_folder = Path(dataset_folder)

        images = [

            p

            for p in dataset_folder.rglob("*")

            if p.suffix.lower()

            in [

                ".jpg",

                ".jpeg",

                ".png"

            ]

        ]

        total = len(images)

        print(f"\nChecking {total} images for duplicates...\n")

        for image in images:

            self.checked += 1

            current_hash = self.compute_hash(image)

            if current_hash is None:

                continue

            duplicate = False

            for saved_hash in self.hashes:

                distance = current_hash - saved_hash

                if distance <= self.threshold:

                    duplicate = True

                    break

            if duplicate:

                image.unlink()

                self.removed += 1

            else:

                self.hashes[current_hash] = image

            if self.checked % 500 == 0:

                print(

                    f"{self.checked}/{total} checked"

                )

        print("\nDuplicate Removal Complete")

        print("--------------------------------")

        print("Checked :", self.checked)

        print("Removed :", self.removed)

        print("Remaining :", self.checked - self.removed)

        print("--------------------------------")