"""
filters/quality.py

Quality Filter
"""

from pathlib import Path
from PIL import Image
import cv2
import numpy as np


class QualityFilter:

    def __init__(

        self,

        min_width=224,

        min_height=224,

        blur_threshold=50

    ):

        self.min_width = min_width

        self.min_height = min_height

        self.blur_threshold = blur_threshold

        self.removed = 0

        self.checked = 0

    #######################################################

    def variance_of_laplacian(self, image):

        return cv2.Laplacian(
            image,
            cv2.CV_64F
        ).var()

    #######################################################

    def blank_image(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        return np.std(gray) < 5

    #######################################################

    def check(self, path):

        try:

            img = Image.open(path)

            img.verify()

            img = Image.open(path)

        except Exception:

            return False

        width, height = img.size

        if width < self.min_width:

            return False

        if height < self.min_height:

            return False

        img_cv = cv2.imread(str(path))

        if img_cv is None:

            return False

        if self.blank_image(img_cv):

            return False

        blur = self.variance_of_laplacian(img_cv)

        if blur < self.blur_threshold:

            return False

        return True

    #######################################################

    def run(self, dataset_folder):

        dataset_folder = Path(dataset_folder)

        images = list(dataset_folder.rglob("*"))

        images = [

            p for p in images

            if p.suffix.lower()

            in [".jpg", ".jpeg", ".png"]

        ]

        total = len(images)

        print(f"\nChecking {total} images...\n")

        for image in images:

            self.checked += 1

            ok = self.check(image)

            if not ok:

                image.unlink()

                self.removed += 1

            if self.checked % 500 == 0:

                print(

                    f"{self.checked}/{total} checked"

                )

        print("\nQuality Filter Complete")

        print("---------------------------")

        print("Checked :", self.checked)

        print("Removed :", self.removed)

        print("Remaining :", self.checked - self.removed)

        print("---------------------------")