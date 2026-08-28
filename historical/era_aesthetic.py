"""
era_aesthetic.py
=================

Historical Palette Adapter.

Applies a period-accurate color grade to an already-colorized BGR
uint8 image, based on the era predicted by EraClassifier (or a
manually chosen era from the GUI).

Each era is graded with three ingredients, all standard photographic
color-grading building blocks:
    1. Saturation scale   (from ERA_METADATA[era]["saturation"])
    2. A BGR channel tint  (per-channel gain, mimics film-stock color response)
    3. A tone curve         (contrast/gamma, mimics print/development characteristics)

Author:
    Micheal Leveiro Project
"""

import cv2
import numpy as np

from historical.era_classifier import ERA_LABELS, ERA_METADATA


# ------------------------------------------------------------
# Per-era grading recipes
#
# tint   : (B, R) multiplicative gain applied to the B and R channels
#          of the BGR image (G is left as the pivot channel).
# gamma  : >1 brightens midtones, <1 darkens/adds contrast.
# contrast: simple linear contrast around mid-gray (128).
# ------------------------------------------------------------

ERA_GRADES = {

    "1900": {  # sepia
        "tint": (0.75, 1.15),
        "gamma": 0.90,
        "contrast": 1.05,
    },

    "1920": {  # vintage
        "tint": (0.85, 1.08),
        "gamma": 0.95,
        "contrast": 1.00,
    },

    "1950": {  # kodachrome
        "tint": (1.05, 1.12),
        "gamma": 1.05,
        "contrast": 1.15,
    },

    "1960": {  # vivid
        "tint": (1.08, 1.10),
        "gamma": 1.05,
        "contrast": 1.20,
    },

    "1970": {  # warm
        "tint": (0.90, 1.15),
        "gamma": 1.00,
        "contrast": 1.05,
    },

    "Modern": {  # natural
        "tint": (1.00, 1.00),
        "gamma": 1.00,
        "contrast": 1.00,
    },

    "WWII": {  # muted
        "tint": (0.95, 0.95),
        "gamma": 0.92,
        "contrast": 0.95,
    },
}


class EraAesthetic:
    """
    Applies era-specific color grading to a colorized image.
    """

    def __init__(self):
        self.eras = list(ERA_LABELS)

    # -----------------------------------------------------
    # List available eras (for a GUI dropdown)
    # -----------------------------------------------------

    def list_eras(self):
        return list(self.eras)

    # -----------------------------------------------------
    # Palette description utility
    # -----------------------------------------------------

    def describe(self, era):
        if era not in ERA_METADATA:
            raise ValueError(f"Unknown era: {era}")
        return ERA_METADATA[era]

    # -----------------------------------------------------
    # Core grading routine
    # -----------------------------------------------------

    def apply(self, image, era):
        """
        Apply the historical color grade for `era` to a BGR uint8 image.

        Parameters
        ----------
        image : numpy.ndarray
            BGR uint8 image.
        era : str
            One of ERA_LABELS (e.g. "1920", "WWII", "Modern").

        Returns
        -------
        numpy.ndarray
            BGR uint8 image with the era's color grade applied.
        """

        if era not in ERA_GRADES:
            raise ValueError(
                f"Unknown era '{era}'. Expected one of: {self.eras}"
            )

        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty.")

        grade = ERA_GRADES[era]
        saturation = ERA_METADATA[era]["saturation"]

        graded = image.astype(np.float32)

        # 1. Channel tint (B and R gain, G is the pivot channel)
        b_gain, r_gain = grade["tint"]
        graded[:, :, 0] *= b_gain
        graded[:, :, 2] *= r_gain
        graded = np.clip(graded, 0, 255)

        # 2. Contrast around mid-gray
        contrast = grade["contrast"]
        graded = (graded - 128.0) * contrast + 128.0
        graded = np.clip(graded, 0, 255)

        # 3. Gamma curve
        gamma = grade["gamma"]
        normalized = graded / 255.0
        graded = np.power(normalized, 1.0 / gamma) * 255.0
        graded = np.clip(graded, 0, 255).astype(np.uint8)

        # 4. Era-accurate saturation
        hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return graded

    def __call__(self, image, era):
        return self.apply(image, era)