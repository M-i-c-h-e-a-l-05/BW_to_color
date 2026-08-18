"""
historical/era_aesthetic.py

Applies historical visual aesthetics based on the
predicted era.

The classifier determines the era.
This module determines how the final colorized
image should look.
"""

import cv2
import numpy as np


class EraAesthetic:

    def __init__(self):

        # --------------------------------------------------
        # Era-specific settings
        # --------------------------------------------------

        self.presets = {

            "1900": {
                "saturation": 0.55,
                "contrast": 0.85,
                "warmth": 18,
                "sepia": 0.45,
            },

            "1920": {
                "saturation": 0.65,
                "contrast": 0.90,
                "warmth": 14,
                "sepia": 0.30,
            },

            "1950": {
                "saturation": 1.05,
                "contrast": 1.10,
                "warmth": 4,
                "sepia": 0.05,
            },

            "1960": {
                "saturation": 1.20,
                "contrast": 1.05,
                "warmth": 0,
                "sepia": 0.00,
            },

            "1970": {
                "saturation": 1.05,
                "contrast": 1.00,
                "warmth": 8,
                "sepia": 0.05,
            },

            "Modern": {
                "saturation": 1.00,
                "contrast": 1.00,
                "warmth": 0,
                "sepia": 0.00,
            },

            "WWII": {
                "saturation": 0.70,
                "contrast": 1.12,
                "warmth": 10,
                "sepia": 0.12,
            }
        }

    # ==================================================
    # Saturation
    # ==================================================

    def adjust_saturation(
        self,
        image,
        amount
    ):

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        hsv = hsv.astype(np.float32)

        hsv[:, :, 1] *= amount

        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1],
            0,
            255
        )

        hsv = hsv.astype(np.uint8)

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

    # ==================================================
    # Contrast
    # ==================================================

    def adjust_contrast(
        self,
        image,
        amount
    ):

        result = image.astype(
            np.float32
        )

        result = (
            (result - 127.5)
            * amount
            + 127.5
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    # ==================================================
    # Warmth
    # ==================================================

    def adjust_warmth(
        self,
        image,
        amount
    ):

        result = image.astype(
            np.float32
        )

        # BGR
        result[:, :, 0] -= amount * 0.5
        result[:, :, 1] += amount * 0.15
        result[:, :, 2] += amount

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    # ==================================================
    # Sepia
    # ==================================================

    def apply_sepia(
        self,
        image,
        strength
    ):

        if strength <= 0:
            return image

        image_float = image.astype(
            np.float32
        )

        # BGR sepia matrix
        sepia = np.zeros_like(
            image_float
        )

        sepia[:, :, 0] = (
            image_float[:, :, 0] * 0.272
            + image_float[:, :, 1] * 0.534
            + image_float[:, :, 2] * 0.131
        )

        sepia[:, :, 1] = (
            image_float[:, :, 0] * 0.349
            + image_float[:, :, 1] * 0.686
            + image_float[:, :, 2] * 0.168
        )

        sepia[:, :, 2] = (
            image_float[:, :, 0] * 0.393
            + image_float[:, :, 1] * 0.769
            + image_float[:, :, 2] * 0.189
        )

        sepia = np.clip(
            sepia,
            0,
            255
        )

        result = (
            image_float * (1 - strength)
            + sepia * strength
        )

        return np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

    # ==================================================
    # Apply Era
    # ==================================================

    def apply(
        self,
        image,
        era
    ):

        if era not in self.presets:

            era = "Modern"

        settings = self.presets[era]

        result = image.copy()

        # 1. Saturation
        result = self.adjust_saturation(
            result,
            settings["saturation"]
        )

        # 2. Contrast
        result = self.adjust_contrast(
            result,
            settings["contrast"]
        )

        # 3. Warm/cool tone
        if settings["warmth"] != 0:

            result = self.adjust_warmth(
                result,
                settings["warmth"]
            )

        # 4. Sepia
        if settings["sepia"] > 0:

            result = self.apply_sepia(
                result,
                settings["sepia"]
            )

        return result

    # ==================================================
    # Get Settings
    # ==================================================

    def get_settings(
        self,
        era
    ):

        return self.presets.get(
            era,
            self.presets["Modern"]
        )