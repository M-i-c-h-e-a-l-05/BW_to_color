"""
Infrared Colorizer

Infrared/thermal and IR satellite bands don't map to "natural" colors
the way a B&W photo does, so this doesn't use DDColor. Instead it
applies a perceptual false-color colormap to the intensity data,
which is the standard way IR imagery is made visually interpretable.
"""

import cv2
import numpy as np


class InfraredColorizer:

    COLORMAPS = {
        "inferno": cv2.COLORMAP_INFERNO,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }

    def __init__(self, colormap="inferno", clahe_clip=2.0):
        if colormap not in self.COLORMAPS:
            raise ValueError(
                f"Unsupported colormap: {colormap}. "
                f"Choose from: {sorted(self.COLORMAPS)}"
            )

        self.colormap_name = colormap
        self.colormap = self.COLORMAPS[colormap]

        # Contrast-limited adaptive histogram equalization brings out
        # detail in low-contrast IR bands before the colormap is applied.
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def set_colormap(self, colormap):
        if colormap not in self.COLORMAPS:
            raise ValueError(
                f"Unsupported colormap: {colormap}. "
                f"Choose from: {sorted(self.COLORMAPS)}"
            )
        self.colormap_name = colormap
        self.colormap = self.COLORMAPS[colormap]

    def colorize(self, image):
        """
        Apply false-color rendering to a single- or multi-band
        infrared image.

        Parameters
        ----------
        image : numpy.ndarray
            Single-channel or multi-channel uint8/uint16/float IR image.

        Returns
        -------
        numpy.ndarray
            BGR uint8 false-color image.
        """

        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image)}"
            )

        if image.size == 0:
            raise ValueError("Input image is empty.")

        if len(image.shape) == 3 and image.shape[2] > 1:
            intensity = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            intensity = image.squeeze() if len(image.shape) == 3 else image

        if intensity.dtype != np.uint8:
            intensity = cv2.normalize(
                intensity, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

        enhanced = self.clahe.apply(intensity)

        return cv2.applyColorMap(enhanced, self.colormap)

    def __call__(self, image):
        return self.colorize(image)