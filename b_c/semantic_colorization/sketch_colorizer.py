"""
Sketch Colorizer

Colorizes line-art / pencil sketches by first converting the sparse
line drawing into a pseudo-photographic grayscale image (so DDColor
has shading information to work with), then running it through the
existing DDColorModel.
"""

import cv2
import numpy as np

from .ddcolor_model import DDColorModel


class SketchColorizer:

    def __init__(
        self,
        checkpoint,
        input_size=512,
        model_size="tiny",
        decoder_type="MultiScaleColorDecoder",
        device="cpu",
        blur_strength=15,
    ):
        print("=" * 60)
        print("Loading Sketch Colorizer")
        print("=" * 60)

        self.model = DDColorModel(
            checkpoint=checkpoint,
            input_size=input_size,
            model_size=model_size,
            decoder_type=decoder_type,
            device=device,
        )

        # Must be odd for cv2.GaussianBlur
        self.blur_strength = blur_strength | 1

        print("Sketch Colorizer Ready")
        print("=" * 60)

    def _to_pseudo_photo(self, image):
        """
        Line art has almost no shading, so DDColor (trained on real
        photographs) tends to leave it desaturated. Softening the lines
        with a blur gives the network gradient/shading cues to key its
        colorization off of, while the line structure is preserved
        underneath via the multiply blend.
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(
            gray, (self.blur_strength, self.blur_strength), 0
        )

        # Multiply-blend the blurred shading back with the sharp lines
        # so edges stay crisp instead of washing out.
        shaded = cv2.multiply(
            gray.astype(np.float32) / 255.0,
            blurred.astype(np.float32) / 255.0,
        )
        shaded = np.clip(shaded * 255.0, 0, 255).astype(np.uint8)

        return cv2.cvtColor(shaded, cv2.COLOR_GRAY2BGR)

    def colorize(self, image):
        """
        Colorize a black-and-white line-art sketch.

        Parameters
        ----------
        image : numpy.ndarray
            RGB, BGR, or single-channel uint8 sketch.

        Returns
        -------
        numpy.ndarray
            BGR uint8 colorized image.
        """

        if image is None:
            raise ValueError("Input image is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(image)}"
            )

        if image.size == 0:
            raise ValueError("Input image is empty.")

        pseudo_photo = self._to_pseudo_photo(image)

        return self.model.colorize(pseudo_photo)

    def __call__(self, image):
        return self.colorize(image)