"""
B&W Photo Colorizer

Uses the existing DDColorModel to colorize
ordinary black-and-white photographs.
"""

import numpy as np

from .ddcolor_model import DDColorModel


class BWPhotoColorizer:

    def __init__(
        self,
        checkpoint,
        input_size=512,
        model_size="tiny",
        decoder_type="MultiScaleColorDecoder",
        device="cpu",
    ):
        print("=" * 60)
        print("Loading B&W Photo Colorizer")
        print("=" * 60)

        self.model = DDColorModel(
            checkpoint=checkpoint,
            input_size=input_size,
            model_size=model_size,
            decoder_type=decoder_type,
            device=device,
        )

        print("B&W Photo Colorizer Ready")
        print("=" * 60)

    def colorize(self, image):
        """
        Colorize an ordinary black-and-white photograph.

        Parameters
        ----------
        image : numpy.ndarray
            RGB or BGR uint8 image.

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

        return self.model.colorize(image)

    def __call__(self, image):
        return self.colorize(image)