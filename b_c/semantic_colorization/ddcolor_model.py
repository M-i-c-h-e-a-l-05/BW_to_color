"""
DDColor wrapper for Semantic Colorization Project

Uses the official DDColor repository.
"""

from pathlib import Path
import cv2
import torch

from ddcolor.model import DDColor
from ddcolor.pipeline import (
    build_ddcolor_model,
    ColorizationPipeline,
)


class DDColorModel:
    def __init__(
        self,
        checkpoint,
        input_size=512,
        model_size="tiny",
        decoder_type="MultiScaleColorDecoder",
        device="cpu",
    ):
        self.device = torch.device(device)

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint}"
            )

        self.model = build_ddcolor_model(
            DDColor,
            model_path=str(checkpoint),
            input_size=input_size,
            model_size=model_size,
            decoder_type=decoder_type,
            device=self.device,
        )

        self.pipeline = ColorizationPipeline(
            self.model,
            input_size=input_size,
            device=self.device,
        )

    def colorize(self, image):
        """
        Parameters
        ----------
        image : numpy.ndarray
            RGB or BGR uint8 image

        Returns
        -------
        numpy.ndarray
            BGR uint8 image
        """

        if image is None:
            raise ValueError("Input image is None.")

        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return self.pipeline.process(image)

    def __call__(self, image):
        return self.colorize(image)