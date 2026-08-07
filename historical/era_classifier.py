"""
era_classifier.py
=================

Historical Era Classification Module

This module predicts the historical era represented by a grayscale
or RGB photograph before colorization.

The predicted era is later used by the Historical Palette Adapter
to produce historically accurate colors.

Architecture
------------
Backbone:
    ResNet18 (PyTorch)

Input:
    RGB Image (224 x 224)

Output:
    {
        "era": "WWII",
        "confidence": 0.94,
        "class_id": 2
    }

Author:
    Micheal Leveiro Project
"""

import os
from typing import Dict
from typing import Tuple

import cv2
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import resnet18


# ------------------------------------------------------------
# Historical Era Labels
# ------------------------------------------------------------

ERA_LABELS = [

    "1900s",

    "1920s",

    "WWII",

    "1950s",

    "1960s",

    "1970s",

    "Modern"

]

NUM_CLASSES = len(ERA_LABELS)


# ------------------------------------------------------------
# Default model location
# ------------------------------------------------------------

DEFAULT_MODEL_PATH = os.path.join(

    os.path.dirname(__file__),

    "weights",

    "era_classifier_resnet18.pth"

)


# ------------------------------------------------------------
# ImageNet normalization
# ------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------
# Image Transform
# ------------------------------------------------------------

IMAGE_TRANSFORM = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=IMAGENET_MEAN,

        std=IMAGENET_STD,

    )

])


# ------------------------------------------------------------
# Historical Metadata
# ------------------------------------------------------------

ERA_METADATA = {

    "1900s":{

        "war_period":False,

        "expected_palette":"sepia",

        "color_temperature":"warm",

        "contrast":"low",

        "saturation":0.55,

    },

    "1920s":{

        "war_period":False,

        "expected_palette":"vintage",

        "color_temperature":"warm",

        "contrast":"medium",

        "saturation":0.65,

    },

    "WWII":{

        "war_period":True,

        "expected_palette":"muted",

        "color_temperature":"warm",

        "contrast":"high",

        "saturation":0.70,

    },

    "1950s":{

        "war_period":False,

        "expected_palette":"kodachrome",

        "color_temperature":"neutral",

        "contrast":"high",

        "saturation":1.10,

    },

    "1960s":{

        "war_period":False,

        "expected_palette":"vivid",

        "color_temperature":"neutral",

        "contrast":"medium",

        "saturation":1.20,

    },

    "1970s":{

        "war_period":False,

        "expected_palette":"warm",

        "color_temperature":"warm",

        "contrast":"medium",

        "saturation":1.05,

    },

    "Modern":{

        "war_period":False,

        "expected_palette":"natural",

        "color_temperature":"neutral",

        "contrast":"high",

        "saturation":1.00,

    }

}


# ============================================================
# Era Classifier
# ============================================================

class EraClassifier:

    """
    Historical Era Classifier.

    Predicts the most likely decade or historical period
    represented by an image.

    Uses a fine-tuned ResNet18 backbone.
    """

    def __init__(

        self,

        checkpoint_path: str = DEFAULT_MODEL_PATH,

        device: str = None,

    ):

        # -----------------------------------------------
        # Device
        # -----------------------------------------------

        if device is None:

            self.device = (

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        else:

            self.device = device

        print()

        print("=" * 60)

        print("Loading Historical Era Classifier")

        print("=" * 60)

        print(f"Device : {self.device}")

        # -----------------------------------------------
        # Build Model
        # -----------------------------------------------

        self.model = resnet18(

            weights=None

        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(

            in_features,

            NUM_CLASSES

        )

        self.model.to(self.device)

        # -----------------------------------------------
        # Load checkpoint
        # -----------------------------------------------

        if not os.path.exists(checkpoint_path):

            raise FileNotFoundError(

                f"\nHistorical Era checkpoint not found:\n"

                f"{checkpoint_path}\n\n"

                "Train the classifier first "

                "or download pretrained weights."

            )

        print()

        print("Loading weights...")

        checkpoint = torch.load(

            checkpoint_path,

            map_location=self.device,

        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                checkpoint = checkpoint["model_state_dict"]

        self.model.load_state_dict(checkpoint)

        self.model.eval()

        print("✓ Era Classifier Loaded")

        print("=" * 60)


        # ------------------------------------------------------------
    # Image Preprocessing
    # ------------------------------------------------------------

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV image into a normalized tensor suitable
        for ResNet18 inference.

        Supports:
            - Grayscale images
            - RGB images
            - BGR images (OpenCV)
        """

        if image is None:
            raise ValueError("Input image is None.")

        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        else:
            # OpenCV BGR → RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = IMAGE_TRANSFORM(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict:

        """
        Predict historical era.

        Returns
        -------
        {
            "era": "WWII",
            "confidence": 0.95,
            "class_id": 2,
            "top_predictions":[]
        }
        """

        tensor = self.preprocess_image(image)

        logits = self.model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

        predicted = predicted.item()

        confidence = confidence.item()

        top_prob, top_idx = torch.topk(
            probabilities,
            k=min(3, NUM_CLASSES)
        )

        top_predictions = []

        for p, idx in zip(
            top_prob[0],
            top_idx[0]
        ):

            top_predictions.append({

                "era": ERA_LABELS[idx.item()],

                "confidence": round(
                    float(p.item()),
                    4
                )

            })

        predicted_era = ERA_LABELS[predicted]

        metadata = ERA_METADATA[predicted_era]

        return {

            "era": predicted_era,

            "confidence": round(confidence,4),

            "class_id": predicted,

            "metadata": metadata,

            "top_predictions": top_predictions,

        }

    # ------------------------------------------------------------
    # Predict directly from file
    # ------------------------------------------------------------

    def predict_from_path(
        self,
        image_path: str,
    ) -> Dict:

        image = cv2.imread(image_path)

        if image is None:

            raise FileNotFoundError(

                f"Unable to read image:\n"

                f"{image_path}"

            )

        return self.predict(image)


        # ------------------------------------------------------------
    # Batch Prediction
    # ------------------------------------------------------------

    def predict_batch(self, images):
        """
        Predict historical eras for a list of OpenCV images.

        Parameters
        ----------
        images : list[np.ndarray]

        Returns
        -------
        list[dict]
        """

        results = []

        for image in images:
            results.append(self.predict(image))

        return results

    # ------------------------------------------------------------
    # Directory Prediction
    # ------------------------------------------------------------

    def predict_directory(
        self,
        directory,
        extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
    ):
        """
        Predict the era of every image inside a directory.

        Returns
        -------
        list[dict]
        """

        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)

        outputs = []

        for filename in sorted(os.listdir(directory)):

            if not filename.lower().endswith(extensions):
                continue

            path = os.path.join(directory, filename)

            try:

                prediction = self.predict_from_path(path)

                prediction["filename"] = filename

                outputs.append(prediction)

            except Exception as e:

                outputs.append({

                    "filename": filename,

                    "error": str(e)

                })

        return outputs

    # ------------------------------------------------------------
    # Metadata Utility
    # ------------------------------------------------------------

    def get_metadata(self, era):
        """
        Return metadata associated with an era.
        """

        if era not in ERA_METADATA:
            raise ValueError(f"Unknown era: {era}")

        return ERA_METADATA[era]

    # ------------------------------------------------------------
    # Palette Recommendation
    # ------------------------------------------------------------

    def recommend_palette(self, prediction):
        """
        Return the recommended palette name for the prediction.
        """

        return prediction["metadata"]["expected_palette"]

    # ------------------------------------------------------------
    # Pretty Print
    # ------------------------------------------------------------

    @staticmethod
    def print_prediction(result):
        """
        Nicely print prediction results.
        """

        print("\n========== Era Prediction ==========")

        print(f"Era        : {result['era']}")

        print(f"Confidence : {result['confidence']:.2%}")

        print()

        print("Metadata")

        for key, value in result["metadata"].items():

            print(f"  {key:20}: {value}")

        print()

        print("Top Predictions")

        for item in result["top_predictions"]:

            print(
                f"  {item['era']:10} "
                f"{item['confidence']:.2%}"
            )

        print("====================================")


if __name__ == "__main__":

    TEST_IMAGE = "sample.jpg"

    classifier = EraClassifier()

    result = classifier.predict_from_path(TEST_IMAGE)

    classifier.print_prediction(result)