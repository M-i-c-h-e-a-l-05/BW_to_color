"""
Realtime Conditional Image Colorization
======================================

Pipeline

Grayscale Image
        │
        ▼
Base Zhang Colorizer
        │
        ▼
ADE20K Semantic Segmentation
        │
        ▼
Detect Objects
        │
        ▼
Apply User Selected Colors
        │
        ▼
Blend with AI Prediction
        │
        ▼
Final Image

Author:
Modified for Conditional Image Colorization
"""

import os
import sys

import cv2 as cv
import numpy as np

from typing import Dict
from typing import Tuple
from typing import List

from PIL import Image

# ---------------------------------------------------------
# Load the original Zhang colorizer
# ---------------------------------------------------------

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model1"
    )
)

from colorize_pretrained import load_colorizer

# ---------------------------------------------------------
# Local modules
# ---------------------------------------------------------

from segmenter import Segmenter

from class_colors import (
    get_color,
    set_user_colors,
    reset_user_colors,
)

# ---------------------------------------------------------
# Main Class
# ---------------------------------------------------------


class RealtimeColorizer:

    def __init__(
        self,
        base_model_dir="../model1/models",
        blend_strength=0.75,
        edge_feather=5,
        saturation_boost=1.8
    ):

        print("=" * 60)
        print("Loading Conditional Colorization Pipeline")
        print("=" * 60)

        self.device = "cpu"

        # -------------------------------------------------

        print("[1/3] Loading Zhang Colorizer...")

        self.base_net = load_colorizer(
            base_model_dir
        )

        # -------------------------------------------------

        print("[2/3] Loading ADE20K Segmenter...")

        self.segmenter = Segmenter()

        # -------------------------------------------------

        print("[3/3] Initializing Parameters...")

        self.blend_strength = blend_strength

        self.edge_feather = edge_feather

        self.saturation_boost = saturation_boost

        print()

        print("Pipeline Ready")

        print("=" * 60)

    # -----------------------------------------------------

    def set_user_colors(
        self,
        color_dictionary
    ):
        """
        Called by the GUI.

        Example

        {
            "sky":(-0.5,-0.8),

            "road":(0.0,0.0),

            "grass":(-0.7,0.5)
        }
        """

        set_user_colors(
            color_dictionary
        )

    # -----------------------------------------------------

    def reset_colors(self):

        reset_user_colors()

    # -----------------------------------------------------

    def detected_objects(
        self,
        class_map
    ):

        """
        Returns

        [
            "sky",
            "road",
            "car"
        ]
        """

        return self.segmenter.detected_classes(
            class_map
        )

    # -----------------------------------------------------

    def create_mask(
        self,
        class_map,
        class_name
    ):

        return self.segmenter.class_mask(
            class_map,
            class_name
        )

    # -----------------------------------------------------

    def feather_mask(
        self,
        mask
    ):

        mask = mask.astype(np.float32)

        if self.edge_feather > 0:

            mask = cv.GaussianBlur(

                mask,

                (0, 0),

                self.edge_feather

            )

        return mask

    # -----------------------------------------------------

    def lab_colour(
        self,
        class_name
    ):

        """
        Returns

        a,b

        in OpenCV scale
        """

        a_norm, b_norm = get_color(
            class_name
        )

        a = a_norm * 127.0

        b = b_norm * 127.0

        return a, b
    # -----------------------------------------------------
    # Base Zhang Colorizer
    # -----------------------------------------------------

    def _base_colorize_ab(
        self,
        bgr_image
    ):
        """
        Runs the original Zhang et al. model and returns only
        the predicted a,b channels.
        """

        scaled = bgr_image.astype(np.float32) / 255.0

        lab = cv.cvtColor(
            scaled,
            cv.COLOR_BGR2LAB
        )

        resized = cv.resize(
            lab,
            (224, 224)
        )

        L = resized[:, :, 0]

        L -= 50

        self.base_net.setInput(

            cv.dnn.blobFromImage(L)

        )

        ab = self.base_net.forward()[0]

        ab = ab.transpose((1, 2, 0))

        ab = cv.resize(

            ab,

            (

                bgr_image.shape[1],

                bgr_image.shape[0]

            )

        )

        ab *= self.saturation_boost

        ab = np.clip(

            ab,

            -127,

            127

        )

        return ab

    # -----------------------------------------------------
    # Extract L Channel
    # -----------------------------------------------------

    def _prepare_lab(
        self,
        bgr_image
    ):

        scaled = bgr_image.astype(

            np.float32

        ) / 255.0

        lab = cv.cvtColor(

            scaled,

            cv.COLOR_BGR2LAB

        )

        L = lab[:, :, 0]

        return L

    # -----------------------------------------------------
    # Blend Two a,b Colours
    # -----------------------------------------------------

    def _blend_lab(
        self,
        original,
        target,
        strength
    ):

        return (

            (1.0 - strength)

            * original

            +

            strength

            * target

        )

    # -----------------------------------------------------
    # Apply Colour To One Object
    # -----------------------------------------------------

    def _apply_user_colour(
        self,
        ab_image,
        class_map,
        class_name
    ):
        """
        Applies the selected colour only to one semantic class.
        """

        try:

            mask = self.create_mask(

                class_map,

                class_name

            )

        except Exception:

            return ab_image

        if mask.sum() == 0:

            return ab_image

        mask = self.feather_mask(

            mask

        )

        a_target, b_target = self.lab_colour(

            class_name

        )

        original_a = ab_image[:, :, 0]

        original_b = ab_image[:, :, 1]

        new_a = self._blend_lab(

            original_a,

            a_target,

            self.blend_strength

        )

        new_b = self._blend_lab(

            original_b,

            b_target,

            self.blend_strength

        )

        ab_image[:, :, 0] = (

            original_a * (1 - mask)

            +

            new_a * mask

        )

        ab_image[:, :, 1] = (

            original_b * (1 - mask)

            +

            new_b * mask

        )

        return ab_image

    # -----------------------------------------------------
    # Apply Colours To Every Detected Object
    # -----------------------------------------------------

    def apply_semantic_colours(
        self,
        ab_image,
        class_map
    ):

        detected = self.detected_objects(

            class_map

        )

        for cls in detected:

            ab_image = self._apply_user_colour(

                ab_image,

                class_map,

                cls

            )

        return ab_image

    # -----------------------------------------------------
    # Main Processing Pipeline
    # -----------------------------------------------------

    def process_frame(
        self,
        bgr_image
    ):
        """
        Conditional Image Colorization Pipeline

        Input
        -----
        BGR uint8 image

        Output
        ------
        Colorized BGR uint8 image
        """

        # ---------------------------------------------
        # Original L channel
        # ---------------------------------------------

        L = self._prepare_lab(

            bgr_image

        )

        # ---------------------------------------------
        # Base AI Colorization
        # ---------------------------------------------

        ab = self._base_colorize_ab(

            bgr_image

        )

        # ---------------------------------------------
        # Semantic Segmentation
        # ---------------------------------------------

        rgb = cv.cvtColor(

            bgr_image,

            cv.COLOR_BGR2RGB

        )

        class_map = self.segmenter.segment(

            rgb

        )

        # ---------------------------------------------
        # Store detected objects
        # ---------------------------------------------

        self.last_detected_objects = (

            self.detected_objects(

                class_map

            )

        )

        # ---------------------------------------------
        # Apply User Colours
        # ---------------------------------------------

        ab = self.apply_semantic_colours(

            ab,

            class_map

        )

        # ---------------------------------------------
        # Rebuild LAB image
        # ---------------------------------------------

        lab = np.zeros(

            (

                bgr_image.shape[0],

                bgr_image.shape[1],

                3

            ),

            dtype=np.float32

        )

        lab[:, :, 0] = L

        lab[:, :, 1] = ab[:, :, 0]

        lab[:, :, 2] = ab[:, :, 1]

        # ---------------------------------------------
        # Convert LAB → BGR
        # ---------------------------------------------

        result = cv.cvtColor(

            lab,

            cv.COLOR_LAB2BGR

        )

        result = np.clip(

            result,

            0,

            1

        )

        result = (

            result

            * 255

        ).astype(

            np.uint8

        )

        return result

    # -----------------------------------------------------
    # Process Single Image
    # -----------------------------------------------------

    def process_image(

        self,

        image_path

    ):

        img = cv.imread(

            image_path

        )

        if img is None:

            raise FileNotFoundError(

                image_path

            )

        return self.process_frame(

            img

        )

    # -----------------------------------------------------
    # Get Detected Classes
    # -----------------------------------------------------

    def get_detected_objects(

        self

    ):

        if hasattr(

            self,

            "last_detected_objects"

        ):

            return self.last_detected_objects

        return []

    # -----------------------------------------------------
    # Set Blend Strength
    # -----------------------------------------------------

    def set_blend_strength(

        self,

        value

    ):

        value = float(

            value

        )

        value = max(

            0,

            min(

                1,

                value

            )

        )

        self.blend_strength = value

    # -----------------------------------------------------
    # Set Saturation
    # -----------------------------------------------------

    def set_saturation(

        self,

        value

    ):

        self.saturation_boost = float(

            value

        )

    # -----------------------------------------------------
    # Reset Everything
    # -----------------------------------------------------

    def reset(

        self

    ):

        self.reset_colors()

        self.last_detected_objects = []

        # -----------------------------------------------------
    # Process Video
    # -----------------------------------------------------

    def process_video_file(
        self,
        input_path,
        output_path,
        frame_skip=0,
        max_frames=None
    ):

        cap = cv.VideoCapture(input_path)

        if not cap.isOpened():

            raise FileNotFoundError(input_path)

        fps = cap.get(cv.CAP_PROP_FPS)

        if fps <= 0:
            fps = 24

        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        writer = cv.VideoWriter(

            output_path,

            cv.VideoWriter_fourcc(*"mp4v"),

            fps,

            (width, height)

        )

        frame_index = 0

        last_frame = None

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            if (

                max_frames is not None

                and

                frame_index >= max_frames

            ):

                break

            if (

                frame_skip > 0

                and

                frame_index % (frame_skip + 1) != 0

                and

                last_frame is not None

            ):

                result = last_frame

            else:

                result = self.process_frame(frame)

                last_frame = result

            writer.write(result)

            frame_index += 1

            if frame_index % 10 == 0:

                print(

                    f"Processed {frame_index} frames"

                )

        cap.release()

        writer.release()

        print()

        print("Video Saved")

        print(output_path)

if __name__ == "__main__":

    print("=" * 60)
    print("Realtime Conditional Image Colorizer")
    print("=" * 60)

    pipeline = RealtimeColorizer()

    print()

    print("Model Loaded Successfully")

    print()

    print("Ready for")

    print(" • Images")

    print(" • Webcam")

    print(" • Video Processing")

    print("=" * 60)