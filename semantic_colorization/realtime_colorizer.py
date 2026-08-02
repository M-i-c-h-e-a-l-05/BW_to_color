"""
Real-time multi-object colorization pipeline.

This is the key piece that builds directly on the earlier project rather
than starting fresh: for the "background" colorization it reuses the
pretrained Zhang et al. colorizer (colorize_pretrained.py) already built
and verified in the base project. Semantic segmentation is layered on top
to identify specific object regions, which are then overridden with the
predetermined per-class colors from class_colors.py instead of the
generic AI-predicted color.

Pipeline per frame:
    1. Run the base AI colorizer over the whole frame (reused from the
       earlier project) -> generic colorized guess.
    2. Run semantic segmentation -> per-pixel class map.
    3. For pixels belonging to a class with a predetermined color,
       override the base colorizer's a,b output with that fixed color.
    4. Recombine with the original L channel -> final frame.

The L (lightness) channel is never altered, which is what keeps object
edges and detail sharp even where colors are overridden.
"""
import os
import sys
import numpy as np
import cv2 as cv

# Import the base colorizer from the parent project directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model1"))
from model1.colorize_pretrained import load_colorizer  # noqa: E402

from class_colors import build_index_to_color, VOC_CLASSES
from segmenter import Segmenter


class RealtimeColorizer:
    def __init__(self, base_model_dir: str = "../model1/models", edge_feather: int = 3,
                 saturation_boost: float = 2.0):
        """
        base_model_dir: path to the pretrained Zhang et al. model files
                         from the base project (see ../download_pretrained_model.py)
        edge_feather:    pixel radius for softening class-region edges so
                         color overrides blend smoothly rather than
                         showing a hard cutout outline
        saturation_boost: multiplier applied to the base colorizer's a,b
                         output. The Zhang et al. (2016) model is known to
                         produce quite desaturated ("safe"/muted) colors --
                         without this boost, background regions can look
                         nearly gray next to the fully-saturated fixed
                         class-color overrides. 1.0 = no boost, matches
                         the base project's own output; try 1.5-2.5 if the
                         background still looks too flat.
        """
        print("Loading base colorizer (from earlier project)...")
        self.base_net = load_colorizer(base_model_dir)

        print("Loading semantic segmentation model...")
        self.segmenter = Segmenter()

        self.index_to_color = build_index_to_color()
        self.edge_feather = edge_feather
        self.saturation_boost = saturation_boost

    def _base_colorize_ab(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Runs the base pretrained colorizer and returns just the predicted
        a,b channels (full resolution), reusing the same logic as
        colorize_pretrained.colorize() but stopping before the final
        BGR conversion so we can still override specific regions.
        """
        scaled = bgr_image.astype("float32") / 255.0
        lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

        resized = cv.resize(lab, (224, 224))
        L_resized = resized[:, :, 0] - 50

        self.base_net.setInput(cv.dnn.blobFromImage(L_resized))
        ab_pred = self.base_net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_full = cv.resize(ab_pred, (bgr_image.shape[1], bgr_image.shape[0]))
        return ab_full  # (H, W, 2), in OpenCV's Lab a,b scale (roughly [-127,127] after cvtColor's internal handling)

    def process_frame(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        bgr_image: (H, W, 3) uint8, as read by cv.imread / cv.VideoCapture.
        Returns: (H, W, 3) uint8 colorized frame, BGR.
        """
        h, w = bgr_image.shape[:2]

        # 1. Base AI colorization (reused from the earlier project)
        scaled = bgr_image.astype("float32") / 255.0
        lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        ab_base = self._base_colorize_ab(bgr_image)  # OpenCV Lab scale
        ab_base = ab_base * self.saturation_boost
        ab_base = np.clip(ab_base, -127, 127)

        # 2. Semantic segmentation
        rgb = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)
        class_map = self.segmenter.segment(rgb)  # (H, W) int

        # 3. Override with predetermined class colors where applicable
        ab_final = ab_base.copy()
        for class_idx, (a_norm, b_norm) in self.index_to_color.items():
            mask = (class_map == class_idx).astype(np.float32)
            if mask.sum() == 0:
                continue
            if self.edge_feather > 0:
                mask = cv.GaussianBlur(mask, (0, 0), self.edge_feather)

            # Convert normalized [-1,1] class color to OpenCV's Lab a,b scale
            a_val = a_norm * 127.0
            b_val = b_norm * 127.0

            ab_final[:, :, 0] = ab_final[:, :, 0] * (1 - mask) + a_val * mask
            ab_final[:, :, 1] = ab_final[:, :, 1] * (1 - mask) + b_val * mask

        # 4. Recombine and convert back to BGR
        colorized_lab = np.concatenate((L[:, :, np.newaxis], ab_final), axis=2)
        colorized_bgr = cv.cvtColor(colorized_lab.astype(np.float32), cv.COLOR_LAB2BGR)
        colorized_bgr = np.clip(colorized_bgr, 0, 1)
        return (255 * colorized_bgr).astype(np.uint8)

    def process_video_file(self, input_path: str, output_path: str,
                            frame_skip: int = 0, max_frames: int = None):
        """
        Batch-processes a video file frame by frame.
        frame_skip: process every (frame_skip + 1)th frame, repeating the
                    last colorized result on skipped frames (speed/quality tradeoff).
        """
        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")

        fps = cap.get(cv.CAP_PROP_FPS) or 24
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(output_path, fourcc, fps, (w, h))

        frame_idx = 0
        last_result = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_skip > 0 and frame_idx % (frame_skip + 1) != 0 and last_result is not None:
                result = last_result
            else:
                result = self.process_frame(frame)
                last_result = result

            writer.write(result)
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  processed {frame_idx} frames...")

        cap.release()
        writer.release()
        print(f"Done. {frame_idx} frames written to {output_path}")


if __name__ == "__main__":
    print("This module is meant to be imported (see app_realtime.py) or")
    print("used via RealtimeColorizer.process_video_file(...).")
    print(f"Classes with predetermined colors: {list(build_index_to_color().keys())}")
