#!/usr/bin/env python
"""
Colorize a black-and-white image using the pretrained Zhang et al. (2016)
"Colorful Image Colorization" model via OpenCV's DNN module.

Run download_pretrained_model.py first to get the required model files.

Usage:
    python colorize_pretrained.py --image path/to/bw_photo.jpg --output result.png
"""
import argparse
import os

import cv2 as cv
import numpy as np


def load_colorizer(model_dir: str = "models"):
    proto = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    weights = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    pts_path = os.path.join(model_dir, "pts_in_hull.npy")

    for p in (proto, weights, pts_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing model file: {p}\n"
                "Run `python download_pretrained_model.py` first."
            )

    net = cv.dnn.readNetFromCaffe(proto, weights)
    pts = np.load(pts_path)

    # The network has two special layers that need the cluster-center
    # points injected as 1x1 conv weights -- this is how the pretrained
    # model maps its 313 color-class predictions back to actual a,b values.
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

    return net


def colorize(net, bgr_image: np.ndarray) -> np.ndarray:
    """
    bgr_image: input image as read by cv.imread (BGR, uint8), any size.
    Returns: colorized image, BGR uint8, same size as input.
    """
    scaled = bgr_image.astype("float32") / 255.0
    lab = cv.cvtColor(scaled, cv.COLOR_BGR2LAB)

    # The network was trained on 224x224 inputs -- resize just for the
    # forward pass, then upscale the predicted a,b back to the original size.
    L_orig = lab[:, :, 0]
    resized = cv.resize(lab, (224, 224))
    L_resized = resized[:, :, 0]
    L_resized -= 50  # mean-centering, as the network expects

    net.setInput(cv.dnn.blobFromImage(L_resized))
    ab_pred = net.forward()[0, :, :, :].transpose((1, 2, 0))  # (H, W, 2)

    ab_pred_full = cv.resize(ab_pred, (bgr_image.shape[1], bgr_image.shape[0]))

    colorized_lab = np.concatenate(
        (L_orig[:, :, np.newaxis], ab_pred_full), axis=2
    )
    colorized_bgr = cv.cvtColor(colorized_lab, cv.COLOR_LAB2BGR)
    colorized_bgr = np.clip(colorized_bgr, 0, 1)
    colorized_bgr = (255 * colorized_bgr).astype("uint8")
    return colorized_bgr


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input B&W image")
    p.add_argument("--output", default="colorized_output.png")
    p.add_argument("--model_dir", default="models")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    img = cv.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    print("Loading pretrained model...")
    net = load_colorizer(args.model_dir)

    print(f"Colorizing {args.image} ...")
    result = colorize(net, img)

    cv.imwrite(args.output, result)
    print(f"Saved: {args.output}")
