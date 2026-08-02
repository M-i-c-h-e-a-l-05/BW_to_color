#!/usr/bin/env python
"""
Downloads the pretrained Zhang et al. (2016) "Colorful Image Colorization"
Caffe model files needed for inference. Run this once before using
colorize_pretrained.py or app_pretrained.py.

Files downloaded into ./models/:
    colorization_deploy_v2.prototxt   (network architecture definition)
    pts_in_hull.npy                    (313 quantized color cluster centers)
    colorization_release_v2.caffemodel (trained weights, ~130MB)
"""
import os
import urllib.request

MODEL_DIR = "models"

FILES = {
    "colorization_deploy_v2.prototxt": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    ],
    "pts_in_hull.npy": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    ],
    "colorization_release_v2.caffemodel": [
        "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
        "https://www.dropbox.com/scl/fi/d8zffur3wmd4wet58dp9x/colorization_release_v2.caffemodel?rlkey=iippu6vtsrox3pxkeohcuh4oy&dl=1",
    ],
}


def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"Already exists, skipping: {dest}")
        return
    print(f"Downloading {dest} ...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req) as response, open(dest, "wb") as out_file:
        out_file.write(response.read())
    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"  Done ({size_mb:.1f} MB)")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    for filename, urls in FILES.items():
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"Already exists, skipping: {dest}")
            continue
        success = False
        for url in urls:
            try:
                download(url, dest)
                success = True
                break
            except Exception as e:
                print(f"  FAILED from {url}: {e}")
        if not success:
            print(f"  Could not download {filename} from any source.")
            print(f"  Try downloading manually from: {urls[-1]}")
    print("\nDone. Model files should now be in ./models/")