# Photo Colorization Projects

## Structure

- **`model1/`** -- base colorization project: a pretrained inference
  pipeline (Zhang et al., ECCV 2016, via OpenCV DNN) plus a custom
  trainable U-Net in Lab color space. See `model1/README.md`.
- **`semantic_colorization/`** -- internship task, built on top of
  `model1`: real-time multi-object colorization using semantic
  segmentation, with a Gradio GUI for webcam/video input. See
  `semantic_colorization/README.md`.

## Setup

Each folder has its own `requirements.txt`. The pretrained model weights
for `model1` (~130MB) are not committed to this repo -- download them
with:

```bash
cd model1
python download_pretrained_model.py
```

`semantic_colorization` reuses `model1`'s pretrained colorizer directly
(see its README for how the two connect), so `model1`'s model files must
be downloaded before running anything in `semantic_colorization`.
