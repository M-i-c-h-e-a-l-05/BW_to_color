# Real-time Multi-Object Colorization with Semantic Segmentation

**Internship task, built on top of the base colorization project** (see
the top-level `README.md` for that project). This module adds semantic
segmentation so different object types in a frame can be colorized with
their own predetermined color scheme, and wraps the whole thing in a
real-time GUI for webcam or uploaded video.

## How it builds on the base project

This isn't a separate project -- `realtime_colorizer.py` directly imports
and reuses `load_colorizer()` from `../colorize_pretrained.py`. The
general "what color should this pixel be" prediction still comes from
the same pretrained Zhang et al. colorizer used in the base project.
What's new here is a semantic segmentation pass that identifies specific
object regions and *overrides* the AI's color guess in those regions
with a fixed, predetermined color for that class -- e.g. always tinting
detected cars/buses/bikes toward a "vehicle" color, regardless of what
the general colorizer alone would have guessed.

## Pipeline

```
frame -> [base AI colorizer]  -> generic a,b color prediction (whole frame)
      -> [segmentation model] -> per-pixel class map
      -> for each pixel: if its class has a predetermined color,
         override the base a,b prediction with that color (soft-edged,
         so it blends rather than looking like a cutout)
      -> recombine with original L channel -> final colorized frame
```

The L (lightness/detail) channel is never touched, so edges and texture
stay sharp even in overridden regions.

## Object classes and color schemes

The segmentation model (torchvision's DeepLabV3, MobileNetV3 backbone,
trained on COCO with Pascal VOC's 21-class label set) recognizes:

| Group | VOC classes included | Color |
|---|---|---|
| Vehicle | car, bus, motorbike, bicycle, train, aeroplane, boat | muted red/orange |
| Vegetation | pottedplant | green |
| Person | person | warm skin tone |
| Animal | bird, cat, cow, dog, horse, sheep | warm brown/tan |
| Furniture | chair, diningtable, sofa, tvmonitor, bottle | neutral warm gray |

See `class_colors.py` for the exact Lab (a,b) values and how to change them.

### Known limitation: no dedicated "building" class

The task description names vehicles, trees, and buildings as example
categories. Readily available, easily-verified pretrained segmentation
models (COCO/VOC-trained, as used here) don't include a "building"
class -- that requires a Cityscapes-trained model (which does have
building/road/sky/etc.), and those pretrained checkpoints are larger and
come from less standardized sources.

**Trees** are approximated using VOC's `pottedplant` class, the closest
available proxy -- it won't catch large background trees/forests well,
only plant-like foreground objects.

**To add real building detection:** swap `segmenter.py`'s model for a
Cityscapes-pretrained one (e.g. via `segmentation-models-pytorch` with a
Cityscapes checkpoint, or a repo like `VainF/DeepLabV3Plus-Pytorch`), and
update `class_colors.py`'s `VOC_CLASSES` list and `CLASS_GROUPS` mapping
to match that model's label set (which includes `building`, `road`,
`sky`, `vegetation`, `sidewalk`, etc. directly). The rest of the pipeline
(`realtime_colorizer.py`) doesn't need to change -- it just consumes
whatever class list `segmenter.py` provides.

## Setup

```bash
cd semantic_colorization
pip install torch torchvision opencv-python numpy pillow gradio
```

Requires the base project's pretrained colorizer model files already
downloaded (see `../download_pretrained_model.py` in the parent folder).

The segmentation model's weights download automatically from
`download.pytorch.org` the first time `Segmenter()` is created.

## Running it

```bash
python app_realtime.py
```

Opens a Gradio interface with two tabs:
- **Webcam (live):** streams your camera, colorizing each frame in real time.
- **Upload video:** upload a video file, get back a fully colorized version.
  A frame-skip slider trades quality for speed (repeats the last
  colorized frame between processed ones) -- useful on CPU-only machines.

## Performance notes

Running both a colorization network and a segmentation network per frame
is heavier than either alone. On CPU, expect roughly 1-3 frames/sec
depending on hardware -- real-time in the sense of "continuously
updating," but not high frame rate. A CUDA GPU will be substantially
faster. The MobileNetV3 segmentation backbone was specifically chosen
over the more accurate ResNet backbones available in torchvision for
this reason.

## Files

- `class_colors.py` -- VOC class list, class groupings, and predetermined colors
- `segmenter.py` -- loads the pretrained segmentation model, runs inference
- `realtime_colorizer.py` -- combines segmentation + base colorizer into one pipeline
- `app_realtime.py` -- Gradio GUI (webcam + video upload)
