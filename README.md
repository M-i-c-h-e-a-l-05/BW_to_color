# Semantic & Cross-Domain Image Colorization

This project colorizes grayscale and non-photographic images using a
**DDColor** backbone guided by **semantic segmentation**, with additional
support for **historical-era-accurate colorization** and **cross-domain
inputs** (black-and-white photos, line-art sketches, and infrared imagery).

Instead of relying only on grayscale image features, the system identifies
what's in the scene, routes the image to the colorization strategy suited
to its domain, and — for historical photos — grades the result to match
the color palette of the photo's era.

The application provides an interactive **Gradio interface** for
processing images, webcam streams, and videos in real time.

---

# Project Overview

A generic colorization network can produce visually pleasing but
inconsistent results — a sky that's assigned green, or a road that's
tinted brown. This project addresses that in two ways:

1. **Semantic guidance** — a `SegFormer` model trained on ADE20K detects
   objects and scene regions (sky, road, vegetation, buildings, vehicles,
   people, ...) so each region can be steered toward a realistic,
   user-adjustable color prior.
2. **Domain routing** — not every input is a natural grayscale photo. A
   `DomainRouter` sends the image to the pipeline built for its domain:

| Domain       | Strategy                                                            |
| ------------ | -------------------------------------------------------------------|
| `historical` | DDColor + semantic guidance, then era-specific color grading        |
| `bw_photo`   | DDColor directly (no semantic/era stage)                            |
| `sketch`     | Pseudo-shading preprocessing, then DDColor                          |
| `infrared`   | CLAHE contrast enhancement + false-color colormap (no DDColor)      |

---

# Processing Pipeline (historical domain)

```
Input Image / Webcam / Video
            │
            ▼
Convert to Grayscale
            │
            ▼
Semantic Segmentation
(SegFormer + ADE20K)
            │
            ▼
Generate Per-Pixel Class Map
            │
            ▼
DDColor Colorization Network
            │
            ▼
Semantic Color Guidance
(Object-Specific Color Priors)
            │
            ▼
LAB Color Reconstruction
            │
            ▼
Historical Era Classification
(ResNet18 → 1900 / 1920 / 1950 / 1960 / 1970 / WWII / Modern)
            │
            ▼
Era-Specific Color Grading
(tint + contrast + gamma + saturation)
            │
            ▼
Final Colorized Output
```

The luminance (**L**) channel is preserved from the original grayscale
image, while only the chrominance (**a** and **b**) channels are modified
by the colorization network. This preserves detail and sharp edges. The
era grading stage that runs afterward adjusts tone/contrast/saturation on
the finished RGB result — it doesn't touch luminance detail either.

Other domains (`bw_photo`, `sketch`, `infrared`) skip the segmentation and
era stages and use the domain-specific strategy described above.

---

# Features

* Semantic-aware image colorization
* Cross-domain input support: historical photos, plain B&W photos,
  line-art sketches, infrared/thermal imagery
* Historical-era detection and period-accurate color grading (auto or
  manually selected: 1900, 1920, 1950, 1960, 1970, WWII, Modern)
* Real-time webcam colorization
* Video colorization
* Object-specific color guidance
* Adjustable blending strength
* Adjustable color saturation
* CPU and GPU support
* Interactive Gradio interface with domain and era selectors

---

# Semantic Segmentation

The project uses **SegFormer**, a transformer-based semantic segmentation
architecture, trained on the **ADE20K** dataset containing **150 semantic
classes**.

Unlike object detection, semantic segmentation predicts a class label for
every pixel in the image, allowing different regions to receive different
color guidance.

Example detected categories include:

* Buildings
* Sky
* Trees
* Grass
* Water
* Roads
* Cars
* Trucks
* Buses
* Motorcycles
* Bicycles
* People
* Furniture
* Mountains
* Rivers
* Walls
* Sidewalks
* Windows
* Doors
* Vegetation
* Indoor objects

---

# Color Guidance

After segmentation, each semantic class is mapped to a predefined color
prior.

Example mappings include:

| Semantic Class | Example Color Bias     |
| --------------- | ---------------------- |
| Sky             | Blue                    |
| Vegetation      | Green                   |
| Water           | Cyan / Blue             |
| Road            | Gray                    |
| Building        | Beige / Gray            |
| Vehicle         | Red / Blue / Silver     |
| Person          | Natural skin tones      |
| Animals         | Brown / Natural colors  |
| Furniture       | Neutral colors          |

These color priors are softly blended with the base colorization output
to maintain realistic transitions and avoid harsh boundaries.

---

# Historical Era Colorization

For the `historical` domain, a ResNet18 classifier (`historical/era_classifier.py`)
predicts which of seven eras a photo belongs to, each with its own expected
palette and saturation level:

| Era    | Palette      | Saturation |
| ------ | ------------ | ---------- |
| 1900   | Sepia        | 0.55       |
| 1920   | Vintage      | 0.65       |
| 1950   | Kodachrome   | 1.05       |
| 1960   | Vivid        | 1.20       |
| 1970   | Warm         | 1.05       |
| Modern | Natural      | 1.00       |
| WWII   | Muted        | 0.70       |

`historical/era_aesthetic.py` applies the corresponding color grade
(channel tint + contrast/gamma curve + saturation scale) to the DDColor
output. The era can be auto-detected, or forced via
`RealtimeColorizer.set_forced_era(era)` — surfaced in the GUI as an
"Historical Period" dropdown.

The classifier is trained with `historical/train_era_classifier.py`
against a dataset of labeled historical photos; weights live in
`historical/weights/best_model.pth`.

---

# Cross-Domain Colorizers

* **`semantic_colorization/bw_photo_colorizer.py`** — wraps DDColor for
  plain black-and-white photographs, no semantic/era stages.
* **`semantic_colorization/sketch_colorizer.py`** — line art has almost no
  shading, so DDColor alone leaves it flat. This softens the lines into a
  pseudo-shaded grayscale image (blurred, then multiply-blended back with
  the sharp lines) before handing it to DDColor.
* **`semantic_colorization/infrared_colorizer.py`** — infrared intensity
  isn't "natural color" to predict, so this doesn't use DDColor. It
  CLAHE-enhances contrast and applies a false-color colormap
  (inferno/jet/turbo/hot/viridis), the standard way to make IR imagery
  visually interpretable.
* **`semantic_colorization/domain_router.py`** — `DomainRouter` dispatches
  an input image to whichever of the above (or the historical pipeline)
  matches the selected domain.

---

# Project Structure

```
BW_to_col/
│
├── historical/
│   ├── era_classifier.py        # ResNet18 era classifier
│   ├── era_aesthetic.py         # Era-specific color grading
│   ├── train_era_classifier.py
│   └── weights/
│       └── best_model.pth
│
├── semantic_colorization/
│   ├── app_realtime.py          # Gradio GUI (domain + era selectors)
│   ├── realtime_colorizer.py    # Historical pipeline (DDColor + semantic + era)
│   ├── domain_router.py         # Cross-domain dispatch
│   ├── bw_photo_colorizer.py    # Plain B&W photo domain
│   ├── sketch_colorizer.py      # Line-art sketch domain
│   ├── infrared_colorizer.py    # Infrared / thermal domain
│   ├── ddcolor_model.py         # DDColor wrapper
│   ├── segmenter.py             # SegFormer + ADE20K
│   ├── class_colors.py          # Semantic class → color priors
│   ├── context_aware.py
│   └── DDColor/                 # DDColor model + pretrained weights
│
└── model1/
    └── models/
```

---

# Requirements

Install the required packages:

```bash
pip install torch torchvision transformers opencv-python pillow numpy gradio
```

The SegFormer model is automatically downloaded from the Hugging Face Hub
the first time the application is executed. DDColor pretrained weights are
expected at `semantic_colorization/DDColor/pretrain/ddcolor_paper_tiny.pth`.

---

# Running the Application

Run as a module from the project root (required so `historical` and
`semantic_colorization` resolve as packages):

```bash
cd ~/Projects/BW_to_col
python -m semantic_colorization.app_realtime
```

The Gradio interface provides:

* **Input Domain** selector — historical / bw_photo / sketch / infrared
* **Historical Period** selector — Auto or a specific era (historical domain only)
* Image Colorization
* Webcam Colorization
* Video Colorization

---

# Performance

The pipeline performs semantic segmentation, colorization, and (for the
historical domain) era classification and grading for every frame.
Performance depends on hardware.

Approximate CPU performance:

* Image Processing: 1–5 seconds per image
* Webcam: 1–3 FPS
* Video: Depends on frame skipping and resolution

Using a CUDA-enabled GPU significantly improves throughput and allows
near real-time processing.

---

# Core Files

### app_realtime.py
Gradio UI: domain/era selection, image, webcam, and video processing.

### realtime_colorizer.py
The historical pipeline — DDColor + SegFormer semantic guidance + era
classification and grading.

### domain_router.py
Dispatches an input image to the colorizer for its selected domain.

### bw_photo_colorizer.py / sketch_colorizer.py / infrared_colorizer.py
Domain-specific colorization strategies for plain B&W photos, sketches,
and infrared imagery respectively.

### era_classifier.py / era_aesthetic.py
Historical era detection (ResNet18) and the corresponding color grade.

### segmenter.py
Loads the pretrained SegFormer semantic segmentation model and performs inference.

### class_colors.py
Defines semantic class mappings and corresponding color priors used during colorization.

---

# Technologies Used

* Python
* PyTorch
* Transformers (Hugging Face)
* OpenCV
* NumPy
* Pillow
* Gradio

---

# Future Improvements

* Domain selection for webcam/video tabs (currently image-tab only)
* Automatic scene-specific color adaptation
* Temporal consistency for videos
* Higher-resolution semantic segmentation
* GPU optimization
* Batch image processing
* Custom semantic color profiles
* Exportable semantic masks

---

# References

1. Xuan Kou et al., *DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders*, ICCV 2023.
2. Richard Zhang, Phillip Isola, Alexei A. Efros, *Colorful Image Colorization*, ECCV 2016.
3. Enze Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
4. ADE20K Scene Parsing Dataset.
5. PyTorch.
6. Hugging Face Transformers.
