# Real-time Semantic-Aware Image & Video Colorization

This project extends the original **Zhang et al. image colorization model** by combining it with **semantic segmentation** to produce more context-aware colorization. Instead of relying only on grayscale image features, the system first identifies the objects present in the scene and then applies object-specific color guidance before generating the final colorized output.

The application provides an interactive **Gradio interface** for processing images, webcam streams, and videos in real time.

---

# Project Overview

The original colorization model predicts realistic colors for grayscale images using deep learning. While the results are often visually pleasing, the model may assign inconsistent colors to important objects.

This project improves the output by introducing a semantic understanding stage.

A **SegFormer semantic segmentation model trained on the ADE20K dataset** detects objects and scene elements such as:

* Sky
* Buildings
* Roads
* Trees
* Grass
* Water
* Mountains
* Cars
* Buses
* Motorcycles
* Bicycles
* People
* Animals
* Furniture
* Walls
* Sidewalks
* Many other urban and natural scene classes

Each detected semantic class is assigned a predefined color palette that guides the colorization network toward more consistent and realistic outputs.

---

# Processing Pipeline

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
Original Zhang Colorization Network
            │
            ▼
Semantic Color Guidance
(Object-Specific Color Priors)
            │
            ▼
LAB Color Reconstruction
            │
            ▼
Final Colorized Output
```

The luminance (**L**) channel is preserved from the original grayscale image, while only the chrominance (**a** and **b**) channels are modified. This preserves image detail and sharp edges.

---

# Features

* Semantic-aware image colorization
* Real-time webcam colorization
* Video colorization
* Object-specific color guidance
* Adjustable blending strength
* Adjustable color saturation
* CPU and GPU support
* Interactive Gradio interface

---

# Semantic Segmentation

The project uses **SegFormer**, a transformer-based semantic segmentation architecture, trained on the **ADE20K** dataset containing **150 semantic classes**.

Unlike object detection, semantic segmentation predicts a class label for every pixel in the image, allowing different regions to receive different color guidance.

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

This enables the model to produce more contextually appropriate colorization than a generic colorization model alone.

---

# Color Guidance

After segmentation, each semantic class is mapped to a predefined color prior.

Example mappings include:

| Semantic Class | Example Color Bias     |
| -------------- | ---------------------- |
| Sky            | Blue                   |
| Vegetation     | Green                  |
| Water          | Cyan / Blue            |
| Road           | Gray                   |
| Building       | Beige / Gray           |
| Vehicle        | Red / Blue / Silver    |
| Person         | Natural skin tones     |
| Animals        | Brown / Natural colors |
| Furniture      | Neutral colors         |

These color priors are softly blended with the base colorization output to maintain realistic transitions and avoid harsh boundaries.

---

# Project Structure

```
semantic_colorization/
│
├── app_realtime.py
├── realtime_colorizer.py
├── segmenter.py
├── class_colors.py
├── README.md
│
└── ../model1/
    ├── colorize_pretrained.py
    └── models/
```

---

# Requirements

Install the required packages:

```bash
pip install torch torchvision transformers opencv-python pillow numpy gradio
```

The project also requires the pretrained Zhang colorization model files from the parent project.

The SegFormer model is automatically downloaded from the Hugging Face Hub the first time the application is executed.

---

# Running the Application

```bash
cd semantic_colorization
python app_realtime.py
```

The Gradio interface provides multiple processing modes:

* Image Colorization
* Webcam Colorization
* Video Colorization

---

# Performance

The pipeline performs both semantic segmentation and image colorization for every frame.

Performance depends on hardware.

Approximate CPU performance:

* Image Processing: 1–5 seconds per image
* Webcam: 1–3 FPS
* Video: Depends on frame skipping and resolution

Using a CUDA-enabled GPU significantly improves throughput and allows near real-time processing.

---

# Core Files

### app_realtime.py

Provides the Gradio user interface and handles image, webcam, and video processing.

### realtime_colorizer.py

Implements the complete semantic-aware colorization pipeline by combining semantic segmentation with the Zhang colorization network.

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

* User-selectable color palettes
* Automatic scene-specific color adaptation
* Temporal consistency for videos
* Higher-resolution semantic segmentation
* GPU optimization
* Batch image processing
* Custom semantic color profiles
* Exportable semantic masks

---

# References

1. Richard Zhang, Phillip Isola, Alexei A. Efros, *Colorful Image Colorization*, ECCV 2016.

2. Enze Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.

3. ADE20K Scene Parsing Dataset.

4. PyTorch.

5. Hugging Face Transformers.
