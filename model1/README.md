# B&W Photo Colorization

Two approaches are included:
1. **Pretrained** (`download_pretrained_model.py`, `colorize_pretrained.py`,
   `app_pretrained.py`) -- no training needed, works immediately.
2. **Train-your-own** (`model.py`, `dataset.py`, `train.py`, `app.py`) --
   a U-Net you train yourself on your own dataset.

---

## Option 1: Pretrained model (fastest path to working results)

Uses Zhang, Isola, and Efros' "Colorful Image Colorization" (ECCV 2016)
model, distributed as pretrained Caffe weights, run through OpenCV's DNN
module. This is the model most online colorization tutorials use.

```bash
pip install opencv-python numpy pillow gradio

# Downloads model files into ./models/ (~130MB total)
python download_pretrained_model.py

# Colorize a single image from the command line
python colorize_pretrained.py --image path/to/bw_photo.jpg --output result.png

# Or launch the interactive web demo
python app_pretrained.py
```

**Verification note:** I confirmed the `prototxt` and `pts_in_hull.npy`
files download correctly and have the expected shapes/layer names that
the inference code expects. The `.caffemodel` weight file (hosted on
`eecs.berkeley.edu`) I could not fetch from my sandboxed environment
(outside its allowed domains), so I wasn't able to run a full forward
pass end-to-end myself -- if the download fails on your machine, use the
Dropbox mirror linked as a comment in `download_pretrained_model.py`.

This model is fixed -- you can't fine-tune it further without the
original Caffe training setup, which is largely obsolete today. If your
task requires improving on it, that's what Option 2 (below) is for.

---

## Option 2: Train your own model

A trainable image colorization pipeline: given a grayscale photo, predicts
plausible colors for it.

## How it works

Colorization is framed as a **regression problem in Lab color space**,
not RGB:

- **L** (lightness) = your grayscale image. This is the model's *input*.
- **a, b** (color channels) = what's "missing" and what the model predicts.

A **U-Net** (encoder-decoder with skip connections) takes the L channel
and predicts the a,b channels. Recombining predicted `a,b` with the
original `L` and converting back to RGB gives the colorized image.

This is trained **self-supervised**: any ordinary color photo can be
turned into a training pair automatically (convert to grayscale for the
input, use the original as the color target). No manual labeling needed.

## Files

- `model.py` — the U-Net architecture
- `dataset.py` — loads a folder of color images, converts to L/ab pairs;
  also has `lab_to_rgb()` to convert model output back into a viewable image
- `train.py` — training loop (checkpointing, validation split, LR decay)
- `app.py` — Gradio web UI for uploading a photo and viewing the colorized result

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get training data

Any folder of ordinary color photos works. Options:
- **COCO** (`http://images.cocodataset.org/zips/train2017.zip`) — general scenes, ~118k images
- **Places365** — scenery/interiors, good variety
- Your own photo collection, if it's large and varied enough (a few thousand+ images minimum to see real learning; tens of thousands for good generalization)

Put them all in a single folder, e.g. `data/train_images/`.

## 2. Train

```bash
python train.py --data_dir data/train_images --epochs 30 --batch_size 16
```

Useful flags:
- `--base_channels 32` — smaller/faster model if you're short on GPU memory
- `--image_size 128` — smaller images train faster (good for a first test run)
- `--resume checkpoints/latest.pt` — continue training from a saved checkpoint

Checkpoints save to `checkpoints/latest.pt` (every epoch) and
`checkpoints/best.pt` (best validation loss so far).

**Sanity-check tip before a long training run:** train for 1-2 epochs on a
small subset (e.g. 200 images) first, just to confirm the loss actually
goes down and nothing crashes, before committing to a big dataset/many epochs.

## 3. Run the demo

```bash
python app.py --checkpoint checkpoints/best.pt
```

Opens a Gradio interface where you can upload a photo and see the colorized output.

## Where to go from here (once the basic version works)

- **Better loss**: the current L1 loss tends to produce somewhat desaturated
  ("safe") colors, since averaging plausible colors pulls toward gray/brown.
  A classification-based loss over quantized color bins (Zhang et al., 2016,
  "Colorful Image Colorization") gives more vivid, confident colors.
  Alternatively, add a GAN discriminator (pix2pix-style) for more realistic outputs.
- **Perceptual loss**: adding a VGG-feature-based loss alongside L1 often
  improves perceived realism.
- **Pretrained backbone**: swapping the encoder for a pretrained ResNet
  (fine-tuning instead of training from scratch) usually converges faster
  and generalizes better with less data — this is roughly the approach
  DeOldify uses.
- **Evaluation**: track PSNR/SSIM on a held-out validation set in addition
  to eyeballing outputs, so you have a number to report on progress.
