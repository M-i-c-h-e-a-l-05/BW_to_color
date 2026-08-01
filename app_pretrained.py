"""
Gradio demo using the pretrained Zhang et al. colorization model.
No training required -- just download the model files first:

    python download_pretrained_model.py

Then run:
    python app_pretrained.py
"""
import cv2 as cv
import numpy as np
from PIL import Image
import gradio as gr

from colorize_pretrained import load_colorizer, colorize


print("Loading pretrained model (this happens once at startup)...")
net = load_colorizer(model_dir="models")
print("Model loaded.")


def colorize_pil(image: Image.Image):
    if image is None:
        return None
    rgb = np.array(image.convert("RGB"))
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    result_bgr = colorize(net, bgr)
    result_rgb = cv.cvtColor(result_bgr, cv.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


with gr.Blocks(title="Photo Colorizer (Pretrained)") as demo:
    gr.Markdown("""
    # Black & White Photo Colorizer
    Uses the pretrained "Colorful Image Colorization" model
    (Zhang, Isola, Efros -- ECCV 2016). No training required.
    """)
    with gr.Row():
        input_image = gr.Image(label="Input Photo", type="pil")
        output_image = gr.Image(label="Colorized Result", type="pil")

    colorize_btn = gr.Button("Colorize", variant="primary")
    colorize_btn.click(fn=colorize_pil, inputs=input_image, outputs=output_image)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
