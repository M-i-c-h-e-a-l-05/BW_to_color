"""
GUI for real-time multi-object colorization.

Two modes, per the task requirements:
    1. Webcam tab -- live streaming, frame-by-frame colorization.
    2. Upload tab -- upload a video file, get back a fully colorized video.

Usage:
    python app_realtime.py
(loads the base colorizer + segmentation model once at startup, then
serves both tabs from the same pipeline instance)
"""
import os
import tempfile

import cv2 as cv
import numpy as np
from PIL import Image
import gradio as gr

from realtime_colorizer import RealtimeColorizer
from class_colors import VOC_CLASSES, build_index_to_color


print("Starting up -- loading models (this happens once)...")
pipeline = RealtimeColorizer(base_model_dir="../model1/models")
print("Ready.")


def colorize_webcam_frame(frame: np.ndarray):
    """Gradio webcam streaming callback -- called repeatedly with each new frame."""
    if frame is None:
        return None
    bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    result_bgr = pipeline.process_frame(bgr)
    return cv.cvtColor(result_bgr, cv.COLOR_BGR2RGB)


def colorize_uploaded_video(video_path: str, frame_skip: int, progress=gr.Progress()):
    if video_path is None:
        return None

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    progress(0, desc="Processing video...")
    pipeline.process_video_file(video_path, out_path, frame_skip=int(frame_skip))
    return out_path


detected_classes_note = ", ".join(
    f"{VOC_CLASSES[i]}" for i in build_index_to_color().keys()
)

with gr.Blocks(title="Real-time Multi-Object Colorization") as demo:
    gr.Markdown(f"""
    # Real-time Multi-Object Colorization
    Semantic segmentation identifies objects in the frame; each recognized
    class is colorized using a predetermined color scheme, while
    everything else falls back to the general-purpose AI colorizer from
    the base project.

    **Classes with a fixed color scheme:** {detected_classes_note}
    """)

    with gr.Tab("Webcam (live)"):
        gr.Markdown(
            "Grant camera access, then colorized frames update in real time. "
            "Speed depends on your hardware -- CPU-only machines will see "
            "lower frame rates than GPU machines."
        )
        webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
        webcam_output = gr.Image(label="Colorized (live)")
        webcam_input.stream(
            fn=colorize_webcam_frame,
            inputs=webcam_input,
            outputs=webcam_output,
        )

    with gr.Tab("Upload video"):
        gr.Markdown("Upload a video file to colorize it end-to-end.")
        video_input = gr.Video(label="Input video")
        frame_skip_slider = gr.Slider(
            0, 5, value=0, step=1,
            label="Frame skip (higher = faster but choppier; repeats last colorized frame)"
        )
        process_btn = gr.Button("Colorize video", variant="primary")
        video_output = gr.Video(label="Colorized result")

        process_btn.click(
            fn=colorize_uploaded_video,
            inputs=[video_input, frame_skip_slider],
            outputs=video_output,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
