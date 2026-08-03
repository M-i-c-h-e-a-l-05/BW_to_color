"""
Conditional Image Colorization GUI

Uses

• SegFormer ADE20K
• Zhang Colorizer
• Dynamic Semantic Colour Selection

Author:
Realtime Conditional Colorization
"""

import os
import tempfile

import cv2 as cv
import numpy as np

import gradio as gr

from realtime_colorizer import RealtimeColorizer

from class_colors import (

    available_classes,

    update_from_hex_dictionary

)

# -------------------------------------------------------
# Load pipeline once
# -------------------------------------------------------

print()

print("=" * 60)

print("Loading Pipeline")

print("=" * 60)

pipeline = RealtimeColorizer(

    base_model_dir="../model1/models"

)

print()

print("Pipeline Ready")

print("=" * 60)

# -------------------------------------------------------
# Default colours shown in GUI
# -------------------------------------------------------

DEFAULT_GUI_COLOURS = {

    "sky":"#4A90E2",

    "grass":"#4CAF50",

    "tree":"#228B22",

    "road":"#777777",

    "building":"#D2B48C",

    "water":"#2196F3",

    "car":"#FF0000",

    "person":"#FFC0A8"

}

def build_colour_dictionary(

    sky,

    grass,

    tree,

    road,

    building,

    water,

    car,

    person

):

    return {

        "sky":sky,

        "grass":grass,

        "tree":tree,

        "road":road,

        "building":building,

        "water":water,

        "car":car,

        "person":person

    }


def apply_user_colours(colour_dictionary):

    update_from_hex_dictionary(colour_dictionary)


# ==========================================================
# GUI
# ==========================================================

with gr.Blocks(

    title="Conditional Image Colorization"

) as demo:

    gr.Markdown("""

# 🎨 Conditional Image Colorization using ADE20K

Upload an image, use your webcam, or upload a video.

Choose custom colours for:

- Sky
- Grass
- Trees
- Roads
- Buildings
- Water
- Cars
- People

The AI first predicts realistic colours using the Zhang model,
then semantic segmentation applies your selected colours only to
the detected regions.

""")

    # -----------------------------------------------------

    with gr.Row():

        blend_slider = gr.Slider(

            minimum=0,

            maximum=1,

            value=0.75,

            step=0.05,

            label="Blend Strength"

        )

        saturation_slider = gr.Slider(

            minimum=1,

            maximum=3,

            value=1.8,

            step=0.1,

            label="Saturation Boost"

        )

    # -----------------------------------------------------

    gr.Markdown("## Semantic Colour Selection")

    with gr.Row():

        sky_picker = gr.ColorPicker(

            label="Sky",

            value=DEFAULT_GUI_COLOURS["sky"]

        )

        grass_picker = gr.ColorPicker(

            label="Grass",

            value=DEFAULT_GUI_COLOURS["grass"]

        )

        tree_picker = gr.ColorPicker(

            label="Tree",

            value=DEFAULT_GUI_COLOURS["tree"]

        )

        road_picker = gr.ColorPicker(

            label="Road",

            value=DEFAULT_GUI_COLOURS["road"]

        )

    with gr.Row():

        building_picker = gr.ColorPicker(

            label="Building",

            value=DEFAULT_GUI_COLOURS["building"]

        )

        water_picker = gr.ColorPicker(

            label="Water",

            value=DEFAULT_GUI_COLOURS["water"]

        )

        car_picker = gr.ColorPicker(

            label="Car",

            value=DEFAULT_GUI_COLOURS["car"]

        )

        person_picker = gr.ColorPicker(

            label="Person",

            value=DEFAULT_GUI_COLOURS["person"]

        )

    # -----------------------------------------------------

    detected_box = gr.Textbox(

        label="Detected Objects",

        interactive=False,

        lines=3

    )

    # =====================================================
    # IMAGE TAB
    # =====================================================

    with gr.Tab("Image"):

        with gr.Row():

            image_input = gr.Image(

                type="numpy",

                label="Input Image"

            )

            image_output = gr.Image(

                label="Colorized Image"

            )

        image_button = gr.Button(

            "Colorize Image",

            variant="primary"

        )

    # =====================================================
    # WEBCAM TAB
    # =====================================================

    with gr.Tab("Webcam"):

        webcam_input = gr.Image(

            sources=["webcam"],

            streaming=True,

            type="numpy",

            label="Camera"

        )

        webcam_output = gr.Image(

            label="Output"

        )

    # =====================================================
    # VIDEO TAB
    # =====================================================

    with gr.Tab("Video"):

        video_input = gr.Video(

            label="Input Video"

        )

        frame_skip = gr.Slider(

            0,

            5,

            value=0,

            step=1,

            label="Frame Skip"

        )

        process_video = gr.Button(

            "Colorize Video",

            variant="primary"

        )

        video_output = gr.Video(

            label="Output Video"

        )

    # =====================================================
    # RESET BUTTON (must be created inside the Blocks
    # context, otherwise it never gets attached to the app
    # and won't render)
    # =====================================================

    reset_button = gr.Button(

        "Reset Colours"

    )

    # =====================================================
    # FOOTER (same reason -- must stay inside the Blocks
    # context to actually render)
    # =====================================================

    gr.Markdown("""

---

## Conditional Image Colorization

✔ Zhang et al. Colorization Network

✔ SegFormer ADE20K Semantic Segmentation

✔ User Controlled Object Colouring

✔ Texture Preserving LAB Blending

""")

    # ==========================================================
    # IMAGE CALLBACK
    # ==========================================================

    def colourize_image(

        image,

        blend,

        saturation,

        sky,

        grass,

        tree,

        road,

        building,

        water,

        car,

        person

    ):

        if image is None:

            return None, "No image uploaded."

        # ---------------------------------------------
        # Update pipeline parameters
        # ---------------------------------------------

        pipeline.set_blend_strength(

            blend

        )

        pipeline.set_saturation(

            saturation

        )

        # ---------------------------------------------
        # Update semantic colours
        # ---------------------------------------------

        colours = build_colour_dictionary(

            sky,

            grass,

            tree,

            road,

            building,

            water,

            car,

            person

        )

        update_from_hex_dictionary(

            colours

        )

        # ---------------------------------------------
        # Convert image
        # ---------------------------------------------

        bgr = cv.cvtColor(

            image,

            cv.COLOR_RGB2BGR

        )

        # ---------------------------------------------
        # Run AI
        # ---------------------------------------------

        result = pipeline.process_frame(

            bgr

        )

        # ---------------------------------------------
        # Detected objects
        # ---------------------------------------------

        detected = pipeline.get_detected_objects()

        if len(detected) == 0:

            text = "No ADE20K objects detected."

        else:

            text = ", ".join(

                detected

            )

        # ---------------------------------------------
        # Convert back
        # ---------------------------------------------

        result = cv.cvtColor(

            result,

            cv.COLOR_BGR2RGB

        )

        return result, text


    # ==========================================================
    # RESET BUTTON
    # ==========================================================

    def reset_everything():

        pipeline.reset()

        return (

            DEFAULT_GUI_COLOURS["sky"],

            DEFAULT_GUI_COLOURS["grass"],

            DEFAULT_GUI_COLOURS["tree"],

            DEFAULT_GUI_COLOURS["road"],

            DEFAULT_GUI_COLOURS["building"],

            DEFAULT_GUI_COLOURS["water"],

            DEFAULT_GUI_COLOURS["car"],

            DEFAULT_GUI_COLOURS["person"],

            "",

            None

        )


    # ==========================================================
    # CLEAR OUTPUT
    # ==========================================================

    def clear_output():

        return None, ""

    # ==========================================================
    # WEBCAM CALLBACK
    # ==========================================================

    def colourize_webcam_live(

        frame,

        blend,

        saturation,

        sky,

        grass,

        tree,

        road,

        building,

        water,

        car,

        person

    ):

        if frame is None:

            return None, ""

        # ---------------------------------------------
        # Update pipeline settings
        # ---------------------------------------------

        pipeline.set_blend_strength(

            blend

        )

        pipeline.set_saturation(

            saturation

        )

        # ---------------------------------------------
        # Update semantic colours
        # ---------------------------------------------

        colours = build_colour_dictionary(

            sky,

            grass,

            tree,

            road,

            building,

            water,

            car,

            person

        )

        update_from_hex_dictionary(

            colours

        )

        # ---------------------------------------------
        # Convert RGB -> BGR
        # ---------------------------------------------

        bgr = cv.cvtColor(

            frame,

            cv.COLOR_RGB2BGR

        )

        # ---------------------------------------------
        # AI Processing
        # ---------------------------------------------

        output = pipeline.process_frame(

            bgr

        )

        # ---------------------------------------------
        # Get detected ADE20K classes
        # ---------------------------------------------

        detected = pipeline.get_detected_objects()

        if len(detected) == 0:

            detected_text = "No objects detected"

        else:

            detected_text = ", ".join(

                detected

            )

        # ---------------------------------------------
        # Convert back
        # ---------------------------------------------

        output = cv.cvtColor(

            output,

            cv.COLOR_BGR2RGB

        )

        return output, detected_text


    # ==========================================================
    # SLIDER CALLBACKS
    # ==========================================================

    def update_blend_strength(

        value

    ):

        pipeline.set_blend_strength(

            value

        )


    def update_saturation(

        value

    ):

        pipeline.set_saturation(

            value

        )


    # ==========================================================
    # DETECT OBJECTS ONLY
    # ==========================================================

    def detect_objects(

        image

    ):

        if image is None:

            return ""

        bgr = cv.cvtColor(

            image,

            cv.COLOR_RGB2BGR

        )

        rgb = cv.cvtColor(

            bgr,

            cv.COLOR_BGR2RGB

        )

        class_map = pipeline.segmenter.segment(

            rgb

        )

        detected = pipeline.segmenter.detected_classes(

            class_map

        )

        if len(detected) == 0:

            return "No objects detected"

        return ", ".join(

            detected

        )

    # ==========================================================
    # IMAGE EVENTS
    # ==========================================================

    image_button.click(

        fn=colourize_image,

        inputs=[

            image_input,

            blend_slider,

            saturation_slider,

            sky_picker,

            grass_picker,

            tree_picker,

            road_picker,

            building_picker,

            water_picker,

            car_picker,

            person_picker

        ],

        outputs=[

            image_output,

            detected_box

        ]

    )

    image_input.change(

        fn=clear_output,

        inputs=[],

        outputs=[

            image_output,

            detected_box

        ]

    )

    # ==========================================================
    # WEBCAM EVENTS
    # ==========================================================

    webcam_input.stream(

        fn=colourize_webcam_live,

        inputs=[

            webcam_input,

            blend_slider,

            saturation_slider,

            sky_picker,

            grass_picker,

            tree_picker,

            road_picker,

            building_picker,

            water_picker,

            car_picker,

            person_picker

        ],

        outputs=[

            webcam_output,

            detected_box

        ]

    )

    # ==========================================================
    # SLIDER EVENTS
    # ==========================================================

    blend_slider.change(

        fn=update_blend_strength,

        inputs=blend_slider,

        outputs=[]

    )

    saturation_slider.change(

        fn=update_saturation,

        inputs=saturation_slider,

        outputs=[]

    )

    # ==========================================================
    # LIVE OBJECT DETECTION
    # ==========================================================

    image_input.change(

        fn=detect_objects,

        inputs=image_input,

        outputs=detected_box

    )

    # ==========================================================
    # RESET BUTTON EVENT
    # ==========================================================

    reset_button.click(

        fn=reset_everything,

        inputs=[],

        outputs=[

            sky_picker,

            grass_picker,

            tree_picker,

            road_picker,

            building_picker,

            water_picker,

            car_picker,

            person_picker,

            detected_box,

            image_output

        ]

    )

    # ==========================================================
    # VIDEO CALLBACK
    # ==========================================================

    def colourize_video(

        video_path,

        frame_skip,

        blend,

        saturation,

        sky,

        grass,

        tree,

        road,

        building,

        water,

        car,

        person,

        progress=gr.Progress()

    ):

        if video_path is None:

            return None

        # ---------------------------------------------
        # Update pipeline
        # ---------------------------------------------

        pipeline.set_blend_strength(

            blend

        )

        pipeline.set_saturation(

            saturation

        )

        colours = build_colour_dictionary(

            sky,

            grass,

            tree,

            road,

            building,

            water,

            car,

            person

        )

        update_from_hex_dictionary(

            colours

        )

        # ---------------------------------------------
        # Output path
        # ---------------------------------------------

        output_video = tempfile.NamedTemporaryFile(

            suffix=".mp4",

            delete=False

        ).name

        progress(

            0,

            desc="Processing Video..."

        )

        pipeline.process_video_file(

            video_path,

            output_video,

            frame_skip=int(frame_skip)

        )

        return output_video


    # ==========================================================
    # VIDEO EVENTS
    # ==========================================================

    process_video.click(

        fn=colourize_video,

        inputs=[

            video_input,

            frame_skip,

            blend_slider,

            saturation_slider,

            sky_picker,

            grass_picker,

            tree_picker,

            road_picker,

            building_picker,

            water_picker,

            car_picker,

            person_picker

        ],

        outputs=video_output

    )

    # ==========================================================
    # START APP
    # ==========================================================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=7861,

        share=False

    )