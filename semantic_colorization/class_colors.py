"""
class_colors.py

Color configuration for ADE20K semantic segmentation.

Supports:
1. Default semantic colors
2. User-defined colors from the GUI

Colors are stored as normalized Lab (a,b) values in [-1,1].
Only the a and b channels are modified.
"""

import cv2
import numpy as np

# ============================================================
# Default colors (ADE20K)
# ============================================================

DEFAULT_CLASS_COLORS = {

    # Nature
    "sky": (-0.05, -0.75),
    "cloud": (0.00, -0.10),
    "tree": (-0.55, 0.35),
    "grass": (-0.60, 0.45),
    "plant": (-0.50, 0.30),
    "flower": (0.45, 0.20),
    "water": (-0.15, -0.60),
    "river": (-0.15, -0.60),
    "sea": (-0.15, -0.65),
    "mountain": (0.08, 0.10),
    "sand": (0.10, 0.45),
    "earth": (0.18, 0.30),

    # Roads
    "road": (0.00, 0.00),
    "sidewalk": (0.05, 0.05),
    "path": (0.10, 0.20),

    # Buildings
    "building": (0.12, 0.15),
    "house": (0.18, 0.22),
    "wall": (0.10, 0.08),
    "bridge": (0.05, 0.05),

    # Vehicles
    "car": (0.45, -0.10),
    "bus": (0.35, 0.15),
    "truck": (0.28, 0.10),
    "bicycle": (-0.10, -0.25),
    "motorcycle": (0.30, -0.10),

    # Humans
    "person": (0.18, 0.22),

    # Animals
    "dog": (0.08, 0.28),
    "cat": (0.10, 0.25),
    "bird": (0.05, 0.18),

    # Furniture
    "chair": (0.10, -0.05),
    "table": (0.12, 0.10),
    "sofa": (0.18, 0.12),
    "bed": (0.15, 0.10),

    # Indoor
    "floor": (0.05, 0.05),
    "ceiling": (0.00, 0.00),
    "door": (0.15, 0.18),
    "window": (-0.05, -0.20),
}

# ============================================================
# User colors
# ============================================================

USER_CLASS_COLORS = {}

# ============================================================
# Get default color
# ============================================================

def get_color(class_name):

    class_name = class_name.lower()

    class_name = class_name.split(",")[0].strip()

    if class_name in USER_CLASS_COLORS:
        return USER_CLASS_COLORS[class_name]

    return DEFAULT_CLASS_COLORS.get(
        class_name,
        (0.0, 0.0)
    )


# ============================================================
# Set user colors
# ============================================================

def set_user_colors(color_dict):

    USER_CLASS_COLORS.clear()

    for cls, colour in color_dict.items():
        USER_CLASS_COLORS[cls.lower()] = tuple(colour)


# ============================================================
# Reset
# ============================================================

def reset_user_colors():

    USER_CLASS_COLORS.clear()


# ============================================================
# Available classes
# ============================================================

def available_classes():

    return sorted(DEFAULT_CLASS_COLORS.keys())


# ============================================================
# RGB -> Lab(a,b)
# ============================================================

def rgb_to_lab_ab(rgb):

    rgb_img = np.uint8([[rgb]])

    lab = cv2.cvtColor(
        rgb_img,
        cv2.COLOR_RGB2LAB
    )[0][0]

    a = (lab[1] - 128) / 127.0
    b = (lab[2] - 128) / 127.0

    return float(a), float(b)


# ============================================================
# HEX -> Lab(a,b)
# ============================================================

def hex_to_lab_ab(hex_color):

    rgb = tuple(
        int(hex_color[i:i+2], 16)
        for i in (1, 3, 5)
    )

    return rgb_to_lab_ab(rgb)


# ============================================================
# Update from GUI
# ============================================================

def update_from_hex_dictionary(hex_dict):

    colours = {}

    for cls, value in hex_dict.items():

        colours[cls] = hex_to_lab_ab(value)

    set_user_colors(colours)


# ============================================================
# Build lookup table
# ============================================================

def build_color_lookup(id2label):

    """
    Converts the Hugging Face ADE20K id2label mapping into:

    {class_id : (a,b)}

    using either the user's selected color
    or the default color.
    """

    lookup = {}

    for idx, name in id2label.items():

        name = name.lower()

        lookup[idx] = get_color(name)

    return lookup


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":

    print("Available Semantic Classes\n")

    for c in available_classes():
        print(c)

    print("\nExample User Override\n")

    update_from_hex_dictionary({

        "sky": "#8000FF",
        "grass": "#00FF00",
        "road": "#444444",
        "car": "#FFFFFF"

    })

    print(get_color("sky"))
    print(get_color("grass"))
    print(get_color("road"))
    print(get_color("car"))