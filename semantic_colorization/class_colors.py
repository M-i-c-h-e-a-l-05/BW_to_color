"""
Predetermined color schemes for each semantic class the segmentation model
can detect.

Colors are defined directly as Lab (a, b) offsets -- consistent with the
Lab-space approach used throughout the rest of this project (see
../dataset.py and ../colorize_pretrained.py). Reusing Lab here means the
class colors blend naturally with the base colorization instead of looking
like a flat paint-bucket fill.

The underlying model (torchvision DeepLabV3, trained on COCO with the
Pascal VOC 21-class label set) does not include a dedicated "building"
class -- see the project README for why, and how to extend this mapping
if a Cityscapes-style model is swapped in later.
"""

# VOC class list, in the exact index order the model outputs.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

# Group raw VOC classes into the broader semantic buckets requested by the
# task ("vehicles", "trees", ...). Classes not listed here fall back to
# the AI-predicted base color (see realtime_colorizer.py) rather than a
# fixed tint, since forcing everything into 3 buckets would misrepresent
# categories the model isn't actually able to distinguish.
CLASS_GROUPS = {
    "vehicle": ["car", "bus", "motorbike", "bicycle", "train", "aeroplane", "boat"],
    "vegetation": ["pottedplant"],   # closest available proxy for "trees"
    "person": ["person"],
    "animal": ["bird", "cat", "cow", "dog", "horse", "sheep"],
    "furniture": ["chair", "diningtable", "sofa", "tvmonitor", "bottle"],
}

# Predetermined (a, b) tint per group, in the same normalized [-1, 1]
# Lab space used elsewhere in the project (see dataset.py's normalization).
# Positive a = red/magenta, negative a = green. Positive b = yellow,
# negative b = blue.
GROUP_COLORS = {
    "vehicle":    (0.35, -0.10),   # muted red/orange
    "vegetation": (-0.45, 0.35),   # green
    "person":     (0.15, 0.20),    # warm skin-toned tint
    "animal":     (0.05, 0.30),    # warm brown/tan
    "furniture":  (0.10, -0.05),   # neutral warm gray
}


def build_class_to_group():
    """Returns {voc_class_name: group_name} for every class with a defined group."""
    mapping = {}
    for group, classes in CLASS_GROUPS.items():
        for c in classes:
            mapping[c] = group
    return mapping


def build_index_to_color():
    """
    Returns {voc_class_index: (a, b)} for every class index that has a
    predetermined color. Classes not present here (background, and any
    VOC class not assigned to a group) are left to the AI base colorizer.
    """
    class_to_group = build_class_to_group()
    index_to_color = {}
    for idx, name in enumerate(VOC_CLASSES):
        if name in class_to_group:
            index_to_color[idx] = GROUP_COLORS[class_to_group[name]]
    return index_to_color


if __name__ == "__main__":
    mapping = build_index_to_color()
    print(f"{len(mapping)} of {len(VOC_CLASSES)} classes have a predetermined color:")
    for idx, color in mapping.items():
        print(f"  [{idx:2d}] {VOC_CLASSES[idx]:12s} -> a={color[0]:+.2f} b={color[1]:+.2f}")
