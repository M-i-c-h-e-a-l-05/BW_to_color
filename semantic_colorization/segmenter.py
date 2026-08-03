"""
Semantic segmentation using SegFormer trained on ADE20K.

Model:
    nvidia/segformer-b2-finetuned-ade-512-512

Dataset:
    ADE20K (150 semantic classes)

Returns:
    H x W semantic segmentation mask
"""

import numpy as np
import torch
from PIL import Image
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class Segmenter:

    def __init__(self, device=None):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading SegFormer ADE20K model...")

        self.processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        ).to(self.device)

        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

        print(f"Loaded {len(self.id2label)} ADE20K classes")

    @torch.no_grad()
    def segment(self, rgb_uint8: np.ndarray):

        image = Image.fromarray(rgb_uint8)

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        prediction = logits.argmax(dim=1)[0]

        return prediction.cpu().numpy()

    def class_name(self, class_id):

        return self.id2label[int(class_id)]

    def detected_classes(self, class_map):

        ids = np.unique(class_map)

        names = []

        for idx in ids:

            if idx in self.id2label:

                names.append(self.id2label[idx])

        return sorted(names)

    def class_mask(self, class_map, class_name):

        if class_name not in self.label2id:

            raise ValueError(f"{class_name} not found")

        idx = self.label2id[class_name]

        return class_map == idx


if __name__ == "__main__":

    print("Testing ADE20K Segmenter")

    img = np.random.randint(
        0,
        255,
        (512, 512, 3),
        dtype=np.uint8
    )

    seg = Segmenter()

    mask = seg.segment(img)

    print("Mask Shape:", mask.shape)

    print()

    print("Detected Classes:")

    for cls in seg.detected_classes(mask):

        print(cls)