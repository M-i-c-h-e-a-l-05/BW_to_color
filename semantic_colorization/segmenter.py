"""
Semantic segmentation for identifying object regions in a frame.

Uses torchvision's pretrained DeepLabV3 (MobileNetV3 backbone -- chosen
over the ResNet backbones specifically for real-time speed, at some cost
to segmentation accuracy). Trained on COCO, using the Pascal VOC 21-class
label set (see class_colors.py for the full list and its limitations).
"""
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


class Segmenter:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        self.model = deeplabv3_mobilenet_v3_large(weights=weights).to(self.device).eval()
        self.transforms = weights.transforms()
        self.categories = weights.meta["categories"]

    @torch.no_grad()
    def segment(self, rgb_uint8: "np.ndarray") -> "np.ndarray":
        """
        rgb_uint8: (H, W, 3) uint8 RGB image.
        Returns: (H, W) int64 array where each pixel is a class index
                 into self.categories (0 = background).
        """
        import numpy as np
        h, w = rgb_uint8.shape[:2]

        img_tensor = torch.from_numpy(rgb_uint8).permute(2, 0, 1)  # (3, H, W)
        batch = self.transforms(img_tensor).unsqueeze(0).to(self.device)

        output = self.model(batch)["out"]  # (1, num_classes, h', w')
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
        class_map = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        return class_map


if __name__ == "__main__":
    import numpy as np
    print("Smoke test: building segmenter and running on a random image.")
    print("(Requires downloading pretrained weights on first run.)")
    seg = Segmenter()
    dummy = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    class_map = seg.segment(dummy)
    print("Output shape:", class_map.shape, "dtype:", class_map.dtype)
    print("Unique classes detected:", np.unique(class_map))
