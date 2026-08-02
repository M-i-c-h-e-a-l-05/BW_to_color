"""
Dataset for colorization training.

Takes a folder of ORDINARY COLOR IMAGES (jpg/png). No labeling needed --
we generate the training pairs ourselves:
    input  = L channel  (i.e. what the image looks like in grayscale)
    target = a,b channels (the color info the model must learn to predict)

This is why colorization is "self-supervised": any color photo becomes a
free training example.
"""
import os
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ColorizationDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 256, augment: bool = True):
        self.image_dir = image_dir
        self.paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        ]
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {image_dir}")

        aug = [T.Resize((image_size, image_size))]
        if augment:
            aug.append(T.RandomHorizontalFlip())
        self.transform = T.Compose(aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)

        img_np = np.array(img).astype(np.float32) / 255.0  # [0,1]
        lab = rgb2lab(img_np).astype(np.float32)            # L:[0,100], a,b:[-128,127]

        L = lab[:, :, 0]
        ab = lab[:, :, 1:]

        # Normalize to roughly [-1, 1] so the network trains stably
        L = (L / 50.0) - 1.0          # [0,100]   -> [-1, 1]
        ab = ab / 110.0                # [-110,110]-> ~[-1, 1]

        L_tensor = torch.from_numpy(L).unsqueeze(0).float()        # (1, H, W)
        ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).float()  # (2, H, W)

        return L_tensor, ab_tensor


def lab_to_rgb(L_tensor, ab_tensor):
    """
    Inverse of the normalization above. Used at inference time to turn
    model output back into a viewable RGB image.
    L_tensor:  (1, H, W) or (H, W), normalized [-1,1]
    ab_tensor: (2, H, W), normalized [-1,1]
    Returns: (H, W, 3) uint8 RGB numpy array
    """
    from skimage.color import lab2rgb

    L = L_tensor.squeeze().cpu().numpy()
    ab = ab_tensor.cpu().numpy()

    L = (L + 1.0) * 50.0            # back to [0, 100]
    ab = ab * 110.0                  # back to [-110, 110]

    lab = np.zeros((L.shape[0], L.shape[1], 3), dtype=np.float32)
    lab[:, :, 0] = L
    lab[:, :, 1] = ab[0]
    lab[:, :, 2] = ab[1]

    rgb = lab2rgb(lab)
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb


if __name__ == "__main__":
    print("This module expects a directory of color images to test with.")
    print("Example:")
    print("  ds = ColorizationDataset('/path/to/color/images', image_size=256)")
    print("  L, ab = ds[0]")
    print("  print(L.shape, ab.shape)  # (1,256,256) (2,256,256)")
