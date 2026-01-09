from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class DeepCrackDataset(Dataset):
    """
    DeepCrack crack segmentation dataset.

    Returns:
      image: torch.float32 tensor (3, H, W) in [0, 1]
      mask : torch.int64 tensor (H, W) with values:
             0 = background
             1 = crack
    """

    def __init__(self, img_dir: str | Path, mask_dir: str | Path, size=(256, 256)):
        """
        size: (width, height) for PIL resize (both image and mask).
        """
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.size = tuple(size)

        self.img_files = sorted(self.img_dir.glob("*.jpg"))
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No .jpg files found in {self.img_dir}")

        # Map masks by stem name
        self.mask_map = {p.stem: p for p in self.mask_dir.glob("*.png")}
        if len(self.mask_map) == 0:
            raise FileNotFoundError(f"No .png files found in {self.mask_dir}")

        missing = [p.name for p in self.img_files if p.stem not in self.mask_map]
        if missing:
            raise RuntimeError(
                f"Missing masks for {len(missing)} images (showing up to 5): {missing[:5]}"
            )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        mask_path = self.mask_map[img_path.stem]

        # Read
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        # Resize (mask uses NEAREST to preserve labels)
        img = img.resize(self.size, resample=Image.BILINEAR)
        mask = mask.resize(self.size, resample=Image.NEAREST)

        # To numpy
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        mask = np.array(mask, dtype=np.uint8)          # (H, W), values 0/255

        # Binarize: crack=1 if >0 (covers 255 or any nonzero)
        mask = (mask > 0).astype(np.uint8)

        # To torch
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3, H, W)
        mask_t = torch.from_numpy(mask).long().contiguous()          # (H, W)

        return img_t, mask_t
