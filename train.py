from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DeepCrackDataset
from model import UNet


def main():
    # -------------------------
    # Paths + file counts
    # -------------------------
    root = Path(__file__).resolve().parent

    train_img_dir = root / "train_img"
    train_lab_dir = root / "train_lab"
    test_img_dir  = root / "test_img"
    test_lab_dir  = root / "test_lab"

    assert train_img_dir.exists() and train_lab_dir.exists(), "Missing train_img/train_lab"
    assert test_img_dir.exists() and test_lab_dir.exists(), "Missing test_img/test_lab"

    train_imgs = sorted(train_img_dir.glob("*.jpg"))
    test_imgs  = sorted(test_img_dir.glob("*.jpg"))
    train_msks = sorted(train_lab_dir.glob("*.png"))
    test_msks  = sorted(test_lab_dir.glob("*.png"))

    print("Train images:", len(train_imgs))
    print("Test images :", len(test_imgs))
    print("Train masks :", len(train_msks))
    print("Test masks  :", len(test_msks))

    assert len(train_imgs) > 0 and len(train_msks) > 0, "No training files found"
    assert len(test_imgs) > 0 and len(test_msks) > 0, "No test files found"

    # -------------------------
    # Mask value check (sanity)
    # -------------------------
    m = np.array(Image.open(train_msks[0]).convert("L"))
    print("Unique values in a sample mask:", np.unique(m))

    if not (np.array_equal(np.unique(m), np.array([0, 255])) or np.array_equal(np.unique(m), np.array([0, 1]))):
        print("WARNING: mask unique values are not exactly [0,255] or [0,1]. "
              "We will still binarize with (mask>0).")

    # -------------------------
    # Dataset + DataLoader
    # -------------------------
    input_size = (256, 256)  # (W, H) for PIL
    batch_size = 8

    train_ds = DeepCrackDataset(train_img_dir, train_lab_dir, size=input_size)
    test_ds  = DeepCrackDataset(test_img_dir,  test_lab_dir,  size=input_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print("Train batches:", len(train_loader))
    print("Test batches :", len(test_loader))

    # Quick batch shape check
    images, masks = next(iter(train_loader))
    print("Batch images shape:", tuple(images.shape))  # (B,3,H,W)
    print("Batch masks shape :", tuple(masks.shape))   # (B,H,W)
    print("Mask values (unique in batch):", torch.unique(masks))

    # -------------------------
    # Model + training setup
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet(in_channels=3, num_classes=2, base=32).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # -------------------------
    # Train 
    # -------------------------
    def train_one_epoch(epoch: int):
        model.train()
        total_loss = 0.0

        for imgs, msks in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # (B,2,H,W)
            loss = criterion(logits, msks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d} | train loss: {avg_loss:.4f}")

    num_epochs = 10  #changed to 5 for quicker testing
    for ep in range(1, num_epochs + 1):
        train_one_epoch(ep)

    # Save weights
    weights_path = root / "unet_deepcrack.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model to {weights_path.name}")

    # -------------------------
    # Evaluate on TEST (Accuracy, IoU, F1)
    # -------------------------
    model.eval()

    TP = FP = FN = TN = 0

    with torch.no_grad():
        for imgs, msks in test_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)  # (B,H,W) 0/1

            logits = model(imgs)                 # (B,2,H,W)
            preds = torch.argmax(logits, dim=1)  # (B,H,W) 0/1

            TP += torch.sum((preds == 1) & (msks == 1)).item()
            FP += torch.sum((preds == 1) & (msks == 0)).item()
            FN += torch.sum((preds == 0) & (msks == 1)).item()
            TN += torch.sum((preds == 0) & (msks == 0)).item()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    jaccard  = TP / (TP + FP + FN + 1e-8)  # IoU for crack class
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n=== TEST RESULTS ===")
    print(f"Pixel Accuracy : {accuracy:.4f}")
    print(f"Jaccard (IoU)  : {jaccard:.4f}")
    print(f"F1-score      : {f1:.4f}")


if __name__ == "__main__":
    main()
