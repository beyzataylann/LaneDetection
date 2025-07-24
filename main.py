import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# --- Ayarlar ---
IMG_DIR = "10k/train"           # Eğitim görsellerinin klasörü
MASK_DIR = "labels/train"       # Maskelerin klasörü
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 2
PATIENCE = 5                    # Early stopping için bekleme sayısı


# --- Görsel Dönüştürme ---
def get_transforms():
    return T.Compose([
        T.Resize((256, 512)),
        T.ToTensor()
    ])


def mask_to_binary(mask_tensor):
    return (mask_tensor == 2).long()  # sınıf 2: drivable area


# --- Dataset Sınıfı ---
class BDD100KDataset(Dataset):
    def __init__(self, img_dir, mask_dir, max_samples=200):  # max_samples eklendi
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))[:max_samples]
        self.masks = sorted(os.listdir(mask_dir))[:max_samples]
        self.transforms = get_transforms()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transforms(image)
        mask = T.Resize((256, 512))(mask)
        mask = T.ToTensor()(mask)[0]  # grayscale olarak al
        binary_mask = mask_to_binary(mask)

        return image, binary_mask


# --- IoU Hesaplama ---
def compute_iou(preds, targets):
    preds = preds.bool()
    targets = targets.bool()
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# --- Model Getirme ---
def get_deeplab_model():
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    return model


# --- Ana Eğitim Fonksiyonu ---
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_deeplab_model().to(device)

    dataset = BDD100KDataset(IMG_DIR, MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_iou = 0.0
    epochs_no_improve = 0

    print("Eğitime başlanıyor...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        total_iou = 0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            total_iou += compute_iou(preds, masks)

        avg_loss = total_loss / len(loader)
        avg_iou = total_iou / len(loader)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}, IoU: {avg_iou:.4f}")

        # Early stopping kontrolü
        if avg_iou > best_iou:
            best_iou = avg_iou
            epochs_no_improve = 0
            torch.save(model.state_dict(), "deeplabv3_bdd100k.pth")
            print("Yeni en iyi model kaydedildi.")
        else:
            epochs_no_improve += 1
            print(f"İyileşme yok: {epochs_no_improve}/{PATIENCE} epoch")

        if epochs_no_improve >= PATIENCE:
            print("Early stopping: Eğitim durduruldu.")
            break


# --- Ana Fonksiyon ---
if __name__ == "__main__":
    train()
