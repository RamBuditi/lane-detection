import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- Dataset Class (with fraction parameter) ---
class CulaneDataset(Dataset):
    def __init__(self, data_root, list_path, transform=None, fraction=1.0):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        full_list_path = os.path.join(data_root, list_path)
        with open(full_list_path, 'r') as f:
            all_lines = f.readlines()
        num_samples_to_use = int(len(all_lines) * fraction)
        print(f"Using {num_samples_to_use} samples ({fraction*100:.1f}%) of the original dataset.")
        lines_to_use = all_lines[:num_samples_to_use]
        for line in lines_to_use:
            parts = line.strip().split()
            self.samples.append({
                "image": os.path.join(self.data_root, parts[0].lstrip('/')),
                "label": os.path.join(self.data_root, parts[1].lstrip('/'))
            })
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(sample["label"], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        label = label.long()
        return {"image": image, "label": label}

# --- Main Training Function ---
def main():
    # --- 1. Hyperparameters for Baseline Training ---
    DATA_ROOT = 'data/CULane'
    LIST_PATH = 'list/train_gt.txt'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 2      # Train for 2 full epochs
    TRAIN_FRACTION = 0.05 # Use 5% of the data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Transformations and Dataset ---
    train_transforms = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    dataset = CulaneDataset(
        data_root=DATA_ROOT,
        list_path=LIST_PATH,
        transform=train_transforms,
        fraction=TRAIN_FRACTION
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # --- 4. Model, Optimizer, and Loss Function ---
    print("Loading pre-trained DeepLabV3 model...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # --- 5. The Training Loop ---
    print("\nStarting baseline training run...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        total_loss = 0.0

        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)['out']
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

    # --- 6. Save the Trained Model ---
    print("\nTraining complete. Saving model...")
    MODEL_SAVE_PATH = "models/culane_deeplabv3_baseline.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()