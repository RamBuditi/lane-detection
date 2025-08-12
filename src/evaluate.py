import os
import torch
from torch.utils.data import Dataset, DataLoader # Correctly imported here
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import mlflow.pytorch
import cv2 # Correctly imported here

# Re-using the Dataset class from your train.py.
class CulaneDataset(Dataset):
    def __init__(self, data_root, list_path, transform=None, fraction=1.0):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        full_list_path = os.path.join(data_root, list_path)
        with open(full_list_path, 'r') as f:
            all_lines = f.readlines()
        num_samples_to_use = int(len(all_lines) * fraction)
        print(f"Using {num_samples_to_use} samples ({fraction*100:.1f}%) of the validation set.")
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

def calculate_iou(pred, label, num_classes):
    pred = pred.view(-1)
    label = label.view(-1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (label == cls)
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append(float(intersection) / float(union))
    return iou_per_class

def evaluate_model(args):
    # --- 1. Configuration ---
    DATA_ROOT = 'data/CULane'
    LIST_PATH = 'list/val_gt.txt'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    EVAL_FRACTION = 0.1

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Model from MLflow ---
    print(f"Loading model from MLflow Run ID: {args.run_id}")
    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    model.eval()
    print("Model loaded successfully.")

    # --- 4. Prepare Validation Dataset ---
    val_transforms = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_dataset = CulaneDataset(
        data_root=DATA_ROOT,
        list_path=LIST_PATH,
        transform=val_transforms,
        fraction=EVAL_FRACTION
    )
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 5. Run Evaluation Loop ---
    print("Running evaluation...")
    total_iou = []
    loop = tqdm(val_dataloader, desc="Evaluating")

    with torch.no_grad():
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(preds.shape[0]):
                iou_scores = calculate_iou(preds[i], labels[i], NUM_CLASSES)
                total_iou.append(iou_scores)
    
    # --- 6. Calculate and Log Final Metrics ---
    total_iou_np = np.array(total_iou)
    mean_iou_per_class = np.nanmean(total_iou_np, axis=0)
    mean_iou_overall = np.nanmean(total_iou_np)

    print("\n--- Evaluation Results ---")
    print(f"Overall Mean IoU (mIoU): {mean_iou_overall:.4f}")
    for cls, iou in enumerate(mean_iou_per_class):
        print(f"  - Class {cls} IoU: {iou:.4f}")
    
    # --- MLFLOW INTEGRATION: Log metrics back to the original run ---
    print(f"\nLogging metrics back to MLflow Run ID: {args.run_id}")
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metric("mIoU_validation", mean_iou_overall)
        for cls, iou in enumerate(mean_iou_per_class):
            mlflow.log_metric(f"mIoU_class_{cls}", iou)
    
    print("✅ Metrics logged successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model from an MLflow run.")
    parser.add_argument('--run_id', type=str, required=True, help="The MLflow Run ID of the model to evaluate.")
    args = parser.parse_args()
    
    evaluate_model(args)