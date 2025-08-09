import torch
import torchvision
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# We can import the dataset class directly from our train script
from train import CulaneDataset

def calculate_iou(pred, label, num_classes):
    """Calculates IoU for a single image."""
    iou_list = []
    pred = pred.view(-1)
    label = label.view(-1)

    # Note: We skip class 0 (background) in the calculation
    for cls in range(1, num_classes):
        pred_inds = (pred == cls)
        target_inds = (label == cls)
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        # Add a small epsilon to avoid division by zero
        if union == 0:
            iou_list.append(float('nan')) # or 1.0 if no union and no intersection
        else:
            iou_list.append(intersection / union)
            
    return iou_list

def main():
    # --- 1. Configuration ---
    DATA_ROOT = 'data/CULane'
    # Use the validation set to evaluate performance
    LIST_PATH = 'list/val_gt.txt' # Or 'list/val_gt.txt' depending on your dataset structure
    MODEL_PATH = 'models/culane_deeplabv3_baseline.pth'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5
    BATCH_SIZE = 8 # Can be larger than training batch size

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Model ---
    print("Loading trained model...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval() # CRITICAL: Set model to evaluation mode

    # --- 4. Validation Dataset and DataLoader ---
    # IMPORTANT: No data augmentation for validation/evaluation!
    val_transforms = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_dataset = CulaneDataset(data_root=DATA_ROOT, list_path=LIST_PATH, transform=val_transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 5. The Evaluation Loop ---
    print("\nStarting evaluation...")
    total_iou = []
    
    with torch.no_grad(): # CRITICAL: No gradients needed for evaluation
        loop = tqdm(val_dataloader, desc="Evaluating")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each image in the batch
            for i in range(preds.shape[0]):
                iou_scores = calculate_iou(preds[i], labels[i], NUM_CLASSES)
                total_iou.append(iou_scores)
    
    # --- 6. Calculate and Print Final Metrics ---
    total_iou_np = np.array(total_iou)
    # Calculate mean IoU for each class, ignoring NaNs
    mean_iou_per_class = np.nanmean(total_iou_np, axis=0)
    # Calculate overall mean IoU (mIoU)
    mIoU = np.nanmean(total_iou_np)

    print("\n--- Evaluation Complete ---")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    for i, iou in enumerate(mean_iou_per_class):
        print(f"  - IoU for Lane {i+1}: {iou:.4f}")

if __name__ == '__main__':
    main()