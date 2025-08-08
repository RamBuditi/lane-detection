import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class CulaneDataset(Dataset):
    """
    Custom PyTorch Dataset for the CULane dataset.
    This class is responsible for loading images and their corresponding
    segmentation masks.
    """
    def __init__(self, data_root, list_path):
        """
        Args:
            data_root (str): The root directory of the CULane dataset.
            list_path (str): The path to the text file containing the list of
                             image and label paths (e.g., 'list/train_gt.txt').
        """
        self.data_root = data_root
        self.samples = []

        # Read the list file and parse it
        # The file contains relative paths to images and labels
        full_list_path = os.path.join(data_root, list_path)
        with open(full_list_path, 'r') as f:
            for line in f:
                # The line is split into: image path, label path, and lane existence flags
                parts = line.strip().split()
                img_path = parts[0]
                label_path = parts[1]
                
                # We store the full paths to the image and label
                self.samples.append({
                    "image": os.path.join(self.data_root, img_path.lstrip('/')),
                    "label": os.path.join(self.data_root, label_path.lstrip('/'))
                })

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            A dictionary containing the 'image' and 'label'.
            - 'image': The loaded image as a NumPy array (H, W, C).
            - 'label': The loaded segmentation mask as a NumPy array (H, W).
        """
        sample = self.samples[idx]

        # Load image using OpenCV (BGR format)
        image = cv2.imread(sample["image"])
        if image is None:
            raise FileNotFoundError(f"Image not found: {sample['image']}")

        # Load label mask (grayscale)
        label = cv2.imread(sample["label"], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Label not found: {sample['label']}")

        # Here we would typically apply transformations (e.g., resizing, normalization,
        # converting to PyTorch tensors). We will add this in the next step.
        
        return {
            "image": image,
            "label": label
        }

# --- This is a test block to verify the Dataset works correctly ---
# We will run this block directly to test our class
if __name__ == "__main__":
    DATA_ROOT = 'data/CULane'
    LIST_PATH = 'list/train_gt.txt'

    print("Initializing dataset...")
    # Create an instance of our dataset
    dataset = CulaneDataset(data_root=DATA_ROOT, list_path=LIST_PATH)

    print(f"Dataset contains {len(dataset)} samples.")

    # Get the first sample from the dataset
    print("Fetching sample 0...")
    first_sample = dataset[0]

    # Check the data
    image = first_sample["image"]
    label = first_sample["label"]

    print(f"Image shape: {image.shape}, Image data type: {image.dtype}")
    print(f"Label shape: {label.shape}, Label data type: {label.dtype}")
    print(f"Unique values in label mask: {np.unique(label)}")

  

