import os
import cv2
import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import random # --- NEW: For picking random images
import argparse # --- NEW: For command-line arguments

def run_inference(args):
    # --- 1. Configuration ---
    DATA_ROOT = 'data/CULane'
    MODEL_PATH = 'models/culane_deeplabv3_baseline.pth'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Model ---
    print("Loading trained model...")
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 4. Get a Test Image (Now with options!) ---
    # --- NEW LOGIC ---
    if args.image:
        # Use the image provided by the user
        test_image_full_path = args.image
        if not os.path.exists(test_image_full_path):
            print(f"Error: Image not found at {test_image_full_path}")
            return
    else:
        # Pick a random image from the test list
        print("No specific image provided, picking a random one from the test set...")
        test_list_path = os.path.join(DATA_ROOT, 'list/test.txt')
        with open(test_list_path, 'r') as f:
            test_images = f.readlines()
        
        test_image_rel_path = random.choice(test_images).strip()
        test_image_full_path = os.path.join(DATA_ROOT, test_image_rel_path.lstrip('/'))
    
    print(f"Running inference on: {test_image_full_path}")

    # --- 5. Preprocess the Image ---
    transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    original_image = cv2.imread(test_image_full_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=original_image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # --- 6. Run Inference ---
    with torch.no_grad():
        output = model(input_tensor)['out']

    # --- 7. Post-process the Output ---
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # --- 8. Visualize the Result ---
    color_map = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0]], dtype=np.uint8)
    prediction_color = color_map[prediction]

    resized_original = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))
    overlay = cv2.addWeighted(resized_original, 0.6, prediction_color, 0.4, 0)

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Test Image', fontsize=16)
    ax[0].axis('off')

    ax[1].imshow(prediction_color)
    ax[1].set_title('Predicted Lane Mask', fontsize=16)
    ax[1].axis('off')

    ax[2].imshow(overlay)
    ax[2].set_title('Prediction Overlay', fontsize=16)
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- NEW: Setup argument parser ---
    parser = argparse.ArgumentParser(description="Run lane detection inference on an image.")
    parser.add_argument('--image', type=str, help="Optional: Path to a specific image file.")
    args = parser.parse_args()
    
    run_inference(args)