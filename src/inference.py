import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import random
import argparse
import mlflow.pytorch

def run_inference(args):
    # --- 1. Configuration ---
    DATA_ROOT = 'data/CULane' 
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    RUN_ID = args.run_id

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load Model From MLflow ---
    print(f"Loading model from MLflow Run ID: {RUN_ID}")
    model_uri = f"runs:/{RUN_ID}/model"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)
    print("Model loaded successfully.")
    model.eval()

    # --- 4. Get a Test Image ---
    if args.image:
        test_image_full_path = args.image
        if not os.path.exists(test_image_full_path):
            print(f"Error: Image not found at {test_image_full_path}")
            return
    else:
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
        output = model(input_tensor)

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

    # Define the output directory and save the plot there
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "inference_result.png")
    plt.savefig(output_path)
    print(f"Inference result plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run lane detection inference on an image using a model from MLflow.")
    parser.add_argument('--run_id', type=str, required=True, help="MLflow Run ID of the model to use.")
    parser.add_argument('--image', type=str, help="Optional: Path to a specific image file.")
    args = parser.parse_args()
    
    run_inference(args)