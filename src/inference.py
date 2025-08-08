import os
import cv2
import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def run_inference():
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
    # Instantiate the same model architecture as during training
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
    # Modify the final layer to match the number of classes
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
    
    # Load your trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # Set the model to evaluation mode (this is very important!)
    print("Model loaded successfully.")

    # --- 4. Get a Test Image ---
    # We'll grab the first image from the official test list
    test_list_path = os.path.join(DATA_ROOT, 'list/test.txt')
    with open(test_list_path, 'r') as f:
        # The test list only contains the relative paths to images
        test_image_rel_path = f.readline().strip()
    
    test_image_full_path = os.path.join(DATA_ROOT, test_image_rel_path.lstrip('/'))
    print(f"Running inference on: {test_image_full_path}")

    # --- 5. Preprocess the Image for the Model ---
    # NOTE: Transformations must be the SAME as validation/training, but without data augmentation.
    transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Load the original image using OpenCV for display later
    original_image = cv2.imread(test_image_full_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Apply transformations to prepare the image for the model
    transformed = transform(image=original_image)
    input_tensor = transformed['image'].unsqueeze(0).to(device) # Add batch dimension (B, C, H, W) and send to device

    # --- 6. Run Inference ---
    with torch.no_grad(): # Disable gradient calculation for efficiency
        output = model(input_tensor)['out']

    # --- 7. Post-process the Output ---
    # The output is (B, C, H, W). We take the argmax along the class dimension (C) to get the prediction
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # --- 8. Visualize the Result ---
    # Create a color map for the segmentation mask for better visualization
    # Class 0: background (black), Class 1-4: lanes (different colors)
    color_map = np.array([[0, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0]], dtype=np.uint8)
    prediction_color = color_map[prediction]

    # Resize original image to match prediction size for a nice overlay
    resized_original = cv2.resize(original_image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Blend the original image with the colored prediction mask
    overlay = cv2.addWeighted(resized_original, 0.6, prediction_color, 0.4, 0)

    # Display the results using Matplotlib
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
    run_inference()