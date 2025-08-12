import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import json
import argparse

import mlflow
import mlflow.pytorch

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

def load_hyperparameters(config_path=None):
    """Load hyperparameters from file or use defaults"""
    default_params = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "train_fraction": 0.01,
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "augmentation_prob": 0.5,
        "num_epochs": 1
    }
    
    if config_path and os.path.exists(config_path):
        print(f"Loading hyperparameters from {config_path}")
        with open(config_path, 'r') as f:
            loaded_params = json.load(f)
        
        # Merge with defaults
        for key, value in loaded_params.items():
            if key in default_params:
                default_params[key] = value
        print("✅ Hyperparameters loaded successfully")
    else:
        print("Using default hyperparameters")
    
    return default_params

def calculate_model_complexity(model):
    """Calculate model complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024**2)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb
    }

def setup_optimizer_and_scheduler(model, params):
    """Setup optimizer and scheduler based on parameters"""
    
    # Optimizer setup
    if params['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    else:  # default to adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    
    # Scheduler setup
    scheduler = None
    if params['scheduler'].lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=params['num_epochs']
        )
    elif params['scheduler'].lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=max(1, params['num_epochs']//3), 
            gamma=0.1
        )
    elif params['scheduler'].lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95
        )
    # 'none' or any other value means no scheduler
    
    return optimizer, scheduler

def main():
    parser = argparse.ArgumentParser(description="Enhanced lane detection training")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to hyperparameter config JSON file')
    parser.add_argument('--experiment-name', type=str, default="Lane Detection - Enhanced",
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    args = parser.parse_args()
    
    # Load hyperparameters
    params = load_hyperparameters(args.config)
    
    # Update epochs from params
    NUM_EPOCHS = params.get('num_epochs', 1)
    
    # --- 1. Configuration ---
    DATA_ROOT = 'data/CULane'
    LIST_PATH = 'list/train_gt.txt'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5
    NUM_WORKERS = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # GPU memory optimization for RTX 4060
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        
        # Set memory fraction for RTX 4060 (8GB)
        if "4060" in gpu_name:
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of 8GB
            print("🔧 RTX 4060 detected, optimizing memory usage")

    # Configure MLflow tracking URI
    mlflow_dir = os.path.abspath("./mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log all hyperparameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log additional system info
        mlflow.log_param("device", str(device))
        mlflow.log_param("img_height", IMG_HEIGHT)
        mlflow.log_param("img_width", IMG_WIDTH)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_param("dataset", "CULane")
        mlflow.log_param("model_architecture", "DeepLabV3-ResNet50")

        # --- 3. Enhanced Transformations ---
        aug_prob = params['augmentation_prob']
        train_transforms = A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.HorizontalFlip(p=aug_prob),
            A.RandomBrightnessContrast(p=aug_prob),
            A.RandomGamma(p=aug_prob * 0.5),
            A.HueSaturationValue(p=aug_prob * 0.3),
            A.GaussNoise(p=aug_prob * 0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Create dataset and dataloader
        dataset = CulaneDataset(
            data_root=DATA_ROOT, 
            list_path=LIST_PATH, 
            transform=train_transforms, 
            fraction=params['train_fraction']
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=params['batch_size'], 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            drop_last=True
        )

        # --- 4. Model Setup ---
        print("Loading pre-trained DeepLabV3 model...")
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights='DeepLabV3_ResNet50_Weights.DEFAULT'
        )
        model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
        model.to(device)
        
        # Log model complexity
        complexity = calculate_model_complexity(model)
        for key, value in complexity.items():
            mlflow.log_param(key, value)
            print(f"{key}: {value:,}")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_and_scheduler(model, params)
        print(f"Optimizer: {params['optimizer']}")
        print(f"Scheduler: {params['scheduler']}")
        
        # Enhanced loss function with class weights (optional)
        loss_fn = torch.nn.CrossEntropyLoss()

        # --- 5. Enhanced Training Loop ---
        print(f"\nStarting enhanced training run for {NUM_EPOCHS} epochs...")
        
        best_loss = float('inf')
        training_losses = []
        learning_rates = []
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(loop):
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(images)['out']
                    loss = loss_fn(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Update metrics
                    current_loss = loss.item()
                    total_loss += current_loss
                    num_batches += 1
                    
                    # Update progress bar
                    loop.set_postfix(loss=current_loss)
                    
                    # Log batch-level metrics periodically
                    if batch_idx % 50 == 0:
                        mlflow.log_metric(f"batch_loss", current_loss, step=epoch * len(dataloader) + batch_idx)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ GPU OOM at batch {batch_idx}, clearing cache...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Calculate epoch metrics
            avg_loss = total_loss / max(num_batches, 1)
            training_losses.append(avg_loss)
            
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")
            
            # Log epoch metrics
            mlflow.log_metric("epoch_avg_loss", avg_loss, step=epoch)
            mlflow.log_metric("total_batches", num_batches, step=epoch)
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Update best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                mlflow.log_metric("best_loss", best_loss, step=epoch)
                
                # Save best model checkpoint
                best_model_path = "models/best_model_checkpoint.pth"
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'loss': avg_loss,
                    'hyperparameters': params
                }, best_model_path)
            
            # Step scheduler
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()
            
            # GPU memory cleanup
            torch.cuda.empty_cache()

        # --- 6. Enhanced Model Saving ---
        print("\nTraining complete. Saving model...")
        
        # Save final model
        MODEL_SAVE_PATH = "models/latest_model_run.pth"
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        # Save complete checkpoint
        CHECKPOINT_PATH = "models/final_checkpoint.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'hyperparameters': params,
            'training_losses': training_losses,
            'learning_rates': learning_rates,
            'final_loss': training_losses[-1] if training_losses else float('inf'),
            'best_loss': best_loss
        }, CHECKPOINT_PATH)
        
        print(f"Model saved to {MODEL_SAVE_PATH}")
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

        # --- 7. Enhanced MLflow Logging ---
        print("Preparing comprehensive MLflow logging...")
        
        try:
            # Log final metrics
            mlflow.log_metric("final_avg_loss", training_losses[-1] if training_losses else float('inf'))
            mlflow.log_metric("best_training_loss", best_loss)
            mlflow.log_metric("total_epochs_completed", len(training_losses))
            
            # Log training curve data
            for i, (loss, lr) in enumerate(zip(training_losses, learning_rates)):
                mlflow.log_metric("training_loss_curve", loss, step=i)
                mlflow.log_metric("lr_curve", lr, step=i)
            
            # Create model wrapper for MLflow
            class DeepLabV3Wrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    return self.model(x)['out']
            
            # Prepare model for logging
            model.cpu()
            model.eval()
            wrapped_model = DeepLabV3Wrapper(model)
            wrapped_model.eval()
            
            # Get sample input
            sample_batch = next(iter(dataloader))
            input_example = sample_batch['image'][0:1].cpu().float()
            
            # Test model
            with torch.no_grad():
                test_output = wrapped_model(input_example)
                print(f"Model test successful. Output shape: {test_output.shape}")
            
            # Log MLflow model
            print("Logging MLflow model...")
            mlflow.pytorch.log_model(
                pytorch_model=wrapped_model,
                artifact_path="model",
                input_example=input_example.numpy(),
                registered_model_name="culane-lane-detector"
            )
            print("✅ MLflow model logged and registered")
            
            # Log artifacts
            mlflow.log_artifact(MODEL_SAVE_PATH, artifact_path="model_files")
            mlflow.log_artifact(CHECKPOINT_PATH, artifact_path="checkpoints")
            
            if os.path.exists("models/best_model_checkpoint.pth"):
                mlflow.log_artifact("models/best_model_checkpoint.pth", artifact_path="checkpoints")
            
            # Log hyperparameter config if used
            if args.config and os.path.exists(args.config):
                mlflow.log_artifact(args.config, artifact_path="config")
            
            # Create and log training summary
            summary = {
                "run_id": run_id,
                "hyperparameters": params,
                "model_complexity": complexity,
                "training_summary": {
                    "total_epochs": len(training_losses),
                    "final_loss": training_losses[-1] if training_losses else float('inf'),
                    "best_loss": best_loss,
                    "loss_improvement": (training_losses[0] - training_losses[-1]) / training_losses[0] * 100 if len(training_losses) > 1 else 0
                },
                "system_info": {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "pytorch_version": torch.__version__
                }
            }
            
            summary_file = "training_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            mlflow.log_artifact(summary_file, artifact_path="summaries")
            os.remove(summary_file)
            
            # Log model architecture
            arch_file = "model_architecture.txt"
            with open(arch_file, "w") as f:
                f.write("Enhanced Model Architecture Summary\n")
                f.write("=" * 40 + "\n")
                f.write(f"Base Model: DeepLabV3 with ResNet50 backbone\n")
                f.write(f"Input Resolution: {IMG_HEIGHT} x {IMG_WIDTH}\n")
                f.write(f"Output Classes: {NUM_CLASSES}\n")
                f.write(f"Total Parameters: {complexity['total_parameters']:,}\n")
                f.write(f"Model Size: {complexity['model_size_mb']:.2f} MB\n")
                f.write(f"Optimizer: {params['optimizer']}\n")
                f.write(f"Scheduler: {params['scheduler']}\n")
                f.write(f"\nHyperparameters:\n")
                for key, value in params.items():
                    f.write(f"  {key}: {value}\n")
            
            mlflow.log_artifact(arch_file, artifact_path="model_info")
            os.remove(arch_file)
            
            print("✅ Enhanced MLflow logging completed")
            
        except Exception as e:
            print(f"❌ Error during enhanced MLflow logging: {e}")
            raise
        
        print(f"\n🎉 Enhanced training completed successfully!")
        print(f"📊 Final loss: {training_losses[-1] if training_losses else 'N/A':.4f}")
        print(f"🏆 Best loss: {best_loss:.4f}")
        print(f"📈 Loss improvement: {((training_losses[0] - training_losses[-1]) / training_losses[0] * 100) if len(training_losses) > 1 else 0:.1f}%")
        print(f"📁 Run ID: {run_id}")
        print(f"🌐 MLflow UI: http://localhost:5000")
        
        # Return metrics for pipeline use
        return {
            "run_id": run_id,
            "final_loss": training_losses[-1] if training_losses else float('inf'),
            "best_loss": best_loss,
            "hyperparameters": params
        }

if __name__ == "__main__":
    main()