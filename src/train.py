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
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

import mlflow
import mlflow.pytorch

def get_default_config() -> Dict[str, Any]:
    """Get default configuration - can be imported by other modules"""
    return {
        'data_root': 'data/CULane',
        'train_list': 'list/train_gt.txt',
        'val_list': 'list/val_gt.txt',
        'img_height': 288,
        'img_width': 800,
        'num_classes': 5,
        'num_workers': 4,
        'class_names': ['background', 'lane_1', 'lane_2', 'lane_3', 'lane_4']
    }

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

class ValidationMetrics:
    """Comprehensive validation metrics calculator"""
    
    def __init__(self, num_classes: int, class_names: list = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f'class_{i}' for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.total_iou = []
        self.pixel_accuracies = []
        self.class_pixel_counts = [0] * self.num_classes
        self.confusion_matrices = []
    
    def calculate_iou(self, pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, list]:
        """Calculate IoU for each class and mean IoU"""
        pred = pred.view(-1).cpu().numpy()
        label = label.view(-1).cpu().numpy()
        
        iou_per_class = []
        for cls in range(self.num_classes):
            pred_inds = (pred == cls)
            target_inds = (label == cls)
            intersection = np.logical_and(pred_inds, target_inds).sum()
            union = np.logical_or(pred_inds, target_inds).sum()
            
            if union == 0:
                iou_per_class.append(0.0)  # No pixels of this class
            else:
                iou_per_class.append(intersection / union)
        
        mean_iou = np.mean([iou for iou in iou_per_class if iou > 0])  # Exclude empty classes
        return mean_iou, iou_per_class
    
    def calculate_pixel_accuracy(self, pred: torch.Tensor, label: torch.Tensor) -> float:
        """Calculate pixel-wise accuracy"""
        pred = pred.view(-1).cpu().numpy()
        label = label.view(-1).cpu().numpy()
        return (pred == label).mean()
    
    def update(self, pred_batch: torch.Tensor, label_batch: torch.Tensor):
        """Update metrics with a batch of predictions"""
        batch_size = pred_batch.shape[0]
        
        for i in range(batch_size):
            # Calculate IoU
            miou, class_ious = self.calculate_iou(pred_batch[i], label_batch[i])
            self.total_iou.append(miou)
            
            # Calculate pixel accuracy
            pixel_acc = self.calculate_pixel_accuracy(pred_batch[i], label_batch[i])
            self.pixel_accuracies.append(pixel_acc)
            
            # Update class pixel counts
            pred_flat = pred_batch[i].view(-1).cpu().numpy()
            for cls in range(self.num_classes):
                self.class_pixel_counts[cls] += (pred_flat == cls).sum()
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics"""
        if not self.total_iou:
            return {'mean_iou': 0.0, 'pixel_accuracy': 0.0}
        
        metrics = {
            'mean_iou': np.mean(self.total_iou),
            'pixel_accuracy': np.mean(self.pixel_accuracies),
            'std_iou': np.std(self.total_iou),
            'median_iou': np.median(self.total_iou)
        }
        
        return metrics

def validate_model(model: torch.nn.Module, 
                  val_dataloader: DataLoader, 
                  device: torch.device, 
                  num_classes: int = 5,
                  class_names: list = None,
                  quiet: bool = False) -> Dict[str, float]:
    """Comprehensive model validation"""
    model.eval()
    val_metrics = ValidationMetrics(num_classes, class_names)
    
    val_loss = 0.0
    num_batches = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    
    if not quiet:
        print("Running validation...")
        val_pbar = tqdm(val_dataloader, desc="Validation")
    else:
        val_pbar = val_dataloader
        
    with torch.no_grad():
        for batch in val_pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            outputs = model(images)['out']
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate loss
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            val_metrics.update(preds, labels)
            
            # Update progress
            if not quiet:
                current_metrics = val_metrics.compute_metrics()
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mIoU': f'{current_metrics["mean_iou"]:.4f}',
                    'acc': f'{current_metrics["pixel_accuracy"]:.4f}'
                })
    
    # Compute final metrics
    final_metrics = val_metrics.compute_metrics()
    final_metrics['val_loss'] = val_loss / max(num_batches, 1)
    
    return final_metrics

def load_hyperparameters(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load hyperparameters with enhanced defaults"""
    default_params = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "train_fraction": 0.01,
        "val_fraction": 0.05,  # Separate validation fraction
        "optimizer": "adam",
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "augmentation_prob": 0.5,
        "num_epochs": 1,
        "dropout_rate": 0.1,
        "early_stopping_patience": 5,
        "validation_frequency": 1,  # Validate every N epochs
        "save_best_model": True,
        "gradient_clip_norm": 1.0
    }
    
    if config_path and os.path.exists(config_path):
        print(f"Loading hyperparameters from {config_path}")
        with open(config_path, 'r') as f:
            loaded_params = json.load(f)
        
        for key, value in loaded_params.items():
            if key in default_params:
                default_params[key] = value
        print("✅ Hyperparameters loaded successfully")
    else:
        print("Using default hyperparameters")
    
    return default_params

def setup_optimizer_and_scheduler(model: torch.nn.Module, params: Dict[str, Any]):
    """Enhanced optimizer and scheduler setup"""
    
    # Optimizer setup with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    if params['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': params['learning_rate'] * 0.1},  # Lower LR for backbone
            {'params': classifier_params, 'lr': params['learning_rate']}
        ], weight_decay=params['weight_decay'])
    elif params['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': params['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': params['learning_rate']}
        ], weight_decay=params['weight_decay'], momentum=0.9)
    else:  # adam
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': params['learning_rate'] * 0.1},
            {'params': classifier_params, 'lr': params['learning_rate']}
        ], weight_decay=params['weight_decay'])
    
    # Enhanced scheduler setup
    num_epochs = params.get('num_epochs', 1)
    scheduler = None
    
    if params['scheduler'].lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, num_epochs//3), T_mult=2
        )
    elif params['scheduler'].lower() == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[num_epochs//3, 2*num_epochs//3], 
            gamma=0.1
        )
    elif params['scheduler'].lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif params['scheduler'].lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    
    return optimizer, scheduler

def create_enhanced_model(num_classes: int, dropout_rate: float = 0.1) -> torch.nn.Module:
    """Create enhanced model with regularization"""
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights='DeepLabV3_ResNet50_Weights.DEFAULT'
    )
    
    # Replace classifier with enhanced version
    model.classifier = torch.nn.Sequential(
        model.classifier[0],  # ASPP
        torch.nn.Conv2d(256, 256, 3, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout2d(p=dropout_rate),
        torch.nn.Conv2d(256, num_classes, 1)
    )
    
    return model

def plot_training_curves(train_losses: list, 
                        val_losses: list, 
                        val_mious: list, 
                        save_path: str = None):
    """Plot and save training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation loss
    ax2.plot(val_losses, label='Validation Loss', color='red')
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Validation mIoU
    ax3.plot(val_mious, label='Validation mIoU', color='green')
    ax3.set_title('Validation mIoU')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mIoU')
    ax3.legend()
    ax3.grid(True)
    
    # Combined plot
    ax4_twin = ax4.twinx()
    ax4.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    ax4.plot(val_losses, label='Val Loss', color='red', alpha=0.7)
    ax4_twin.plot(val_mious, label='Val mIoU', color='green', alpha=0.7)
    ax4.set_title('Training Overview')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4_twin.set_ylabel('mIoU')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Enhanced lane detection training with validation")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to hyperparameter config JSON file')
    parser.add_argument('--experiment-name', type=str, default="Lane Detection - Enhanced",
                       help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default=None,
                       help='MLflow run name')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip validation during training')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity for hyperparameter tuning')
    parser.add_argument('--promotion-threshold', type=float, default=0.3,
                       help='mIoU threshold for model promotion')
    args = parser.parse_args()
    
    # Load configuration
    params = load_hyperparameters(args.config)
    config = get_default_config()
    
    # GPU optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if not args.quiet:
            print(f"GPU: {gpu_name}")
        if "4060" in gpu_name:
            torch.cuda.set_per_process_memory_fraction(0.7)
            if not args.quiet:
                print("🔧 RTX 4060 detected, optimizing memory usage")

    # MLflow setup
    mlflow_dir = os.path.abspath("./mlruns")
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name) as run:
        run_id = run.info.run_id
        if not args.quiet:
            print(f"MLflow Run ID: {run_id}")
        else:
            print(f"MLflow Run ID: {run_id}")  # Always show this for hyperparameter extraction
        
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in config.items():
            if key not in ['class_names']:  # Skip non-serializable
                mlflow.log_param(f"config_{key}", value)

        # Enhanced transformations
        aug_prob = params['augmentation_prob']
        train_transforms = A.Compose([
            A.Resize(height=config['img_height'], width=config['img_width']),
            A.HorizontalFlip(p=aug_prob),
            A.RandomBrightnessContrast(p=aug_prob),
            A.RandomGamma(p=aug_prob * 0.5),
            A.HueSaturationValue(p=aug_prob * 0.3),
            A.GaussNoise(p=aug_prob * 0.2),
            A.ElasticTransform(p=aug_prob * 0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        val_transforms = A.Compose([
            A.Resize(height=config['img_height'], width=config['img_width']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Create datasets
        train_dataset = CulaneDataset(
            data_root=config['data_root'], 
            list_path=config['train_list'], 
            transform=train_transforms, 
            fraction=params['train_fraction']
        )
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True, 
            num_workers=config['num_workers'],
            pin_memory=True,
            persistent_workers=True if config['num_workers'] > 0 else False,
            drop_last=True
        )
        
        # Validation dataset (if not skipped)
        val_dataloader = None
        if not args.no_validation:
            val_dataset = CulaneDataset(
                data_root=config['data_root'], 
                list_path=config['val_list'], 
                transform=val_transforms, 
                fraction=params['val_fraction']
            )
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=params['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'],
                pin_memory=True
            )

        # Create enhanced model
        model = create_enhanced_model(config['num_classes'], params['dropout_rate'])
        model.to(device)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = setup_optimizer_and_scheduler(model, params)
        
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training variables
        num_epochs = params['num_epochs']
        best_val_miou = 0.0
        patience_counter = 0
        
        # Tracking lists
        train_losses = []
        val_losses = []
        val_mious = []
        val_pixel_accs = []
        
        if not args.quiet:
            print(f"\nStarting enhanced training for {num_epochs} epochs...")
            print(f"Training samples: {len(train_dataset)}")
            if val_dataloader:
                print(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            # Progress bar setup based on quiet mode
            if args.quiet:
                train_pbar = train_dataloader  # No progress bar
                print_interval = max(1, len(train_dataloader) // 3)  # Print 3 times per epoch
            else:
                train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
                print_interval = 50
            
            for batch_idx, batch in enumerate(train_pbar):
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    # Forward pass
                    outputs = model(images)['out']
                    loss = loss_fn(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    if params.get('gradient_clip_norm', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient_clip_norm'])
                    
                    optimizer.step()
                    
                    # Update metrics
                    current_loss = loss.item()
                    epoch_train_loss += current_loss
                    num_batches += 1
                    
                    # Update progress display
                    if not args.quiet:
                        train_pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                    elif batch_idx % print_interval == 0:
                        progress = (batch_idx + 1) / len(train_dataloader) * 100
                        print(f"Epoch {epoch+1}/{num_epochs} - Progress: {progress:.1f}% - Loss: {current_loss:.4f}")
                    
                    # Log batch metrics periodically
                    if batch_idx % 50 == 0:
                        global_step = epoch * len(train_dataloader) + batch_idx
                        mlflow.log_metric("batch_loss", current_loss, step=global_step)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠️ GPU OOM at batch {batch_idx}, clearing cache...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
            
            # Calculate average training loss
            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase
            val_metrics = {}
            if val_dataloader and (epoch + 1) % params['validation_frequency'] == 0:
                val_metrics = validate_model(
                    model, val_dataloader, device, 
                    config['num_classes'], config['class_names'], quiet=args.quiet
                )
                
                val_losses.append(val_metrics['val_loss'])
                val_mious.append(val_metrics['mean_iou'])
                val_pixel_accs.append(val_metrics['pixel_accuracy'])
                
                print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, "
                      f"mIoU: {val_metrics['mean_iou']:.4f}, "
                      f"Pixel Acc: {val_metrics['pixel_accuracy']:.4f}")
                
                # Early stopping and best model saving
                if val_metrics['mean_iou'] > best_val_miou:
                    best_val_miou = val_metrics['mean_iou']
                    patience_counter = 0
                    
                    if params['save_best_model']:
                        best_model_path = "models/best_model_checkpoint.pth"
                        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'val_miou': val_metrics['mean_iou'],
                            'hyperparameters': params
                        }, best_model_path)
                        if not args.quiet:
                            print(f"💾 New best model saved (mIoU: {best_val_miou:.4f})")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= params['early_stopping_patience']:
                    if not args.quiet:
                        print(f"🛑 Early stopping triggered (patience: {params['early_stopping_patience']})")
                    break
            
            # Log epoch metrics
            mlflow.log_metric("epoch_train_loss", avg_train_loss, step=epoch)
            if val_metrics:
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"epoch_{key}", value, step=epoch)
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics.get('mean_iou', avg_train_loss))
                else:
                    scheduler.step()
                
                current_lr = optimizer.param_groups[0]['lr']
                mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # GPU memory cleanup
            torch.cuda.empty_cache()

        # Final model saving and logging
        if not args.quiet:
            print("\nTraining complete. Saving final model...")
        
        final_model_path = "models/latest_model_run.pth"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        
        # Create and save training curves
        if val_dataloader and val_losses and not args.quiet:
            curves_path = "training_curves.png"
            plot_training_curves(train_losses, val_losses, val_mious, curves_path)
            mlflow.log_artifact(curves_path)
            os.remove(curves_path)
        
        # Enhanced MLflow logging
        try:
            # Log final metrics
            final_train_loss = train_losses[-1] if train_losses else float('inf')
            mlflow.log_metric("final_train_loss", final_train_loss)
            
            # Performance metrics for model registration
            performance_metrics = {
                'final_train_loss': final_train_loss,
                'mean_iou': val_mious[-1] if val_mious else 0.0,
                'best_val_miou': best_val_miou,
                'pixel_accuracy': val_pixel_accs[-1] if val_pixel_accs else 0.0
            }
            
            if val_mious:
                mlflow.log_metric("final_val_miou", val_mious[-1])
                mlflow.log_metric("best_val_miou", best_val_miou)
                mlflow.log_metric("val_miou_improvement", val_mious[-1] - val_mious[0] if len(val_mious) > 1 else 0)
            
            # Model wrapper for MLflow
            class DeepLabV3Wrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    return self.model(x)['out']
            
            # Prepare for logging
            model.cpu()
            model.eval()
            wrapped_model = DeepLabV3Wrapper(model)
            
            # Get sample input
            sample_batch = next(iter(train_dataloader))
            input_example = sample_batch['image'][0:1].cpu().float()
            
            # Log model to MLflow (this creates the model in the run)
        #     mlflow.pytorch.log_model(
        #         pytorch_model=wrapped_model,
        #         artifact_path="model",
        #         input_example=input_example.numpy()
        #     )
            
        #     # Now register the model with proper aliases
        #     model_name = "culane-lane-detector"
            
        #     # Mark if hyperparameters were tuned (check if config file was used)
        #     params['tuned'] = args.config is not None
            
        #     registration_result = register_model_with_proper_aliases(
        #         run_id=run_id,
        #         model_name=model_name,
        #         performance_metrics=performance_metrics,
        #         hyperparams=params,
        #         promotion_threshold=args.promotion_threshold
        #     )
            
        #     # Log registration results
        #     if registration_result['success']:
        #         mlflow.log_param("model_version", registration_result['version'])
        #         mlflow.log_param("model_alias", registration_result['alias'])
        #         mlflow.log_param("promotion_status", registration_result['promotion_status'])
                
        #         if not args.quiet:
        #             print(f"✅ Model registered successfully:")
        #             print(f"   Version: {registration_result['version']}")
        #             print(f"   Alias: {registration_result['alias']}")
        #             print(f"   Status: {registration_result['promotion_status']}")
        #     else:
        #         print(f"❌ Model registration failed: {registration_result.get('error', 'Unknown error')}")
            
        #     # Log artifacts
        #     mlflow.log_artifact(final_model_path, artifact_path="model_files")
        #     if os.path.exists("models/best_model_checkpoint.pth"):
        #         mlflow.log_artifact("models/best_model_checkpoint.pth", artifact_path="checkpoints")
            
        #     if not args.quiet:
        #         print("✅ Enhanced MLflow logging completed")
            
        except Exception as e:
            print(f"❌ Error during MLflow logging: {e}")
            raise
        
        # Training summary
        if not args.quiet:
            print(f"\n🎉 Enhanced training completed!")
            print(f"📊 Final train loss: {final_train_loss:.4f}")
            if val_mious:
                print(f"🏆 Best validation mIoU: {best_val_miou:.4f}")
                print(f"📈 Final validation mIoU: {val_mious[-1]:.4f}")
            print(f"📁 Run ID: {run_id}")
            print(f"🌐 MLflow UI: http://localhost:5000")
        
        return {
            "run_id": run_id,
            "final_train_loss": final_train_loss,
            "best_val_miou": best_val_miou,
            "final_val_miou": val_mious[-1] if val_mious else 0.0,
            "hyperparameters": params,
            "registration_result": registration_result if 'registration_result' in locals() else None
        }

if __name__ == "__main__":
    main()