import os
import torch
import mlflow
import mlflow.pytorch
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
import time
from itertools import product
import subprocess
import tempfile
import shutil

class HyperparameterTuner:
    """
    Parallel hyperparameter tuning optimized for RTX 4060 (8GB VRAM)
    Uses process-based parallelism to avoid CUDA context conflicts
    """
    
    def __init__(self, max_workers=2, gpu_memory_fraction=0.4):
        """
        Initialize tuner with RTX 4060 constraints
        
        Args:
            max_workers: Number of parallel processes (2 for RTX 4060)
            gpu_memory_fraction: Memory fraction per process (0.4 = ~3.2GB)
        """
        self.max_workers = max_workers
        self.gpu_memory_fraction = gpu_memory_fraction
        self.results = []
        
        # RTX 4060 optimized parameter space
        self.param_space = {
            'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
            'batch_size': [2, 4],  # Small batches for 8GB VRAM
            'train_fraction': [0.01, 0.02, 0.05],
            'optimizer': ['adam', 'adamw'],
            'weight_decay': [1e-5, 1e-4, 1e-3],
            'scheduler': ['cosine', 'step', 'none'],
            'augmentation_prob': [0.3, 0.5, 0.7]
        }
    
    def generate_param_combinations(self, max_combinations=16):
        """Generate parameter combinations for tuning"""
        # Create all combinations
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        all_combinations = list(product(*values))
        
        # Limit combinations for RTX 4060
        if len(all_combinations) > max_combinations:
            import random
            random.seed(42)
            combinations = random.sample(all_combinations, max_combinations)
        else:
            combinations = all_combinations
        
        param_sets = []
        for combo in combinations:
            param_dict = dict(zip(keys, combo))
            param_sets.append(param_dict)
        
        return param_sets
    
    def create_training_script(self, params, gpu_id, temp_dir):
        """Create a standalone training script for each parameter set"""
        script_content = f'''
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
import mlflow
import mlflow.pytorch
import json

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

class CulaneDataset(Dataset):
    def __init__(self, data_root, list_path, transform=None, fraction=1.0):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        full_list_path = os.path.join(data_root, list_path)
        with open(full_list_path, 'r') as f:
            all_lines = f.readlines()
        num_samples_to_use = int(len(all_lines) * fraction)
        lines_to_use = all_lines[:num_samples_to_use]
        for line in lines_to_use:
            parts = line.strip().split()
            self.samples.append({{
                "image": os.path.join(self.data_root, parts[0].lstrip('/')),
                "label": os.path.join(self.data_root, parts[1].lstrip('/'))
            }})
    
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
        return {{"image": image, "label": label}}

def main():
    # Parameters for this run
    params = {json.dumps(params)}
    
    # Configuration
    DATA_ROOT = 'data/CULane'
    LIST_PATH = 'list/train_gt.txt'
    IMG_HEIGHT = 288
    IMG_WIDTH = 800
    NUM_CLASSES = 5
    NUM_EPOCHS = 2  # Short epochs for tuning
    
    # Set memory fraction for RTX 4060
    torch.cuda.set_per_process_memory_fraction({self.gpu_memory_fraction})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure MLflow
    mlflow_dir = os.path.abspath("./mlruns")
    mlflow.set_tracking_uri(f"file://{{mlflow_dir}}")
    mlflow.set_experiment("Hyperparameter Tuning - Lane Detection")
    
    with mlflow.start_run() as run:
        # Log all parameters
        mlflow.log_params(params)
        mlflow.log_param("gpu_memory_fraction", {self.gpu_memory_fraction})
        mlflow.log_param("device", str(device))
        
        # Setup transforms with augmentation probability
        aug_prob = params['augmentation_prob']
        train_transforms = A.Compose([
            A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
            A.HorizontalFlip(p=aug_prob),
            A.RandomBrightnessContrast(p=aug_prob),
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
            num_workers=2,
            pin_memory=True
            drop_last=True
        )
        
        # Load model
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            weights='DeepLabV3_ResNet50_Weights.DEFAULT'
        )
        model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1))
        model.to(device)
        
        # Setup optimizer based on parameter
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:  # adamw
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        
        # Setup scheduler
        if params['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
        elif params['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        else:
            scheduler = None
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training loop
        total_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                try:
                    images = batch['image'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)
                    
                    outputs = model(images)['out']
                    loss = loss_fn(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"OOM error, skipping batch")
                        continue
                    else:
                        raise e
            
            avg_loss = epoch_loss / max(num_batches, 1)
            total_losses.append(avg_loss)
            
            mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
            
            if scheduler:
                scheduler.step()
                mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
        
        # Final metrics
        final_loss = np.mean(total_losses) if total_losses else float('inf')
        mlflow.log_metric("final_avg_loss", final_loss)
        
        # Save result to file for parent process
        result = {{
            "run_id": run.info.run_id,
            "params": params,
            "final_loss": final_loss,
            "gpu_id": {gpu_id}
        }}
        
        with open("{temp_dir}/result.json", "w") as f:
            json.dump(result, f)
        
        print(f"Training complete. Final loss: {{final_loss:.4f}}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
'''
        
        script_path = os.path.join(temp_dir, "train_worker.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def run_single_training(self, params, gpu_id):
        """Run a single training job with given parameters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Create training script
                script_path = self.create_training_script(params, gpu_id, temp_dir)
                
                # Run training in separate process
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=os.getcwd()
                )
                
                if result.returncode != 0:
                    print(f"Training failed for params {params}")
                    print(f"Error: {result.stderr}")
                    return {
                        "params": params,
                        "final_loss": float('inf'),
                        "gpu_id": gpu_id,
                        "error": result.stderr
                    }
                
                # Read result
                result_file = os.path.join(temp_dir, "result.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        return json.load(f)
                else:
                    return {
                        "params": params,
                        "final_loss": float('inf'),
                        "gpu_id": gpu_id,
                        "error": "No result file generated"
                    }
                    
            except Exception as e:
                print(f"Exception in training: {e}")
                return {
                    "params": params,
                    "final_loss": float('inf'),
                    "gpu_id": gpu_id,
                    "error": str(e)
                }
    
    def run_parallel_tuning(self, max_combinations=16):
        """Run parallel hyperparameter tuning"""
        print(f"🔧 Starting parallel hyperparameter tuning for RTX 4060")
        print(f"📊 Max workers: {self.max_workers}")
        print(f"🎯 GPU memory fraction per process: {self.gpu_memory_fraction}")
        
        # Generate parameter combinations
        param_sets = self.generate_param_combinations(max_combinations)
        print(f"🧪 Testing {len(param_sets)} parameter combinations")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return []
        
        gpu_count = torch.cuda.device_count()
        print(f"🔍 Found {gpu_count} GPU(s)")
        
        # Assign GPU IDs (for RTX 4060, typically just GPU 0)
        gpu_assignments = []
        for i, params in enumerate(param_sets):
            gpu_id = i % min(gpu_count, self.max_workers)
            gpu_assignments.append((params, gpu_id))
        
        # Run parallel training
        start_time = time.time()
        results = []
        
        # Use ProcessPoolExecutor for true parallelism with CUDA
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            futures = []
            for params, gpu_id in gpu_assignments:
                future = executor.submit(self.run_single_training, params, gpu_id)
                futures.append((future, params, gpu_id))
            
            # Collect results as they complete
            for i, (future, params, gpu_id) in enumerate(futures):
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)
                    
                    loss = result.get('final_loss', float('inf'))
                    print(f"✅ Completed {i+1}/{len(param_sets)} | GPU {gpu_id} | Loss: {loss:.4f}")
                    
                except Exception as e:
                    print(f"❌ Failed {i+1}/{len(param_sets)} | GPU {gpu_id} | Error: {e}")
                    results.append({
                        "params": params,
                        "final_loss": float('inf'),
                        "gpu_id": gpu_id,
                        "error": str(e)
                    })
        
        total_time = time.time() - start_time
        print(f"⏱️ Total tuning time: {total_time:.1f} seconds")
        
        # Sort results by loss
        valid_results = [r for r in results if r['final_loss'] != float('inf')]
        if valid_results:
            valid_results.sort(key=lambda x: x['final_loss'])
            
            print(f"\n🏆 Top 5 Results:")
            for i, result in enumerate(valid_results[:5]):
                print(f"  {i+1}. Loss: {result['final_loss']:.4f} | Params: {result['params']}")
        else:
            print("❌ No valid results obtained!")
        
        # Save results
        results_file = f"hyperparameter_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to {results_file}")
        
        self.results = results
        return results
    
    def get_best_params(self):
        """Get the best performing parameters"""
        if not self.results:
            print("No results available!")
            return None
        
        valid_results = [r for r in self.results if r['final_loss'] != float('inf')]
        if not valid_results:
            print("No valid results!")
            return None
        
        best_result = min(valid_results, key=lambda x: x['final_loss'])
        return best_result['params']

def main():
    """Main function to run hyperparameter tuning"""
    # Initialize tuner for RTX 4060
    tuner = HyperparameterTuner(
        max_workers=2,  # Conservative for RTX 4060
        gpu_memory_fraction=0.4  # ~3.2GB per process
    )
    
    # Run tuning
    results = tuner.run_parallel_tuning(max_combinations=12)  # Reasonable for testing
    
    # Get best parameters
    best_params = tuner.get_best_params()
    if best_params:
        print(f"\n🎯 Best parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Save best params for later use
        with open("best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print("💾 Best parameters saved to best_hyperparameters.json")

if __name__ == "__main__":
    main()