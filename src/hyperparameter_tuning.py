import os
import torch
import mlflow
import mlflow.pytorch
from concurrent.futures import ProcessPoolExecutor
import json
import time
from itertools import product
import subprocess
import tempfile
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import multiprocessing as mp
import random
import threading
import queue

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning"""
    max_workers: int = 2
    gpu_memory_fraction: float = 0.3
    max_combinations: int = 8
    timeout_minutes: int = 20
    experiment_name: str = "Hyperparameter_Tuning_Lane_Detection"
    verbose_output: bool = False  # Control output verbosity
    
    param_space: Dict[str, List[Any]] = None
    
    def __post_init__(self):
        if self.param_space is None:
            # Simplified parameter space for quick tuning
            self.param_space = {
                'learning_rate': [1e-4, 5e-5, 2e-5],
                'batch_size': [2, 4],
                'train_fraction': [0.01, 0.02],
                'optimizer': ['adam', 'adamw'],
                'weight_decay': [1e-4, 1e-5],
                'scheduler': ['cosine', 'step'],
                'num_epochs': [1],
                'dropout_rate': [0.1, 0.2],
                'val_fraction': [0.02, 0.03],
                'augmentation_prob': [0.3, 0.5]
            }

class SimplifiedHyperparameterTuner:
    """
    Simplified hyperparameter tuner with controlled output
    """
    
    def __init__(self, config: TuningConfig = None):
        self.config = config or TuningConfig()
        self.results = []
        
        os.makedirs("mlruns", exist_ok=True)
        mlflow_uri = f"file://{os.path.abspath('./mlruns')}"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)
        
    def generate_param_combinations(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations using smart sampling"""
        keys = list(self.config.param_space.keys())
        values = list(self.config.param_space.values())
        all_combinations = list(product(*values))
        
        if len(all_combinations) <= self.config.max_combinations:
            combinations = all_combinations
        else:
            random.seed(42)
            combinations = random.sample(all_combinations, self.config.max_combinations)
        
        param_sets = [dict(zip(keys, combo)) for combo in combinations]
        return param_sets

    def _monitor_process_output(self, process, output_queue, run_info):
        """Monitor process output in a separate thread with controlled verbosity"""
        run_id = None
        important_lines = []
        last_progress_time = 0
        
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    line = output.strip()
                    
                    # Always capture MLflow Run ID
                    if "MLflow Run ID:" in line:
                        run_id = line.split(":")[-1].strip()
                        important_lines.append(f"✅ Run ID captured: {run_id}")
                    
                    # Capture important status messages
                    elif any(keyword in line.lower() for keyword in [
                        "starting", "complete", "error", "failed", "✅", "❌", "⚠️"
                    ]):
                        important_lines.append(line)
                    
                    # Handle progress bars with throttling
                    elif "%" in line and ("Epoch" in line or "Train" in line or "Validation" in line):
                        current_time = time.time()
                        if current_time - last_progress_time > 10:  # Only show progress every 10 seconds
                            important_lines.append(f"📊 {line}")
                            last_progress_time = current_time
                    
                    # Show all output if verbose mode is enabled
                    elif self.config.verbose_output:
                        important_lines.append(line)
        
        except Exception as e:
            important_lines.append(f"❌ Output monitoring error: {e}")
        
        finally:
            output_queue.put((run_id, important_lines))

    def run_single_training(self, params_and_gpu):
        """Run single training job with controlled output"""
        params, gpu_id = params_and_gpu
        run_id = None
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(params, f, indent=2)
                config_file = f.name
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
            cmd = [
                "python", "src/train.py",
                "--config", config_file,
                "--experiment-name", self.config.experiment_name,
                "--run-name", f"HyperTune_run_{int(time.time())}_{gpu_id}",
                "--no-validation",
                "--quiet"  # Add quiet flag to reduce output noise
            ]
            
            param_summary = {k: v for k, v in params.items() if k in ['learning_rate', 'batch_size', 'optimizer']}
            print(f"\n🧪 Starting training on GPU {gpu_id}")
            print(f"   Key params: {param_summary}")
            
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=os.getcwd()
            )
            
            # Monitor output in separate thread
            output_queue = queue.Queue()
            monitor_thread = threading.Thread(
                target=self._monitor_process_output, 
                args=(process, output_queue, params)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Wait for process with timeout
            try:
                returncode = process.wait(timeout=self.config.timeout_minutes * 60)
                monitor_thread.join(timeout=5)  # Give monitor thread time to finish
                
                # Get output results
                try:
                    run_id, important_lines = output_queue.get_nowait()
                    
                    # Print condensed summary
                    print(f"📋 Training summary for GPU {gpu_id}:")
                    for line in important_lines[-5:]:  # Show last 5 important lines
                        print(f"   {line}")
                        
                except queue.Empty:
                    print(f"⚠️ No output captured from training on GPU {gpu_id}")
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ Training on GPU {gpu_id} timed out after {self.config.timeout_minutes} minutes")
                process.kill()
                process.wait()
                return {
                    "params": params, "run_id": run_id, "gpu_id": gpu_id, 
                    "success": False, "error": "Timeout"
                }
            
            finally:
                if os.path.exists(config_file):
                    os.unlink(config_file)
            
            if returncode == 0 and run_id:
                print(f"✅ Training on GPU {gpu_id} successful. Run ID: {run_id}")
                metrics = self._get_mlflow_metrics(run_id)
                return {
                    "params": params, "run_id": run_id, "gpu_id": gpu_id, "success": True,
                    "final_train_loss": metrics.get("final_train_loss", float('inf')),
                    "best_val_miou": metrics.get("best_val_miou", 0.0),
                }
            else:
                print(f"❌ Training on GPU {gpu_id} failed with return code {returncode}")
                return {
                    "params": params, "run_id": run_id, "gpu_id": gpu_id, 
                    "success": False, "error": f"Return code: {returncode}"
                }

        except Exception as e:
            print(f"💥 Exception during training on GPU {gpu_id}: {e}")
            return {
                "params": params, "run_id": run_id, "gpu_id": gpu_id, 
                "success": False, "error": str(e)
            }
    
    def _get_mlflow_metrics(self, run_id: str) -> Dict[str, float]:
        """Get metrics from a specific MLflow run"""
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id)
            metrics = run.data.metrics
            return {
                "final_train_loss": metrics.get("final_train_loss", float('inf')),
                "best_val_miou": metrics.get("best_val_miou", 0.0),
                "final_val_miou": metrics.get("final_val_miou", 0.0)
            }
        except Exception as e:
            print(f"⚠️ Could not retrieve MLflow metrics for run {run_id}: {e}")
            return {}
    
    def run_parallel_tuning(self) -> List[Dict]:
        """Run parallel hyperparameter tuning with controlled output"""
        print(f"\n🔧 Starting hyperparameter tuning")
        print(f"📁 Experiment: {self.config.experiment_name}")
        print(f"🔇 Verbose output: {'Enabled' if self.config.verbose_output else 'Disabled'}")
        
        param_sets = self.generate_param_combinations()
        print(f"🧪 Testing {len(param_sets)} parameter combinations")
        print(f"⏱️ Timeout per run: {self.config.timeout_minutes} minutes")
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available!")
            return []
        
        gpu_count = torch.cuda.device_count()
        print(f"🎮 Found {gpu_count} GPU(s)")
        
        param_gpu_pairs = [(params, i % gpu_count) for i, params in enumerate(param_sets)]
        
        start_time = time.time()
        results = []
        
        print(f"\n{'='*60}")
        print("🚀 HYPERPARAMETER TUNING EXECUTION")
        print(f"{'='*60}")
        
        # Run jobs based on max_workers setting
        if self.config.max_workers > 1 and len(param_gpu_pairs) > 1:
            print(f"🔄 Running {len(param_gpu_pairs)} jobs in parallel (max_workers={self.config.max_workers})")
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(self.run_single_training, param_gpu_pairs))
        else:
            print("🔄 Running tuning jobs sequentially...")
            for i, pair in enumerate(param_gpu_pairs):
                print(f"\n--- Job {i+1}/{len(param_gpu_pairs)} ---")
                results.append(self.run_single_training(pair))

        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("📊 HYPERPARAMETER TUNING RESULTS")
        print(f"{'='*60}")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        print(f"📈 Jobs completed: {len(results)}")
        
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(failed_results)}")
        
        if failed_results:
            print(f"\n⚠️ Failed job summary:")
            for i, result in enumerate(failed_results):
                error = result.get("error", "Unknown error")
                gpu_id = result.get("gpu_id", "?")
                print(f"   {i+1}. GPU {gpu_id}: {error}")
        
        if successful_results:
            # Sort by training loss (lower is better)
            successful_results.sort(key=lambda x: x.get("final_train_loss", float('inf')))
            
            print(f"\n🏆 Top Results (sorted by training loss):")
            for i, result in enumerate(successful_results[:5]):
                loss = result.get("final_train_loss", float('inf'))
                run_id = result.get("run_id", "N/A")
                params = result.get("params", {})
                key_params = {k: v for k, v in params.items() if k in ['learning_rate', 'batch_size', 'optimizer']}
                
                print(f"\n   🥇 Rank {i+1}:")
                print(f"      Loss: {loss:.4f}")
                print(f"      Run ID: {run_id}")
                print(f"      Key params: {key_params}")
        else:
            print("\n❌ No successful training runs found!")
            print("💡 Suggestions:")
            print("   - Check GPU memory settings")
            print("   - Verify training data is accessible")
            print("   - Enable verbose output: config.verbose_output = True")
        
        # Save results
        results_file = f"hyperparameter_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 All results saved to: {results_file}")
        
        self.results = results
        return results
    
    def get_best_params(self) -> Optional[Dict]:
        """Get best parameters based on training loss"""
        if not self.results: 
            return None
            
        successful_results = [
            r for r in self.results 
            if r.get("success") and r.get("final_train_loss") != float('inf')
        ]
        
        if not successful_results: 
            return None
        
        best_result = min(successful_results, key=lambda x: x.get("final_train_loss", float('inf')))
        return best_result["params"]

def main():
    """Main function for standalone testing"""
    print("🚀 Starting Hyperparameter Tuning")
    
    config = TuningConfig(
        max_workers=2,
        max_combinations=4,
        timeout_minutes=15,
        verbose_output=False  # Set to True for debugging
    )
    
    tuner = SimplifiedHyperparameterTuner(config)
    results = tuner.run_parallel_tuning()
    
    best_params = tuner.get_best_params()
    if best_params:
        print(f"\n🎯 BEST PARAMETERS FOUND:")
        print(json.dumps(best_params, indent=2))
        
        with open("best_hyperparameters.json", "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n💾 Best parameters saved to: best_hyperparameters.json")
    else:
        print("\n⚠️ No best parameters found.")
        print("💡 Check the failed job errors above for debugging.")

if __name__ == "__main__":
    main()