from metaflow import FlowSpec, step, Parameter, JSONType
import os
import subprocess
import time
import json
import textwrap 
import mlflow
from mlflow.tracking import MlflowClient

class EnhancedLaneDetectionPipeline(FlowSpec):
    """
    Enhanced MLOps pipeline with hyperparameter tuning, model registry,
    and comprehensive evaluation for lane detection.
    """
    
    # Pipeline parameters
    run_hyperparameter_tuning = Parameter(
        'tune-hyperparams',
        help='Whether to run hyperparameter tuning',
        default=False,
        type=bool
    )
    
    max_tuning_combinations = Parameter(
        'max-combinations',
        help='Maximum hyperparameter combinations to test',
        default=8,
        type=int
    )
    
    promotion_threshold = Parameter(
        'promotion-threshold',
        help='mIoU threshold for model promotion',
        default=0.30,
        type=float
    )
    
    use_best_params = Parameter(
        'use-best-params',
        help='Use best hyperparameters from previous tuning',
        default=True,
        type=bool
    )
    
    @step
    def start(self):
        """Initialize the enhanced pipeline"""
        print("🚀 Enhanced Lane Detection MLOps Pipeline Starting...")
        print(f"📊 Hyperparameter tuning: {'Enabled' if self.run_hyperparameter_tuning else 'Disabled'}")
        print(f"🎯 Promotion threshold: mIoU > {self.promotion_threshold}")
        
        # Initialize pipeline state
        self.pipeline_start_time = time.time()
        self.best_hyperparams = None
        self.tuning_results = None
        
        self.next(self.validate_environment)
    
    @step
    def validate_environment(self):
        """Enhanced environment validation"""
        print("--- Step: Enhanced Environment Validation ---")
        
        # Check GPU availability and memory
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ GPU Available: {gpu_name}")
            print(f"📱 GPU Count: {gpu_count}")
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            if "4060" in gpu_name:
                print("🔧 RTX 4060 detected - enabling memory optimizations")
                self.gpu_optimizations = True
            else:
                self.gpu_optimizations = False
        else:
            print("❌ No GPU available - using CPU")
            self.gpu_optimizations = False
        
        # Validate DVC data
        try:
            result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ DVC data is synchronized")
            else:
                print("⚠️ Syncing DVC data...")
                subprocess.run(["dvc", "pull"], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️ DVC not available or data not tracked")
        
        # Check dataset
        if os.path.exists("data/CULane"):
            print("✅ CULane dataset found")
        else:
            raise Exception("❌ CULane dataset not found")
        
        # Ensure MLflow is ready
        os.makedirs("mlruns", exist_ok=True)
        mlflow_uri = f"file://{os.path.abspath('./mlruns')}"
        mlflow.set_tracking_uri(mlflow_uri)
        print(f"✅ MLflow ready at {mlflow_uri}")
        
        # Check for existing hyperparameter results
        if self.use_best_params and os.path.exists("best_hyperparameters.json"):
            with open("best_hyperparameters.json", 'r') as f:
                self.best_hyperparams = json.load(f)
            print("✅ Found existing best hyperparameters")

        # This is the fix: Always go to the hyperparameter_tuning step next.
        # The graph is now linear and simple for the validator to parse.
        print(">>> Unconditionally transitioning to 'hyperparameter_tuning' step.")
        self.next(self.hyperparameter_tuning)

        

    @step
    def hyperparameter_tuning(self):
        """Run parallel hyperparameter tuning (if enabled)"""
        print("--- Step: Hyperparameter Tuning ---")
        
        if not self.run_hyperparameter_tuning:
            print(">>> Skipping hyperparameter tuning as per parameters.")
            self.best_hyperparams = None
            self.next(self.train_model)
            return

        print(">>> Running hyperparameter tuning logic.")
        
        # FIX: Use textwrap.dedent to remove leading whitespace automatically.
        # This robustly fixes the IndentationError.
        tuning_script = textwrap.dedent(f"""
            from src.hyperparameter_tuning import HyperparameterTuner
            tuner = HyperparameterTuner(max_workers=2, gpu_memory_fraction=0.4)
            results = tuner.run_parallel_tuning(max_combinations={self.max_tuning_combinations})
            best_params = tuner.get_best_params()
            print(f"BEST_PARAMS_JSON: {{repr(best_params)}}")
        """)

        cmd = ["python", "-c", tuning_script]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print("❌ Hyperparameter tuning failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            self.best_hyperparams = None # Continue without best params
        else:
            print(result.stdout)
            for line in result.stdout.splitlines():
                if line.startswith("BEST_PARAMS_JSON:"):
                    params_str = line.split("BEST_PARAMS_JSON:", 1)[1].strip()
                    try:
                        self.best_hyperparams = eval(params_str)
                        print(f"✅ Best hyperparameters extracted: {self.best_hyperparams}")
                    except Exception as e:
                        print(f"⚠️ Could not parse best parameters: {e}")
                        self.best_hyperparams = None
                    break
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """Enhanced model training with best hyperparameters"""
        print("--- Step: Enhanced Model Training ---")
        
        # Prepare training command
        cmd = ["python", "src/train.py"]
        
        # Add hyperparameter config if available
        config_file = None
        if self.best_hyperparams:
            config_file = "pipeline_hyperparams.json"
            # Add more epochs for final training
            final_params = self.best_hyperparams.copy()
            final_params['num_epochs'] = 3  # More epochs for final model
            final_params['train_fraction'] = 0.1  # More data for final model
            
            with open(config_file, 'w') as f:
                json.dump(final_params, f, indent=2)
            
            cmd.extend(["--config", config_file])
            print(f"🎯 Using tuned hyperparameters: {final_params}")
        
        # Add experiment naming
        cmd.extend([
            "--experiment-name", "Enhanced Lane Detection Pipeline",
            "--run-name", f"Pipeline-{int(time.time())}"
        ])
        
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up config file
        if config_file and os.path.exists(config_file):
            os.remove(config_file)
        
        if result.returncode != 0:
            print("❌ Enhanced training failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            raise Exception("Training failed")
        
        print(result.stdout)
        
        # Extract run ID
        output_lines = result.stdout.splitlines()
        run_id_line = next((line for line in output_lines if "MLflow Run ID:" in line), None)
        
        if run_id_line is None:
            raise Exception("Could not find MLflow Run ID in training output")
        
        self.run_id = run_id_line.split(":")[1].strip()
        print(f"✅ Enhanced training complete. Run ID: {self.run_id}")
        
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        """Enhanced model evaluation"""
        print("--- Step: Enhanced Model Evaluation ---")
        
        # Run evaluation
        result = subprocess.run(
            ["python", "src/evaluate.py", "--run_id", self.run_id],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Evaluation failed!")
            print("STDERR:", result.stderr)
            raise Exception("Evaluation failed")
        
        print(result.stdout)
        
        # Extract mIoU
        output_lines = result.stdout.splitlines()
        miou_line = next((line for line in output_lines if "Overall Mean IoU (mIoU):" in line), None)
        
        if miou_line:
            self.eval_miou = float(miou_line.split(":")[-1].strip())
            print(f"📊 mIoU: {self.eval_miou:.4f}")
        else:
            print("⚠️ Could not extract mIoU")
            self.eval_miou = 0.0
        
        self.next(self.model_registry_management)
    
    @step
    def model_registry_management(self):
        """Enhanced model registry management with aliases"""
        print("--- Step: Model Registry Management ---")
        
        try:
            # Initialize MLflow client
            client = MlflowClient()
            
            # Check if model is already registered
            model_name = "culane-lane-detector"
            
            try:
                # Get the latest version
                latest_versions = client.get_latest_versions(model_name)
                if latest_versions:
                    latest_version = max([int(v.version) for v in latest_versions])
                    print(f"📦 Found existing model versions up to v{latest_version}")
                else:
                    print("📦 No existing model versions found")
            except Exception as e:
                print(f"📦 Model registry is empty or model not found: {e}")
            
            # Get the model version that was auto-registered during training
            model_uri = f"runs:/{self.run_id}/model"
            
            # Register or update model version
            try:
                mv = mlflow.register_model(model_uri, model_name)
                model_version = mv.version
                print(f"✅ Model registered as version {model_version}")
                
                # Set model version description
                description = f"Lane detection model trained with {'tuned' if self.best_hyperparams else 'default'} hyperparameters. mIoU: {self.eval_miou:.4f}"
                client.update_model_version(
                    name=model_name,
                    version=model_version,
                    description=description
                )
                
                # Manage aliases based on performance
                if self.eval_miou > self.promotion_threshold:
                    # Promote to staging
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="staging",
                        version=model_version
                    )
                    print(f"🚀 Model promoted to 'staging' alias")
                    
                    # If performance is really good, consider for production
                    if self.eval_miou > self.promotion_threshold * 1.5:
                        client.set_registered_model_alias(
                            name=model_name,
                            alias="production",
                            version=model_version
                        )
                        print(f"🌟 Model promoted to 'production' alias")
                        self.promotion_status = "production"
                    else:
                        self.promotion_status = "staging"
                else:
                    # Set as development/testing
                    client.set_registered_model_alias(
                        name=model_name,
                        alias="development",
                        version=model_version
                    )
                    print(f"🔧 Model tagged as 'development' alias")
                    self.promotion_status = "development"
                
                self.model_version = model_version
                
            except Exception as e:
                print(f"❌ Error with model registry: {e}")
                self.promotion_status = "failed"
                self.model_version = None
        
        except Exception as e:
            print(f"❌ Model registry management failed: {e}")
            self.promotion_status = "failed"
            self.model_version = None
        
        self.next(self.generate_model_report)
    
    @step
    def generate_model_report(self):
        """Generate comprehensive model and pipeline report"""
        print("--- Step: Generating Comprehensive Report ---")
        
        try:
            # Generate Evidently AI report for model monitoring
            report_cmd = [
                "python", "-c",
                f"""
import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import *
import pandas as pd
import json

# Create a simple performance report
performance_data = {{
    'run_id': '{self.run_id}',
    'mIoU': {self.eval_miou},
    'promotion_threshold': {self.promotion_threshold},
    'meets_threshold': {self.eval_miou > self.promotion_threshold},
    'hyperparams_tuned': {self.best_hyperparams is not None}
}}

# Save performance data
with open('model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=2)

print("Model performance report generated")
"""
            ]
            
            result = subprocess.run(report_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Evidently report generated")
            else:
                print("⚠️ Could not generate Evidently report, creating basic report")
        
        except Exception as e:
            print(f"⚠️ Report generation had issues: {e}")
        
        # Create comprehensive pipeline report
        pipeline_duration = time.time() - self.pipeline_start_time
        
        self.comprehensive_report = {
            "pipeline_info": {
                "pipeline_id": "enhanced-lane-detection",
                "start_time": self.pipeline_start_time,
                "duration_seconds": pipeline_duration,
                "duration_formatted": f"{pipeline_duration/60:.1f} minutes"
            },
            "hyperparameter_tuning": {
                "enabled": self.run_hyperparameter_tuning,
                "max_combinations_tested": self.max_tuning_combinations if self.run_hyperparameter_tuning else 0,
                "best_params_found": self.best_hyperparams is not None,
                "best_hyperparameters": self.best_hyperparams
            },
            "model_performance": {
                "mlflow_run_id": self.run_id,
                "mIoU": self.eval_miou,
                "promotion_threshold": self.promotion_threshold,
                "meets_threshold": self.eval_miou > self.promotion_threshold,
                "performance_category": "excellent" if self.eval_miou > self.promotion_threshold * 1.5 
                                        else "good" if self.eval_miou > self.promotion_threshold 
                                        else "needs_improvement"
            },
            "model_registry": {
                "promotion_status": self.promotion_status,
                "model_version": self.model_version,
                "registry_name": "culane-lane-detector"
            },
            "system_info": {
                "gpu_optimizations": getattr(self, 'gpu_optimizations', False),
                "pipeline_parameters": {
                    "run_hyperparameter_tuning": self.run_hyperparameter_tuning,
                    "max_tuning_combinations": self.max_tuning_combinations,
                    "promotion_threshold": self.promotion_threshold,
                    "use_best_params": self.use_best_params
                }
            },
            "recommendations": self._generate_recommendations()
        }
        
        # Save comprehensive report
        report_file = f"comprehensive_pipeline_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.comprehensive_report, f, indent=2)
        
        print(f"📄 Comprehensive report saved: {report_file}")
        
        self.next(self.end)
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        if self.eval_miou <= self.promotion_threshold:
            recommendations.append({
                "priority": "high",
                "category": "performance",
                "message": f"Model performance ({self.eval_miou:.4f}) below threshold ({self.promotion_threshold})",
                "actions": [
                    "Increase training epochs",
                    "Use larger train_fraction",
                    "Try different augmentation strategies",
                    "Consider ensemble methods"
                ]
            })
        
        if not self.run_hyperparameter_tuning and not self.best_hyperparams:
            recommendations.append({
                "priority": "medium",
                "category": "optimization",
                "message": "No hyperparameter tuning performed",
                "actions": [
                    "Run hyperparameter tuning with --tune-hyperparams",
                    "Experiment with different optimizers",
                    "Try learning rate scheduling"
                ]
            })
        
        if self.promotion_status in ["production", "staging"]:
            recommendations.append({
                "priority": "low",
                "category": "deployment",
                "message": "Model ready for deployment",
                "actions": [
                    "Set up model monitoring",
                    "Implement A/B testing",
                    "Configure automated retraining"
                ]
            })
        
        return recommendations
    
    @step
    def end(self):
        """Enhanced pipeline completion"""
        print("--- Enhanced Pipeline Summary ---")
        print("🎉 Enhanced Pipeline Completed Successfully!")
        print("=" * 60)
        
        # Performance summary
        print(f"📊 Model Performance:")
        print(f"   • mIoU: {self.eval_miou:.4f}")
        print(f"   • Threshold: {self.promotion_threshold}")
        print(f"   • Status: {'✅ PASSED' if self.eval_miou > self.promotion_threshold else '❌ NEEDS IMPROVEMENT'}")
        
        # Hyperparameter tuning summary
        print(f"\n🔧 Hyperparameter Tuning:")
        if self.run_hyperparameter_tuning:
            print(f"   • Combinations tested: {self.max_tuning_combinations}")
            print(f"   • Best params found: {'✅ Yes' if self.best_hyperparams else '❌ No'}")
        else:
            print(f"   • Status: Skipped")
        
        # Model registry summary
        print(f"\n📦 Model Registry:")
        print(f"   • Version: {self.model_version}")
        print(f"   • Alias: {self.promotion_status}")
        print(f"   • Registry: culane-lane-detector")
        
        # Pipeline info
        duration = time.time() - self.pipeline_start_time
        print(f"\n⏱️ Pipeline Stats:")
        print(f"   • Duration: {duration/60:.1f} minutes")
        print(f"   • MLflow Run: {self.run_id}")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   • {rec['message']}")
        
        print(f"\n🔗 Next Steps:")
        print(f"   1. View results: http://localhost:5000")
        print(f"   2. Test inference: python src/inference.py --alias {self.promotion_status}")
        
        if self.promotion_status in ["staging", "production"]:
            print(f"   3. Deploy model for inference")
            print(f"   4. Set up monitoring and alerts")
        else:
            print(f"   3. Improve model performance")
            print(f"   4. Run with hyperparameter tuning: --tune-hyperparams")

if __name__ == '__main__':
    EnhancedLaneDetectionPipeline()