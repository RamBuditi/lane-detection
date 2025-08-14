from metaflow import FlowSpec, step, Parameter, JSONType
import os
import subprocess
import time
import json
import textwrap 
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

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
            self.tuning_results = []
            self.next(self.train_model)
            return

        print(">>> Running hyperparameter tuning logic.")
        
        try:
            # Import the fixed hyperparameter tuning module
            from hyperparameter_tuning import SimplifiedHyperparameterTuner, TuningConfig
            
            # Create configuration optimized for your setup
            config = TuningConfig(
                max_workers=2,
                gpu_memory_fraction=0.3,  # Conservative for RTX 4060
                max_combinations=self.max_tuning_combinations,
                timeout_minutes=8
            )
            
            # Create tuner and run
            tuner = SimplifiedHyperparameterTuner(config)
            self.tuning_results = tuner.run_parallel_tuning()
            
            # Get best parameters
            self.best_hyperparams = tuner.get_best_params()
            
            if self.best_hyperparams:
                print(f"✅ Best hyperparameters found: {self.best_hyperparams}")
                
                # Save for future use
                with open("best_hyperparameters.json", "w") as f:
                    json.dump(self.best_hyperparams, f, indent=2)
            else:
                print("⚠️ Hyperparameter tuning did not yield a best set of parameters.")
                self.best_hyperparams = None
            
        except Exception as e:
            print(f"❌ Hyperparameter tuning failed: {e}")
            print("📝 Continuing with default parameters...")
            self.best_hyperparams = None
            self.tuning_results = []

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
        """Comprehensive model registry management - PIPELINE HANDLES THIS"""
        print("--- Step: Model Registry Management (Pipeline-Controlled) ---")
        
        try:
            client = MlflowClient()
            
            # Get comprehensive run data
            run = client.get_run(self.run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            # Extract all relevant metrics
            final_train_loss = metrics.get("final_train_loss", float('inf'))
            best_val_miou = metrics.get("best_val_miou", 0.0)
            final_val_miou = metrics.get("final_val_miou", 0.0)
            
            print(f"📊 Training Metrics Retrieved:")
            print(f"   • Final Train Loss: {final_train_loss:.4f}")
            print(f"   • Best Val mIoU: {best_val_miou:.4f}")
            print(f"   • Final Val mIoU: {final_val_miou:.4f}")
            
            # Use best available mIoU for decisions
            decision_miou = max(best_val_miou, final_val_miou)
            
            # Create comprehensive model metadata
            timestamp = int(time.time())
            
            # Determine performance tier
            if decision_miou >= self.promotion_threshold * 1.5:
                tier = "production"
                alias = "production"
                status = "ready-for-deployment"
            elif decision_miou >= self.promotion_threshold:
                tier = "staging"
                alias = "staging"
                status = "ready-for-testing"
            else:
                tier = "development"
                alias = "development" 
                status = "needs-improvement"
            
            # Create descriptive model name
            tuning_indicator = "tuned" if self.best_hyperparams else "baseline"
            unique_model_name = f"culane-detector-{tier}-{tuning_indicator}-{timestamp}"
            
            print(f"🏗️ Creating model: {unique_model_name}")
            print(f"📈 Performance tier: {tier}")
            print(f"🎯 Status: {status}")
            
            # Register model with comprehensive metadata
            model_uri = f"runs:/{self.run_id}/model"
            
            # Create rich description
            description = f"""
            🚗 CULane Detection Model - {tier.title()} Grade

            📊 PERFORMANCE METRICS:
            • Best mIoU: {best_val_miou:.4f}
            • Final mIoU: {final_val_miou:.4f} 
            • Training Loss: {final_train_loss:.4f}
            • Meets Threshold ({self.promotion_threshold:.2f}): {"✅ Yes" if decision_miou >= self.promotion_threshold else "❌ No"}

            🔧 CONFIGURATION:
            • Hyperparameter Tuning: {"✅ Applied" if self.best_hyperparams else "❌ Not Applied"}
            • Optimizer: {params.get('optimizer', 'Unknown')}
            • Learning Rate: {params.get('learning_rate', 'Unknown')}
            • Batch Size: {params.get('batch_size', 'Unknown')}
            • Training Epochs: {params.get('num_epochs', 'Unknown')}

            🏷️ METADATA:
            • Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
            • Pipeline Run: Enhanced MLOps Pipeline
            • MLflow Run: {self.run_id}
            • Model URI: {model_uri}

            🎯 DEPLOYMENT STATUS: {status}
            """.strip()
            
            # Register the model
            model_version = client.create_model_version(
                name=unique_model_name,
                source=model_uri,
                description=description
            )
            
            version_number = model_version.version
            print(f"✅ Registered: {unique_model_name} v{version_number}")
            
            # Wait for model to be ready (sometimes needed)
            print("⏳ Waiting for model version to be ready...")
            time.sleep(3)
            
            # Set alias
            try:
                client.set_registered_model_alias(
                    name=unique_model_name,
                    alias=alias,
                    version=version_number
                )
                print(f"🏷️ Applied alias: {alias}")
            except Exception as alias_error:
                print(f"⚠️ Could not set alias: {alias_error}")
            
            # Add comprehensive tags
            tags = {
                "pipeline_generated": "true",
                "performance_tier": tier,
                "deployment_status": status,
                "miou_score": f"{decision_miou:.4f}",
                "training_loss": f"{final_train_loss:.4f}",
                "hyperparameter_tuned": str(self.best_hyperparams is not None).lower(),
                "promotion_threshold": str(self.promotion_threshold),
                "meets_threshold": str(decision_miou >= self.promotion_threshold).lower(),
                "pipeline_timestamp": str(timestamp),
                "model_architecture": "DeepLabV3_ResNet50",
                "dataset": "CULane"
            }
            
            for key, value in tags.items():
                try:
                    client.set_model_version_tag(
                        name=unique_model_name,
                        version=version_number,
                        key=key,
                        value=value
                    )
                except Exception as tag_error:
                    print(f"⚠️ Could not set tag {key}: {tag_error}")
            
            # Set model version stage for MLflow UI
            try:
                stage_mapping = {
                    "production": "Production",
                    "staging": "Staging", 
                    "development": "None"
                }
                
                if tier in stage_mapping and stage_mapping[tier] != "None":
                    client.transition_model_version_stage(
                        name=unique_model_name,
                        version=version_number,
                        stage=stage_mapping[tier],
                        archive_existing_versions=False
                    )
                    print(f"📋 Set stage: {stage_mapping[tier]}")
                    
            except Exception as stage_error:
                print(f"⚠️ Could not set stage: {stage_error}")
            
            # Store results for pipeline
            self.model_name = unique_model_name
            self.model_version = version_number
            self.model_alias = alias
            self.promotion_status = status
            self.eval_miou = decision_miou
            self.model_uri = f"models:/{unique_model_name}@{alias}"
            
            # Verify registration
            try:
                registered_model = client.get_registered_model(unique_model_name)
                print(f"✅ Verification: Model {unique_model_name} exists with {len(registered_model.latest_versions)} versions")
            except Exception as verify_error:
                print(f"⚠️ Could not verify registration: {verify_error}")
            
            # Print comprehensive summary
            print(f"\n{'='*60}")
            print(f"📦 MODEL REGISTRY SUMMARY")
            print(f"{'='*60}")
            print(f"🏷️ Model Name: {unique_model_name}")
            print(f"📋 Version: {version_number}")
            print(f"🎯 Alias: {alias}")
            print(f"📊 mIoU Score: {decision_miou:.4f}")
            print(f"🚀 Status: {status}")
            print(f"🔗 URI: {self.model_uri}")
            print(f"📁 MLflow Run: {self.run_id}")
            
            if decision_miou >= self.promotion_threshold:
                print(f"🎉 MODEL PROMOTION: SUCCESS!")
                print(f"   Ready for: {'Production Deployment' if tier == 'production' else 'Staging Testing'}")
            else:
                print(f"⚠️ MODEL PROMOTION: NEEDS IMPROVEMENT")
                print(f"   Threshold: {self.promotion_threshold:.3f}, Achieved: {decision_miou:.3f}")
                print(f"   Recommendations: Increase training epochs, tune hyperparameters, use more data")
            
        except Exception as e:
            print(f"❌ Model registry management failed: {e}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            
            # Set failure state
            self.model_name = None
            self.model_version = None
            self.model_alias = None
            self.promotion_status = "failed"
            self.eval_miou = 0.0
            self.model_uri = None
        
        self.next(self.generate_model_report)

    @step
    def generate_model_report(self):
        """Enhanced model report with registry information"""
        print("--- Step: Comprehensive Model & Registry Report ---")
        
        # Generate comprehensive report
        pipeline_duration = time.time() - self.pipeline_start_time
        
        # Create the most comprehensive report
        self.comprehensive_report = {
            "pipeline_execution": {
                "pipeline_id": f"enhanced-lane-detection-{int(self.pipeline_start_time)}",
                "start_time": self.pipeline_start_time,
                "duration_seconds": pipeline_duration,
                "duration_formatted": f"{pipeline_duration/60:.1f} minutes",
                "success": self.promotion_status != "failed"
            },
            
            "hyperparameter_tuning": {
                "enabled": self.run_hyperparameter_tuning,
                "max_combinations_tested": self.max_tuning_combinations if self.run_hyperparameter_tuning else 0,
                "best_params_found": self.best_hyperparams is not None,
                "tuning_successful": self.best_hyperparams is not None if self.run_hyperparameter_tuning else None,
                "best_hyperparameters": self.best_hyperparams
            },
            
            "training_results": {
                "mlflow_run_id": self.run_id,
                "training_completed": True,
                "hyperparameter_source": "tuned" if self.best_hyperparams else "default"
            },
            
            "model_performance": {
                "miou_score": self.eval_miou,
                "promotion_threshold": self.promotion_threshold,
                "meets_threshold": self.eval_miou >= self.promotion_threshold,
                "performance_category": (
                    "excellent" if self.eval_miou >= self.promotion_threshold * 1.5
                    else "good" if self.eval_miou >= self.promotion_threshold
                    else "needs_improvement"
                )
            },
            
            "model_registry": {
                "registration_successful": self.model_name is not None,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "model_alias": self.model_alias,
                "promotion_status": self.promotion_status,
                "model_uri": self.model_uri,
                "deployment_ready": self.promotion_status in ["ready-for-deployment", "ready-for-testing"]
            },
            
            "system_info": {
                "gpu_optimizations_applied": getattr(self, 'gpu_optimizations', False),
                "pipeline_parameters": {
                    "run_hyperparameter_tuning": self.run_hyperparameter_tuning,
                    "max_tuning_combinations": self.max_tuning_combinations,
                    "promotion_threshold": self.promotion_threshold,
                    "use_best_params": self.use_best_params
                }
            },
            
            "next_steps": self._generate_next_steps(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save comprehensive report
        report_file = f"pipeline_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(self.comprehensive_report, f, indent=2, default=str)
        
        print(f"📄 Comprehensive report saved: {report_file}")
        
        # Also create a human-readable summary
        self._create_human_readable_summary()
        
        self.next(self.end)
    
    def _generate_next_steps(self):
        """Generate specific next steps based on results"""
        steps = []
        
        if self.model_uri:
            steps.append(f"Test model inference: python src/inference.py --model-uri {self.model_uri}")
            
        if self.promotion_status == "ready-for-deployment":
            steps.extend([
                "Deploy model to production environment",
                "Set up model monitoring and alerting",
                "Configure automated retraining pipeline"
            ])
        elif self.promotion_status == "ready-for-testing":
            steps.extend([
                "Deploy model to staging environment",
                "Run comprehensive testing suite",
                "Perform A/B testing against current production model"
            ])
        else:
            steps.extend([
                "Improve model performance through additional training",
                "Run hyperparameter tuning if not already done",
                "Consider data augmentation or additional training data"
            ])
            
        steps.append(f"View detailed results: http://localhost:5000")
        steps.append("Monitor training runs in MLflow UI")
        
        return steps
    
    def _create_human_readable_summary(self):
        """Create a human-readable markdown summary"""
        summary_content = f"""
        # Lane Detection Pipeline Report

        ## 🎯 Executive Summary
        - **Pipeline Status**: {"✅ SUCCESS" if self.promotion_status != "failed" else "❌ FAILED"}
        - **Model Performance**: {self.eval_miou:.4f} mIoU {"(Meets Threshold ✅)" if self.eval_miou >= self.promotion_threshold else "(Below Threshold ⚠️)"}
        - **Deployment Status**: {self.promotion_status}

        ## 📊 Performance Details
        - **mIoU Score**: {self.eval_miou:.4f}
        - **Threshold**: {self.promotion_threshold:.3f}
        - **Performance Tier**: {self.comprehensive_report['model_performance']['performance_category']}

        ## 🔧 Configuration
        - **Hyperparameter Tuning**: {"✅ Enabled" if self.run_hyperparameter_tuning else "❌ Disabled"}
        - **Best Parameters Found**: {"✅ Yes" if self.best_hyperparams else "❌ No"}

        ## 📦 Model Registry
        - **Model Name**: `{self.model_name or 'Not registered'}`
        - **Version**: {self.model_version or 'N/A'}
        - **Alias**: `{self.model_alias or 'N/A'}`
        - **URI**: `{self.model_uri or 'Not available'}`

        ## 🚀 Next Steps
        """
        
        for i, step in enumerate(self.comprehensive_report['next_steps'], 1):
            summary_content += f"{i}. {step}\n"
        
        summary_file = f"pipeline_summary_{int(time.time())}.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"📋 Human-readable summary: {summary_file}")

    
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
        if self.promotion_status != "failed":
            print(f"   • URI: models:/culane-lane-detector@{self.promotion_status}")
        
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
        if self.promotion_status != "failed":
            print(f"   2. Test inference: python src/inference.py --model-uri models:/culane-lane-detector@{self.promotion_status}")
        
        if self.promotion_status in ["staging", "production"]:
            print(f"   3. Deploy model for inference")
            print(f"   4. Set up monitoring and alerts")
        else:
            print(f"   3. Improve model performance")
            print(f"   4. Run with hyperparameter tuning: --tune-hyperparams")

if __name__ == '__main__':
    EnhancedLaneDetectionPipeline()