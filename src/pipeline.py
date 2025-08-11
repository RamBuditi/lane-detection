from metaflow import FlowSpec, step
import os
import subprocess
import time
import json

class LaneDetectionPipeline(FlowSpec):
    """
    A Metaflow pipeline that trains and evaluates a lane detection model
    with automatic model promotion based on performance thresholds.
    """
    
    # Performance thresholds for model promotion
    MIOU_THRESHOLD = 0.30  # Promote if mIoU > 30%
    
    @step
    def start(self):
        """
        The starting point of the pipeline.
        """
        print("🚀 Lane Detection MLOps Pipeline Starting...")
        print(f"📊 Model promotion threshold: mIoU > {self.MIOU_THRESHOLD}")
        self.next(self.validate_environment)
    
    @step
    def validate_environment(self):
        """
        Validate that all required components are available.
        """
        print("--- Step: Environment Validation ---")
        
        # Check if DVC is available and data is synchronized
        try:
            result = subprocess.run(["dvc", "status"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ DVC data is synchronized")
            else:
                print("⚠️ DVC data may need synchronization")
                print("Running DVC pull to sync data...")
                subprocess.run(["dvc", "pull"], check=True)
                print("✅ DVC data synchronized")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"❌ DVC error: {e}")
            print("Continuing without DVC validation...")
        
        # Check if training data exists
        if os.path.exists("data/CULane"):
            print("✅ CULane dataset found")
        else:
            print("❌ CULane dataset not found")
            raise Exception("Dataset not available")
        
        # Check if MLflow directory exists
        os.makedirs("mlruns", exist_ok=True)
        print("✅ MLflow directory ready")
        
        self.next(self.train_model)
    
    @step
    def train_model(self):
        """
        This step runs the training script as a separate process.
        """
        print("--- Step: Training Model ---")
       
        # Run the training script
        result = subprocess.run(
            ["python", "src/train.py"],
            capture_output=True,
            text=True
        )
       
        # Check if the training script ran successfully
        if result.returncode != 0:
            print("❌ Training script failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            raise Exception("Training failed")
       
        print(result.stdout)
       
        # Extract the Run ID from the training script's output
        output_lines = result.stdout.splitlines()
        run_id_line = next((line for line in output_lines if "MLflow Run ID:" in line), None)
       
        if run_id_line is None:
            raise Exception("Could not find MLflow Run ID in the training output.")
           
        self.run_id = run_id_line.split(":")[1].strip()
        print(f"✅ Training complete. Found Run ID: {self.run_id}")
        self.next(self.evaluate_model)
    
    @step
    def evaluate_model(self):
        """
        This step uses the run_id from the training step to run evaluation.
        """
        print("--- Step: Evaluating Model ---")
       
        result = subprocess.run(
            ["python", "src/evaluate.py", "--run_id", self.run_id],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Evaluation script failed!")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout)
            raise Exception("Evaluation failed")
       
        print(result.stdout)
        
        # Extract mIoU from evaluation output
        output_lines = result.stdout.splitlines()
        miou_line = next((line for line in output_lines if "Overall Mean IoU (mIoU):" in line), None)
        
        if miou_line:
            # Parse mIoU value (format: "Overall Mean IoU (mIoU): 0.2176")
            self.eval_miou = float(miou_line.split(":")[-1].strip())
            print(f"📊 Extracted mIoU: {self.eval_miou:.4f}")
        else:
            print("⚠️ Could not extract mIoU from evaluation output")
            self.eval_miou = 0.0
        
        print(f"✅ Evaluation complete for Run ID: {self.run_id}")
        self.next(self.promote_model)
    
    @step
    def promote_model(self):
        """
        Decide whether to promote the model to DVC based on performance thresholds.
        """
        print("--- Step: Model Promotion Decision ---")
        
        if self.eval_miou > self.MIOU_THRESHOLD:
            print(f"🎉 Model qualifies for promotion! mIoU: {self.eval_miou:.4f} > {self.MIOU_THRESHOLD}")
            
            # Create champion model with timestamp
            timestamp = int(time.time())
            champion_model_name = f"champion_model_{self.run_id}_{timestamp}.pth"
            champion_model_path = f"models/{champion_model_name}"
            
            try:
                # Ensure models directory exists
                os.makedirs("models", exist_ok=True)
                
                # Copy the latest model to champion model
                if os.path.exists("models/latest_model_run.pth"):
                    subprocess.run([
                        "cp", "models/latest_model_run.pth", champion_model_path
                    ], check=True)
                    
                    # Add to DVC
                    result = subprocess.run(["dvc", "add", champion_model_path], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"✅ Model added to DVC: {champion_model_path}")
                        
                        # Add DVC file to git (if git is initialized)
                        try:
                            subprocess.run(["git", "add", f"{champion_model_path}.dvc"], check=True)
                            subprocess.run(["git", "add", "models/.gitignore"], check=True)
                            
                            # Commit the changes
                            commit_msg = f"Add champion model {self.run_id} (mIoU: {self.eval_miou:.4f})"
                            subprocess.run(["git", "commit", "-m", commit_msg], check=True)
                            
                            # Tag the model
                            tag_name = f"model-v{timestamp}"
                            tag_msg = f"Champion model mIoU: {self.eval_miou:.4f}"
                            subprocess.run(["git", "tag", tag_name, "-m", tag_msg], check=True)
                            
                            print(f"✅ Model promoted to DVC and tagged as {tag_name}")
                            
                        except subprocess.CalledProcessError:
                            print("⚠️ Git operations failed, but DVC model saved successfully")
                        
                        self.promotion_status = "promoted"
                        self.champion_model_path = champion_model_path
                        
                    else:
                        print("❌ Failed to add model to DVC")
                        print(result.stderr)
                        self.promotion_status = "failed"
                else:
                    print("❌ Latest model file not found")
                    self.promotion_status = "failed"
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Error during model promotion: {e}")
                self.promotion_status = "failed"
                
        else:
            print(f"📊 Model below threshold. mIoU: {self.eval_miou:.4f} ≤ {self.MIOU_THRESHOLD}")
            print("Model remains in MLflow only for analysis")
            self.promotion_status = "not_promoted"
        
        self.next(self.generate_report)
    
    @step
    def generate_report(self):
        """
        Generate a summary report of the pipeline execution.
        """
        print("--- Step: Generating Pipeline Report ---")
        
        report = {
            "pipeline_run_id": "1754854285525482",  # This would be dynamic in real implementation
            "mlflow_run_id": self.run_id,
            "model_performance": {
                "mIoU": self.eval_miou,
                "threshold": self.MIOU_THRESHOLD,
                "meets_threshold": self.eval_miou > self.MIOU_THRESHOLD
            },
            "promotion_status": self.promotion_status,
            "timestamp": int(time.time())
        }
        
        if hasattr(self, 'champion_model_path'):
            report["champion_model_path"] = self.champion_model_path
        
        # Save report
        report_file = f"pipeline_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Pipeline report saved: {report_file}")
        
        self.report = report
        self.next(self.end)
    
    @step
    def end(self):
        """
        The end of the pipeline.
        """
        print("--- Pipeline Summary ---")
        print("🎉 Pipeline finished successfully!")
        print(f"📊 Final mIoU: {self.eval_miou:.4f}")
        print(f"🏆 Promotion Status: {self.promotion_status}")
        print(f"📁 MLflow Run ID: {self.run_id}")
        
        if self.promotion_status == "promoted":
            print(f"🚀 Champion model saved and versioned with DVC!")
            print(f"📦 Model path: {self.champion_model_path}")
        elif self.promotion_status == "not_promoted":
            print("📈 Model needs improvement to reach promotion threshold")
            print(f"🎯 Target mIoU: > {self.MIOU_THRESHOLD}")
        else:
            print("❌ Model promotion failed due to technical issues")
        
        print("\n🔗 Next steps:")
        print("1. View results in MLflow UI: http://localhost:5000")
        if self.promotion_status == "promoted":
            print("2. Consider deploying the champion model")
            print("3. Set up monitoring for model performance")
        else:
            print("2. Analyze results and improve model performance")
            print("3. Run pipeline again with better hyperparameters")

if __name__ == '__main__':
    LaneDetectionPipeline()