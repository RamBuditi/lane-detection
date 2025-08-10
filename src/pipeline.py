from metaflow import FlowSpec, step
import os
import subprocess

class LaneDetectionPipeline(FlowSpec):
    """
    A Metaflow pipeline that trains and evaluates a lane detection model.
    """

    @step
    def start(self):
        """
        The starting point of the pipeline.
        """
        print("Pipeline starting...")
        self.next(self.train_model)

    @step
    def train_model(self):
        """
        This step runs the training script as a separate process.
        """
        print("--- Step: Training Model ---")
        
        # We run the script as a subprocess to keep the logic separate
        # and to capture its output.
        result = subprocess.run(
            ["python", "src/train.py"], 
            capture_output=True, 
            text=True
        )
        
        # Check if the training script ran successfully
        if result.returncode != 0:
            print("Training script failed!")
            print(result.stderr)
            raise Exception("Training failed")
        
        print(result.stdout)
        
        # Find the Run ID from the training script's output
        output_lines = result.stdout.splitlines()
        run_id_line = next((line for line in output_lines if "MLflow Run ID:" in line), None)
        
        if run_id_line is None:
            raise Exception("Could not find MLflow Run ID in the training output.")
            
        self.run_id = run_id_line.split(":")[1].strip()
        print(f"Training complete. Found Run ID: {self.run_id}")

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
            print("Evaluation script failed!")
            print(result.stderr)
            raise Exception("Evaluation failed")
        
        print(result.stdout)
        print(f"Evaluation complete for Run ID: {self.run_id}")
        
        self.next(self.end)

    @step
    def end(self):
        """
        The end of the pipeline.
        """
        print("Pipeline finished successfully!")
        print(f"The results for the model from Run ID '{self.run_id}' are now in MLflow.")
        print("Next steps: Go to the MLflow UI, review the metrics, and register the model if it's good.")

if __name__ == '__main__':
    LaneDetectionPipeline()