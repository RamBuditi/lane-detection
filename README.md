# End-to-End MLOps Lane Detection Pipeline

![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This repository contains a complete, end-to-end MLOps pipeline for training, evaluating, and managing a deep learning model for lane detection. The project is designed to demonstrate modern MLOps principles, including workflow orchestration, experiment tracking, model versioning, and reproducibility.

## ✨ Key Features

*   **Automated Workflow:** The entire ML lifecycle is orchestrated as a Directed Acyclic Graph (DAG) using **Metaflow**.
*   **Experiment Tracking:** All training runs, parameters, and metrics are logged and visualized with **MLflow**.
*   **Model Registry & Promotion:** Models are automatically registered in the MLflow Model Registry and promoted to "staging" or "production" tiers based on performance.
*   **Reproducible Environment:** The project is fully containerized with **Docker** and uses **Conda** for dependency management, ensuring a consistent environment.
*   **Large-Scale Data Versioning:** The CULane dataset is version-controlled with **DVC**, decoupling the data from the Git repository.
*   **Automated Hyperparameter Tuning:** Efficiently search for the best model hyperparameters using a parallelized tuning script.
*   **Structured Data Management:** Uses **SQLite** to index the dataset, moving beyond simple text files for more robust and queryable data handling.

## 🚀 Live Demo

Here is a sample of the model's output, detecting multiple lanes in a test image.

`[INSERT_DEMO_GIF_HERE]`

*(**Suggestion:** Create a GIF showing an input image/video on the left and the model's output with segmentation masks on the right.)*

## 🏗️ MLOps Architecture

The project follows a modern MLOps architecture that separates concerns and automates the flow of data and models from experimentation to a production-ready state.

`[INSERT_ARCHITECTURE_DIAGRAM_HERE]`

*(**Suggestion:** Create a diagram that shows the following flow:*
*1. **Data:** DVC pointing to cloud storage for the CULane dataset.*
*2. **Indexing:** The `create_db.py` script reads the DVC-tracked data and populates a central `lanes_dataset.db` (SQLite).*
*3. **Orchestration (Metaflow):** A box showing the Metaflow pipeline with its key steps (`validate`, `hyperparameter_tuning`, `train`, `evaluate`, `model_registry_management`).*
*4. **Tracking & Registry (MLflow):** The Metaflow pipeline communicates with MLflow throughout the process, logging parameters, metrics, and registering the final model.*
*5. **Inference:** The `inference.py` script shows how a final, versioned model can be loaded from the MLflow registry for use.)*

## 🔧 Tech Stack

*   **Orchestration & MLOps:** Metaflow, MLflow, Docker, DVC
*   **ML & Deep Learning:** PyTorch, Torchvision, Scikit-learn
*   **Data & Image Processing:** OpenCV, Albumentations, NumPy, Pandas
*   **Database:** SQLite
*   **Core Language:** Python 3.11

## ⚙️ Setup and Installation

### Prerequisites

*   [Git](https://git-scm.com/)
*   [Docker](https://www.docker.com/products/docker-desktop/) and Docker Compose
*   [DVC](https://dvc.org/doc/install)

### Step 1: Clone the Repository

```bash
git clone https://github.com/RamBuditi/lane-detection.git
cd lane-detection
```

### Step 2: Build the Docker Environment

The `Dockerfile` and `environment.yml` define the entire execution environment. Build the Docker image:

```bash
docker build -t lane-detection-env .
```

### Step 3: Set Up the Data

This project uses DVC to manage the CULane dataset.

1.  **Pull the data from DVC remote storage:**
    ```bash
    dvc pull
    ```
    This will download the `data/CULane` directory tracked by DVC.

2.  **Create the dataset index database:**
    Run the `create_db.py` script inside the Docker container to populate the SQLite database.
    ```bash
    docker run --rm -v "$(pwd):/app" lane-detection-env python src/create_db.py
    ```
    This will create a `lanes_dataset.db` file in the root directory.

## ⚡ Running the Pipeline

You can execute the entire MLOps pipeline using a single `docker run` command.

### Basic Run

This command runs the pipeline with default parameters (no hyperparameter tuning).

```bash
docker run --gpus all -v "$(pwd):/app" lane-detection-env python src/pipeline.py run
```

### Run with Hyperparameter Tuning

Enable the `--tune-hyperparams` flag to run the tuning step. You can control the number of parameter combinations to test with `--max-combinations`.

```bash
docker run --gpus all -v "$(pwd):/app" lane-detection-env \
  python src/pipeline.py run \
    --tune-hyperparams=True \
    --max-combinations=8
```

`[INSERT_TERMINAL_OUTPUT_SCREENSHOT_HERE]`

*(**Suggestion:** Add a screenshot showing the terminal output of a successful Metaflow run, highlighting the final "🎉 Enhanced Pipeline Completed Successfully!" message.)*

## 📊 Exploring the Results with MLflow

MLflow is used to track all experiments. You can launch the MLflow UI to view and compare runs.

1.  **Launch the MLflow UI Server:**
    Run the following command from the project root. It mounts the local `mlruns` directory into the container.

    ```bash
    docker run --rm -p 5000:5000 -v "$(pwd)/mlruns:/mlruns" lane-detection-env \
      mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri /mlruns
    ```

2.  **Open Your Browser:**
    Navigate to **[http://localhost:5000](http://localhost:5000)**.

### Experiment Comparison

`[INSERT_MLFLOW_EXPERIMENT_UI_SCREENSHOT_HERE]`

*(**Suggestion:** Add a screenshot of the MLflow experiment view, showing a few different runs with their parameters (like `learning_rate`) and metrics (like `best_val_miou`).)*

### Model Registry

`[INSERT_MLFLOW_REGISTRY_UI_SCREENSHOT_HERE]`

*(**Suggestion:** Add a screenshot of the MLflow Model Registry page, showing a registered `culane-lane-detector` model with its different versions and assigned aliases like "staging" or "production".)*

## 📂 Project Structure

```
.
├── data/
│   └── CULane/              # (Managed by DVC) Dataset directory
├── mlruns/                  # (Generated by MLflow) Experiment tracking data
├── models/                  # (Generated) Saved model checkpoints
├── src/
│   ├── pipeline.py          # Main Metaflow orchestration pipeline
│   ├── train.py             # Core PyTorch training script
│   ├── evaluate.py          # Final model evaluation script
│   ├── inference.py         # Script for running a saved model on new images
│   ├── hyperparameter_tuning.py # Logic for parallelized HP tuning
│   └── create_db.py         # Utility to build the SQLite dataset index
├── .dvc/                    # DVC metadata files
├── Dockerfile               # Defines the containerized environment
├── environment.yml          # Conda dependencies
├── lanes_dataset.db         # (Generated) SQLite database for dataset indexing
└── README.md                # This file
```

## 🛣️ Future Work & Roadmap

-   [ ] **CI/CD Integration:** Implement GitHub Actions to automatically run the pipeline on push/merge.
-   [ ] **Automated Testing:** Add unit tests for utility functions and integration tests for the pipeline.
-   [ ] **REST API for Inference:** Wrap `inference.py` in a FastAPI service to serve predictions over the network.
-   [ ] **Production Database:** Migrate from SQLite to a server-based database like PostgreSQL for better scalability.
-   [ ] **Model Monitoring:** Implement a system to monitor the deployed model for performance degradation or data drift.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📧 Contact

Ram Buditi - [rsb218@scarletmail.rutgers.edu](mailto:rsb218@scarletmail.rutgers.edu)

Project Link: [https://github.com/RamBuditi/lane-detection](https://github.com/RamBuditi/lane-detection)
```