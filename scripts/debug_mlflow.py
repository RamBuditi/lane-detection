#!/usr/bin/env python3

import os
import mlflow
from mlflow.tracking import MlflowClient

def debug_mlflow_setup():
    """Debug MLflow setup and verify artifacts are accessible"""
    
    # Set the tracking URI
    mlflow_dir = "/app/mlruns"
    tracking_uri = f"file://{mlflow_dir}"
    
    print(f"🔍 Debugging MLflow Setup")
    print(f"=" * 50)
    
    # 1. Check if mlruns directory exists
    print(f"📁 MLflow directory: {mlflow_dir}")
    print(f"   Exists: {os.path.exists(mlflow_dir)}")
    
    if os.path.exists(mlflow_dir):
        print(f"   Contents: {os.listdir(mlflow_dir)}")
    
    # 2. Set tracking URI and verify
    mlflow.set_tracking_uri(tracking_uri)
    print(f"\n🎯 Tracking URI: {mlflow.get_tracking_uri()}")
    
    # 3. Initialize MLflow client
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        print(f"✅ MLflow client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing MLflow client: {e}")
        return
    
    # 4. List experiments
    try:
        experiments = client.search_experiments()
        print(f"\n📊 Found {len(experiments)} experiments:")
        
        for exp in experiments:
            print(f"   - ID: {exp.experiment_id}, Name: {exp.name}")
            
            # List runs for each experiment
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            print(f"     Runs: {len(runs)}")
            
            for run in runs[:3]:  # Show first 3 runs
                print(f"       - Run ID: {run.info.run_id}")
                print(f"         Status: {run.info.status}")
                print(f"         Artifacts URI: {run.info.artifact_uri}")
                
                # Check if artifacts exist
                try:
                    artifacts = client.list_artifacts(run.info.run_id)
                    print(f"         Artifacts ({len(artifacts)}):")
                    for artifact in artifacts:
                        print(f"           * {artifact.path} ({'dir' if artifact.is_dir else 'file'})")
                        if artifact.is_dir:
                            # List contents of directories
                            try:
                                sub_artifacts = client.list_artifacts(run.info.run_id, artifact.path)
                                for sub_artifact in sub_artifacts[:3]:  # Show first 3 sub-artifacts
                                    print(f"             - {sub_artifact.path}")
                                if len(sub_artifacts) > 3:
                                    print(f"             - ... and {len(sub_artifacts) - 3} more")
                            except Exception as e:
                                print(f"             Error listing sub-artifacts: {e}")
                except Exception as e:
                    print(f"         Error listing artifacts: {e}")
                print()
    
    except Exception as e:
        print(f"❌ Error searching experiments: {e}")
        return
    
    # 5. Test artifact download
    print(f"\n🧪 Testing artifact access...")
    try:
        if experiments and runs:
            latest_run = runs[0]
            artifacts = client.list_artifacts(latest_run.info.run_id)
            
            if artifacts:
                test_artifact = artifacts[0]
                if not test_artifact.is_dir:
                    # Try to download a file artifact
                    local_path = client.download_artifacts(latest_run.info.run_id, test_artifact.path)
                    print(f"✅ Successfully downloaded: {test_artifact.path} to {local_path}")
                else:
                    print(f"✅ Found directory artifact: {test_artifact.path}")
            else:
                print(f"⚠️  No artifacts found in latest run")
    except Exception as e:
        print(f"❌ Error testing artifact access: {e}")
    
    # 6. Provide UI startup command
    print(f"\n🌐 To start MLflow UI:")
    print(f"mlflow ui --backend-store-uri {tracking_uri} --host 0.0.0.0 --port 5000")
    print(f"\nThen open: http://localhost:5000")
    print(f"(If in Docker, make sure port 5000 is exposed: -p 5000:5000)")

if __name__ == "__main__":
    debug_mlflow_setup()