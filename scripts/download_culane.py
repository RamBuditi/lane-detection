# scripts/download_culane.py
import os
import subprocess

GDRIVE_FOLDER_ID = "1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu"
DATA_DIR = "data/CULane"

os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading CULane dataset from Google Drive...")
subprocess.run([
    "gdown",
    f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}",
    "--folder",
    "--output",
    DATA_DIR
], check=True)

print(f"Download complete! Dataset is saved in '{DATA_DIR}'")
