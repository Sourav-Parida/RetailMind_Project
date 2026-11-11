from pathlib import Path 

import zipfile  
# Repository root 

ROOT = Path(__file__).resolve().parent  
# Change this to your ZIP filename 

ZIP_PATH = ROOT / "fashion-product-images-small.zip"  
# Extract into data folder 
# 

DATA_DIR = ROOT
DATA_DIR.mkdir(parents=True, exist_ok=True)  

if not ZIP_PATH.exists():     
    raise FileNotFoundError(f"ZIP file not found: {ZIP_PATH}")  
    
print("Extracting dataset...")  

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:     
    zip_ref.extractall(DATA_DIR)  
    
print("Extraction completed!")
print(f"Dataset extracted to: {DATA_DIR}")