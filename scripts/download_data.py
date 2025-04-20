#!/usr/bin/env python3
"""
download_data.py

Fetches and extracts the UCI 'Predict Students’ Dropout and Academic Success' dataset,
saving data.csv into the local `data/` directory.
"""

import os
import requests
import zipfile
from io import BytesIO

# URL of the UCI dataset zip (contains data.csv) :contentReference[oaicite:0]{index=0}
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict%2Bstudents%2Bdropout%2Band%2Bacademic%2Bsuccess.zip"
)
DATA_DIR = "data"

def download_and_extract():
    # 1) Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2) Download the zip archive
    print(f"Downloading dataset from {DATA_URL}...")
    resp = requests.get(DATA_URL)
    resp.raise_for_status()

    # 3) Open the zip in memory and locate the first .csv
    print("Extracting data.csv from archive...")
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        csv_files = [f for f in zf.namelist() if f.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded archive.")
        csv_name = csv_files[0]

        # 4) Extract & move/rename to data/data.csv
        zf.extract(csv_name, DATA_DIR)
        extracted = os.path.join(DATA_DIR, csv_name)
        final = os.path.join(DATA_DIR, "data.csv")
        if extracted != final:
            os.replace(extracted, final)

    print(f"✔ Saved: {final}")

if __name__ == "__main__":
    download_and_extract()