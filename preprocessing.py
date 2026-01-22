# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 14:00:30 2026

Preprocess raw ASRT data to get them ready for RTV data analysis

@author: EmanueleCiardo
"""

# Libraries
import pandas as pd
from pathlib import Path

# Base folder 
data_path = Path(r"C:\Users\EmanueleCiardo\Desktop\analyses\RTV\data")

#%% --- Load the data for dataset 1 ---
day = 1
in_path = data_path / "raw" / "dataset_1" / f"s{day}.csv"
df = pd.read_csv(in_path)

# Add here cleaning steps if needed
# df = ...

# --- Save ---
out_dir = data_path / "processed" / "dataset_1"
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "df.csv"
df.to_csv(out_path, index=False)

#%% --- Load the data for dataset 2 ---
in_path = data_path / "raw" / "dataset_2" / f"s{day}.csv"
df = pd.read_csv(path, encoding="latin1")
