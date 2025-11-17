# src/utils.py
import random, numpy as np, os, json
from datetime import datetime

def ensure_dirs():
    for d in ["data", "models", "results", "data/ml_raw"]:
        os.makedirs(d, exist_ok=True)

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
