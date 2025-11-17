# src/ml_pipeline.py
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib
from src.utils import ensure_dirs
ensure_dirs()

def build_dataset_from_states(states_list, horizon_ticks=4):
    rows=[]
    for per_req, state, summary in states_list:
        for v in state['vms']:
            hist = v.history
            for i in range(len(hist)-horizon_ticks):
                t_now, r_now, c_now = hist[i]
                t_fut, r_fut, c_fut = hist[i+horizon_ticks]
                rows.append({'vm_id':int(v.id), 'time':float(t_now),
                             'ram_now':float(r_now), 'cpu_now':float(c_now),
                             'ram_fut':float(r_fut), 'cpu_fut':float(c_fut)})
    df = pd.DataFrame(rows)
    return df

def train_rf_multi(df, model_path="models/rf_ramcpu_pred.pkl"):
    X = df[['ram_now','cpu_now']].values
    y = df[['ram_fut','cpu_fut']].values
    base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, model_path)
    print(f"Saved RF multi-output model to {model_path}")
    return model

def load_model(path):
    return joblib.load(path)
