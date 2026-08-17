"""Shared data-loading helpers for src/client.py and src/server.py."""
import numpy as np
import pandas as pd

META_COLS = ["label_binary", "label_multiclass", "label_name", "source"]


def load_partition_csv(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label_binary"].to_numpy(dtype=np.float32)
    return X, y, feature_cols
