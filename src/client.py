"""
Phase 2.2 — Flower NumPyClient.

Standalone sanity check (loads a single partition and runs one local
fit+evaluate cycle, no server involved):
    python3 src/client.py --client-id 0 [--malicious --poison-frac 0.5]

The actual FL run (Phase 2.4/2.5) drives clients through client_fn(context)
via flwr.simulation.start_simulation — see scripts/run_simulation.py, which
builds client_fn as a closure so malicious-client selection is controlled
centrally by the orchestration script rather than per-process CLI flags
(simulated clients are spawned by Flower internally, not launched one at a
time from the command line).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_partition_csv
from models import IDSNet

DEVICE = torch.device("cpu")


def poison_labels(y, poison_frac, seed):
    """Label-flipping poisoning: flip a fraction of binary labels (0<->1)."""
    if poison_frac <= 0:
        return y
    rng = np.random.default_rng(seed)
    y = y.copy()
    n_flip = int(len(y) * poison_frac)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y[flip_idx] = 1.0 - y[flip_idx]
    return y


def get_model_params(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_params(model, params):
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)


class IDSClient(NumPyClient):
    def __init__(self, client_id, X, y, malicious=False, poison_frac=0.5,
                 local_epochs=1, batch_size=64, lr=0.01, seed=42):
        self.client_id = client_id

        n_val = max(1, int(len(X) * 0.1))
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        self.X_train, self.y_train = X[train_idx], y[train_idx]
        self.X_val, self.y_val = X[val_idx], y[val_idx]

        self.malicious = malicious
        if malicious:
            self.y_train = poison_labels(self.y_train, poison_frac, seed + client_id)

        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = IDSNet(input_dim=X.shape[1]).to(DEVICE)

    def get_parameters(self, config):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        X = torch.tensor(self.X_train, dtype=torch.float32)
        y = torch.tensor(self.y_train, dtype=torch.float32)

        # IDS datasets here are heavily class-imbalanced (mostly-attack or
        # mostly-benign per client) — an unweighted loss lets the model
        # collapse to predicting the local majority class. Weight the
        # minority class using this client's own local label distribution.
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        n = len(X)
        for _ in range(self.local_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                xb, yb = X[batch_idx], y[batch_idx]
                opt.zero_grad()
                out = self.model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()

        return get_model_params(self.model), n, {
            "client_id": self.client_id, "malicious": self.malicious,
        }

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()
        X = torch.tensor(self.X_val, dtype=torch.float32)
        y = torch.tensor(self.y_val, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss()
        with torch.no_grad():
            out = self.model(X)
            loss = loss_fn(out, y).item()
            preds = (torch.sigmoid(out) > 0.5).float()
            acc = (preds == y).float().mean().item()
        return loss, len(X), {"accuracy": acc}


def client_fn_factory(data_dir, malicious_ids, poison_frac, local_epochs, batch_size, lr, seed):
    """Build a Flower client_fn(context) closure for use with start_simulation.

    malicious_ids: set[int] of partition ids that should poison their labels.
    """
    def client_fn(context):
        partition_id = int(context.node_config["partition-id"])
        X, y, _ = load_partition_csv(f"{data_dir}/client_{partition_id}.csv")
        malicious = partition_id in malicious_ids
        client = IDSClient(
            partition_id, X, y, malicious=malicious, poison_frac=poison_frac,
            local_epochs=local_epochs, batch_size=batch_size, lr=lr, seed=seed,
        )
        return client.to_client()

    return client_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone client sanity check.")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--malicious", action="store_true")
    parser.add_argument("--poison-frac", type=float, default=0.5)
    args = parser.parse_args()

    X, y, feature_cols = load_partition_csv(f"{args.data_dir}/client_{args.client_id}.csv")
    print(f"client_{args.client_id}: X={X.shape}, y positive rate={y.mean():.3f}, "
          f"malicious={args.malicious}")

    client = IDSClient(args.client_id, X, y, malicious=args.malicious,
                        poison_frac=args.poison_frac)
    params = client.get_parameters({})
    new_params, n, fit_metrics = client.fit(params, {})
    loss, n_val, eval_metrics = client.evaluate(new_params, {})
    print(f"local fit on {n} samples ok; local eval on {n_val} samples: "
          f"loss={loss:.4f}, {eval_metrics}")
