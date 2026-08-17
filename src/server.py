"""
Phase 2.3 — Flower server / strategy definitions.

Provides the pieces scripts/run_simulation.py wires together: a centralized
evaluate_fn (runs the global model on data/test.csv each round and logs
accuracy/precision/recall/F1/loss to results/metrics_{strategy}.csv) and a
get_strategy() factory that switches between:
  - fedavg        : flwr.server.strategy.FedAvg (baseline)
  - trimmed_mean  : flwr.server.strategy.FedTrimmedAvg (robust alternative;
                    built-in coordinate-wise trimmed mean over client updates)

Simulation mode (Phase 2.4) runs the server and all clients in one process
via flwr.simulation.start_simulation, so there is no standalone server
process to launch by hand here — run via scripts/run_simulation.py.
"""
import csv
import os
import sys
from pathlib import Path

import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg, FedTrimmedAvg
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_partition_csv
from models import IDSNet

DEVICE = torch.device("cpu")


def make_evaluate_fn(test_csv_path, metrics_csv_path, input_dim):
    """Centralized evaluation on the shared held-out test set, called by the
    strategy after each round's aggregation. Appends one row per round to
    metrics_csv_path."""
    X_test, y_test, _ = load_partition_csv(test_csv_path)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = IDSNet(input_dim=input_dim).to(DEVICE)

    os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
    header = ["round", "loss", "accuracy", "precision", "recall", "f1"]
    with open(metrics_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    def evaluate_fn(server_round, parameters, config):
        keys = list(model.state_dict().keys())
        state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        with torch.no_grad():
            out = model(X_test_t)
            loss = loss_fn(out, y_test_t).item()
            preds = (torch.sigmoid(out) > 0.5).float().numpy()

        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0,
        )

        with open(metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([server_round, loss, acc, precision, recall, f1])

        print(f"[server] round {server_round}: loss={loss:.4f} acc={acc:.4f} "
              f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

        return loss, {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    return evaluate_fn


def get_strategy(name, num_clients, clients_per_round, evaluate_fn,
                  initial_parameters=None, beta=0.2):
    fraction_fit = clients_per_round / num_clients
    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=clients_per_round,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_parameters,
    )

    if name == "fedavg":
        return FedAvg(**common_kwargs)
    if name == "trimmed_mean":
        return FedTrimmedAvg(beta=beta, **common_kwargs)
    raise ValueError(f"Unknown strategy {name!r}, expected 'fedavg' or 'trimmed_mean'")


def make_server_config(num_rounds):
    return ServerConfig(num_rounds=num_rounds)


def initial_parameters_from_model(input_dim):
    model = IDSNet(input_dim=input_dim)
    return ndarrays_to_parameters([val.cpu().numpy() for val in model.state_dict().values()])
