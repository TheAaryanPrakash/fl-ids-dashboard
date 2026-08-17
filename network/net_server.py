"""
Phase C — networked Flower server (real gRPC transport, not simulation).

This does NOT modify src/server.py — it imports and reuses get_strategy(),
make_evaluate_fn(), and initial_parameters_from_model() from it unchanged
(those are transport-agnostic: strategy/evaluation logic, same whether
clients are Ray actors on one machine or real processes on separate
Mininet hosts). The only thing simulation mode didn't need and this adds is
a real server_address to bind to, since traffic must now actually cross the
emulated network.

Usage:
    python3 network/net_server.py --strategy fedavg --rounds 20 \
        --server-address 10.0.0.1:8080 --num-clients 5 --input-dim 154
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402
from flwr.server import ServerConfig, start_server  # noqa: E402

from server import get_strategy, initial_parameters_from_model, make_evaluate_fn  # noqa: E402


def wrap_with_round_timing(evaluate_fn, timing_csv_path):
    """Wrap evaluate_fn to additionally log wall-clock time per round to a
    SEPARATE side file, so Phase D can measure real network overhead without
    touching src/server.py's metrics CSV schema at all (keeping it directly
    comparable, row-for-row, to the non-networked results/metrics_*.csv)."""
    os.makedirs(os.path.dirname(timing_csv_path), exist_ok=True)
    with open(timing_csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "wall_clock_seconds_since_start"])
    start_time = time.monotonic()

    def timed_evaluate_fn(server_round, parameters, config):
        result = evaluate_fn(server_round, parameters, config)
        elapsed = time.monotonic() - start_time
        with open(timing_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([server_round, elapsed])
        return result

    return timed_evaluate_fn


def main():
    parser = argparse.ArgumentParser(description="Networked FL-IDS server (runs on the Mininet 'server' host).")
    parser.add_argument("--strategy", choices=["fedavg", "trimmed_mean"], default="fedavg")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results/networked")
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-dim", type=int, required=True)
    args = parser.parse_args()

    clients_per_round = args.clients_per_round or args.num_clients
    metrics_csv_path = f"{args.results_dir}/metrics_{args.strategy}.csv"
    evaluate_fn = make_evaluate_fn(f"{args.data_dir}/test.csv", metrics_csv_path, args.input_dim)
    timing_csv_path = f"{args.results_dir}/round_times_{args.strategy}.csv"
    evaluate_fn = wrap_with_round_timing(evaluate_fn, timing_csv_path)

    torch.manual_seed(args.seed)
    initial_parameters = initial_parameters_from_model(args.input_dim)

    strategy = get_strategy(
        args.strategy, args.num_clients, clients_per_round, evaluate_fn,
        initial_parameters=initial_parameters, beta=args.beta,
    )

    print(f"[net_server] listening on {args.server_address} | strategy={args.strategy} "
          f"rounds={args.rounds} num_clients={args.num_clients} "
          f"clients_per_round={clients_per_round}", flush=True)

    start_server(
        server_address=args.server_address,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    print(f"[net_server] RUN_COMPLETE strategy={args.strategy}", flush=True)


if __name__ == "__main__":
    main()
