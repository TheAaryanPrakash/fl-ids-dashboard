"""
Phase 2.5 — Simulation entry point.

Wires together data partitions, the client_fn, strategy selection
(fedavg / trimmed_mean), and number of rounds, then runs everything through
Flower's simulation runtime (server + all N_CLIENTS clients in one local
process — flwr.simulation.start_simulation).

Usage:
    python3 scripts/run_simulation.py --strategy fedavg --rounds 20 --num-malicious 2
    python3 scripts/run_simulation.py --strategy trimmed_mean --rounds 20 --num-malicious 2
"""
import argparse
import glob
import os
import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC_DIR)
# Ray spawns each simulated client as a separate worker process that does
# NOT inherit this process's runtime sys.path mutation above — only the
# driver process sees it. Without PYTHONPATH set, worker processes fail to
# import data_utils/models/client and every fit()/evaluate() call silently
# errors out (aggregate_fit/aggregate_evaluate just see 0 results).
os.environ["PYTHONPATH"] = SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")

import torch  # noqa: E402
from flwr.server import ServerConfig  # noqa: E402
from flwr.simulation import start_simulation  # noqa: E402

from client import client_fn_factory  # noqa: E402
from data_utils import load_partition_csv  # noqa: E402
from server import get_strategy, initial_parameters_from_model, make_evaluate_fn  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run an FL-IDS simulation with Flower.")
    parser.add_argument("--strategy", choices=["fedavg", "trimmed_mean"], default="fedavg")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-malicious", type=int, default=0,
                         help="Number of clients (ids 0..k-1) that poison their labels")
    parser.add_argument("--poison-frac", type=float, default=0.5,
                         help="Fraction of a malicious client's local labels to flip")
    parser.add_argument("--clients-per-round", type=int, default=None,
                         help="Defaults to all discovered clients")
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.2,
                         help="FedTrimmedAvg trim fraction per side")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--num-cpus", type=int, default=2,
                         help="CPUs given to the local Ray cluster for simulation")
    args = parser.parse_args()

    client_files = sorted(glob.glob(f"{args.data_dir}/client_*.csv"))
    num_clients = len(client_files)
    if num_clients == 0:
        sys.exit(f"No client partitions found in {args.data_dir}/ — run data/prepare_data.py first")

    clients_per_round = args.clients_per_round or num_clients
    malicious_ids = set(range(args.num_malicious))

    X0, _, _ = load_partition_csv(client_files[0])
    input_dim = X0.shape[1]

    print(f"[run_simulation] strategy={args.strategy} rounds={args.rounds} "
          f"num_clients={num_clients} clients_per_round={clients_per_round} "
          f"malicious_ids={sorted(malicious_ids)} poison_frac={args.poison_frac} "
          f"input_dim={input_dim}")

    client_fn = client_fn_factory(
        data_dir=args.data_dir,
        malicious_ids=malicious_ids,
        poison_frac=args.poison_frac,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    metrics_csv_path = f"{args.results_dir}/metrics_{args.strategy}.csv"
    evaluate_fn = make_evaluate_fn(f"{args.data_dir}/test.csv", metrics_csv_path, input_dim)
    # Fix the initial global model weights so fedavg and trimmed_mean runs
    # start from the identical point — otherwise round-0 accuracy differs
    # purely from unseeded random init, not from the aggregation strategy.
    torch.manual_seed(args.seed)
    initial_parameters = initial_parameters_from_model(input_dim)

    strategy = get_strategy(
        args.strategy, num_clients, clients_per_round, evaluate_fn,
        initial_parameters=initial_parameters, beta=args.beta,
    )

    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        ray_init_args={"num_cpus": args.num_cpus, "ignore_reinit_error": True,
                        "log_to_driver": False, "logging_level": "ERROR"},
        client_resources={"num_cpus": 1},
    )

    print(f"\n[run_simulation] done. metrics written to {metrics_csv_path}")
    if history.metrics_centralized:
        for key, values in history.metrics_centralized.items():
            last_round, last_val = values[-1]
            print(f"[run_simulation] final centralized {key} (round {last_round}): {last_val:.4f}")


if __name__ == "__main__":
    main()
