"""
Phase C — networked Flower client (real gRPC transport, not simulation).

Does NOT modify src/client.py — imports and reuses IDSClient unchanged
(fit/evaluate logic is transport-agnostic). The only thing simulation mode
didn't need and this adds is connecting to a real server_address over the
network, plus a --client-id CLI arg to pick which data/client_{ID}.csv
partition to load (simulation mode got this from Flower's Context instead).

Usage:
    python3 network/net_client.py --client-id 0 --server-address 10.0.0.1:8080 \
        [--malicious --poison-frac 0.5]
"""
import argparse
import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC_DIR)

from flwr.client import start_client  # noqa: E402

from client import IDSClient  # noqa: E402
from data_utils import load_partition_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Networked FL-IDS client (runs on a Mininet client host).")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server-address", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--malicious", action="store_true")
    parser.add_argument("--poison-frac", type=float, default=0.5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y, _ = load_partition_csv(f"{args.data_dir}/client_{args.client_id}.csv")
    print(f"[net_client {args.client_id}] loaded X={X.shape}, connecting to "
          f"{args.server_address}, malicious={args.malicious}", flush=True)

    client = IDSClient(
        args.client_id, X, y, malicious=args.malicious, poison_frac=args.poison_frac,
        local_epochs=args.local_epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
    )

    start_client(server_address=args.server_address, client=client.to_client(), insecure=True)

    print(f"[net_client {args.client_id}] RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
