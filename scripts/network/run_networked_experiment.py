"""
Phase C — scriptable (non-interactive) networked FL experiment.

Uses Mininet's Python API directly (mininet.net.Mininet), not the
interactive CLI, so the whole run — controller start, network build,
server+clients launch, completion wait, teardown — is one script.

IMPORTANT: this script itself must run under the SYSTEM python3
(/usr/bin/python3), not the project venv — the `mininet` package only
exists in system dist-packages (it depends on compiled components,
mnexec/OVS bindings, installed by `apt install mininet`; there is no
working pip-installable equivalent). It shells out to the project venv's
python3 to actually run network/controller.py, network/net_server.py, and
network/net_client.py, which need flwr/torch/pyof instead.

Must run as root (Mininet requirement):
    sudo python3 scripts/network/run_networked_experiment.py \
        --strategy fedavg --num-malicious 2 --rounds 20
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from mininet.link import TCLink  # noqa: E402
from mininet.log import setLogLevel  # noqa: E402
from mininet.net import Mininet  # noqa: E402
from mininet.node import RemoteController  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "network"))
from topo import FLTopo  # noqa: E402


def log(msg):
    print(f"[run_networked_experiment] {msg}", flush=True)


def wait_for_marker(log_path, marker, timeout, poll_interval=2):
    start = time.time()
    while time.time() - start < timeout:
        if log_path.exists() and marker in log_path.read_text():
            return True
        time.sleep(poll_interval)
    return False


def count_csv_rows(path):
    if not path.exists():
        return 0
    with open(path) as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)  # minus header


def main():
    parser = argparse.ArgumentParser(description="Run the FL-IDS experiment over an emulated Mininet network.")
    parser.add_argument("--strategy", choices=["fedavg", "trimmed_mean"], default="fedavg")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--num-malicious", type=int, default=2)
    parser.add_argument("--poison-frac", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clients", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="results/networked")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    if os.geteuid() != 0:
        sys.exit("This script must run as root (Mininet requirement) — use sudo.")

    setLogLevel("info")

    results_dir = PROJECT_ROOT / args.results_dir
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    venv_python = str(PROJECT_ROOT / "venv" / "bin" / "python3")
    controller_script = str(PROJECT_ROOT / "network" / "controller.py")
    server_script = str(PROJECT_ROOT / "network" / "net_server.py")
    client_script = str(PROJECT_ROOT / "network" / "net_client.py")

    # Read just the header of client_0.csv to get the feature count without
    # needing pandas in this (system-python) process.
    with open(PROJECT_ROOT / args.data_dir / "client_0.csv") as f:
        header = f.readline().strip().split(",")
    meta_cols = {"label_binary", "label_multiclass", "label_name", "source"}
    input_dim = len([c for c in header if c not in meta_cols])
    log(f"input_dim={input_dim} (from {args.data_dir}/client_0.csv header)")

    controller_log = logs_dir / "controller.log"
    log(f"starting OpenFlow controller, log -> {controller_log}")
    controller_proc = subprocess.Popen(
        [venv_python, controller_script, "--port", str(args.controller_port)],
        stdout=open(controller_log, "w"), stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    time.sleep(2)
    if controller_proc.poll() is not None:
        sys.exit(f"controller died on startup, see {controller_log}")

    net = Mininet(topo=FLTopo(n_clients=args.n_clients), link=TCLink, build=False)
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=args.controller_port)
    net.build()

    try:
        net.start()
        log("mininet network started")
        log(f"pinging all hosts to confirm connectivity before launching FL: "
            f"dropped={net.pingAll()}%")

        server_host = net.get("h1")
        server_ip = server_host.IP()

        server_log = logs_dir / f"server_{args.strategy}.log"
        malicious_ids = set(range(args.num_malicious))

        server_cmd = (
            f"cd {PROJECT_ROOT} && {venv_python} {server_script} --strategy {args.strategy} "
            f"--rounds {args.rounds} --num-clients {args.n_clients} "
            f"--server-address 0.0.0.0:{args.server_port} --data-dir {args.data_dir} "
            f"--results-dir {args.results_dir} --beta {args.beta} --seed {args.seed} "
            f"--input-dim {input_dim} > {server_log} 2>&1 &"
        )
        log(f"launching server on {server_host.name} ({server_ip}), log -> {server_log}")
        server_host.cmd(server_cmd)

        time.sleep(3)  # let the server bind before clients try to connect

        for i in range(args.n_clients):
            client_host = net.get(f"h{i + 2}")
            malicious_flag = "--malicious" if i in malicious_ids else ""
            client_log = logs_dir / f"client_{i}.log"
            client_cmd = (
                f"cd {PROJECT_ROOT} && {venv_python} {client_script} --client-id {i} "
                f"--server-address {server_ip}:{args.server_port} --data-dir {args.data_dir} "
                f"--poison-frac {args.poison_frac} --seed {args.seed} {malicious_flag} "
                f"> {client_log} 2>&1 &"
            )
            log(f"launching client {i} on {client_host.name} ({client_host.IP()}) "
                f"malicious={i in malicious_ids}, log -> {client_log}")
            client_host.cmd(client_cmd)

        log(f"waiting for run to complete (timeout={args.timeout}s)...")
        completed = wait_for_marker(server_log, "RUN_COMPLETE", args.timeout)
        if completed:
            log("server reported RUN_COMPLETE")
        else:
            log("TIMEOUT waiting for server RUN_COMPLETE marker")

        time.sleep(2)  # let clients finish disconnecting cleanly

    finally:
        log("tearing down mininet")
        net.stop()
        log("stopping controller")
        controller_proc.terminate()
        try:
            controller_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            controller_proc.kill()

    metrics_path = results_dir / f"metrics_{args.strategy}.csv"
    n_rows = count_csv_rows(metrics_path)
    if n_rows >= args.rounds:
        log(f"SUCCESS: {metrics_path} has {n_rows} rounds of metrics")
    else:
        log(f"FAILURE: {metrics_path} has only {n_rows}/{args.rounds} rounds — "
            f"check logs in {logs_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
