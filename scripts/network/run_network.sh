#!/usr/bin/env bash
# Phase B — starts the OpenFlow controller (network/controller.py — a
# from-scratch OF1.3 controller, substituting for Ryu; see that file's
# docstring for why) in the background, then launches Mininet with the
# custom topology in interactive mode. Exit/Ctrl-D at the mininet> prompt
# tears Mininet down; this script then stops the controller too.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p results/network
LOG="results/network/controller.log"

echo "[run_network] starting controller, logging to $LOG"
"$PROJECT_ROOT/venv/bin/python3" network/controller.py > "$LOG" 2>&1 &
CONTROLLER_PID=$!
echo "$CONTROLLER_PID" > results/network/controller.pid
echo "[run_network] controller pid=$CONTROLLER_PID"

sleep 2
if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "[run_network] controller died on startup — see $LOG"
    cat "$LOG"
    exit 1
fi

echo "[run_network] starting Mininet (interactive) — 'exit' or Ctrl-D to tear down"
sudo mn --custom network/topo.py --topo mytopo --controller=remote,ip=127.0.0.1,port=6653

echo "[run_network] mininet exited, stopping controller (pid=$CONTROLLER_PID)"
kill "$CONTROLLER_PID" 2>/dev/null || true
rm -f results/network/controller.pid
