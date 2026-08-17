# FL-IDS: Federated Learning Intrusion Detection, Poisoning, and SDN Networking

**[Live dashboard →](https://theaaryanprakash.github.io/fl-ids-dashboard/)**

A federated-learning intrusion detection system, trained non-IID across simulated clients,
stress-tested under label-flipping poisoning with a robust aggregation defense, and finally
run over a real emulated network through a custom OpenFlow controller. Built end-to-end with
[Claude Code](https://claude.com/claude-code).

## Pipeline

```
raw datasets → non-IID partitioning → Flower FL simulation → Mininet + SDN controller → results comparison
```

1. **Raw datasets** — two pre-cleaned IDS CSVs: `bccc_cleaned.csv` (35,024 rows, 104 MQTT/IoT
   protocol-level features, 18 classes) and `cic_cleaned.csv` (8,000 rows, 50 generic
   network-flow features, 8 classes). The two share exactly one column — the label — so they
   were treated as two different data domains rather than concatenated.
2. **Non-IID partitioning** — split across 5 simulated clients (3 from the first domain, 2 from
   the second), with an additional Dirichlet(α=0.5) label skew within each domain. Since the two
   domains share zero features, both were placed into a single 154-column union feature space
   (each client's own domain populated, the other domain's columns zero-padded), so one global
   model architecture works for every client.
3. **Flower FL simulation** — a small MLP trained via [Flower](https://flower.ai), comparing
   FedAvg against FedTrimmedAvg (coordinate-wise robust aggregation), each run once clean and
   once with 2 of 5 clients performing label-flipping poisoning.
4. **Mininet + SDN controller** — the identical FL client/server code run again unmodified, but
   as separate processes on separate emulated hosts, talking real gRPC over
   [Mininet](http://mininet.org) virtual links through a from-scratch OpenFlow 1.3 controller
   (see [Notes](#notes--known-limitations) for why it isn't Ryu).
5. **Results comparison** — local vs. networked, clean vs. poisoned, summarized on the
   [live dashboard](https://theaaryanprakash.github.io/fl-ids-dashboard/).

## Key results

*(final round = round 20; full per-round data and charts are on the dashboard)*

| | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| FedAvg, clean | 0.7275 | 0.8282 | 0.9957 | 0.7090 |
| Trimmed Mean, clean | 0.7337 | 0.8354 | 0.9775 | 0.7294 |
| FedAvg, poisoned (2/5 malicious) | 0.5232 | 0.6545 | 0.9961 | 0.4873 |
| Trimmed Mean, poisoned (2/5 malicious) | 0.5327 | 0.6640 | 0.9943 | 0.4985 |
| FedAvg, poisoned + networked (Mininet) | 0.5215 | 0.6527 | 0.9966 | 0.4852 |

- Under poisoning, Trimmed Mean beat FedAvg in every post-init round for the first half of
  training (~0.64 vs. ~0.53 average accuracy) but both converged to similar degraded accuracy by
  round 20 — a real but temporary robustness advantage at this corruption ratio (40% malicious).
- The clean baseline confirms the poisoned-run collapse is attributable to the attack itself, not
  a training-setup flaw — and the robust strategy costs essentially nothing when there's no
  attack to defend against.
- The networked run tracks the local in-process simulation closely (max accuracy diff ~0.01,
  consistent with ordinary unseeded-training noise) — the FL math is unchanged by the transport.
  Total wall-clock overhead was modest (~4s / 1.14x for a 20-round run), and steady-state
  per-round latency was small (~0.8s/round) — most of the added cost is one-time process/gRPC
  connection startup, not per-round network latency, at this topology size.

## Repo structure

```
data/               raw + partitioned datasets, inspection/preparation scripts
src/                model, client, server (Flower NumPyClient/strategy definitions)
scripts/            simulation entry point, plotting; scripts/network/ = Mininet orchestration
network/            Mininet topology + from-scratch OpenFlow 1.3 controller + networked client/server
results/            metrics CSVs + comparison plots (clean, poisoned, baseline, networked)
visualization/      self-contained HTML dashboard (this repo's GitHub Pages site)
```

## Reproduce locally

Requires Python 3.10+, Mininet + Open vSwitch (for the networking phases only).

```bash
python3 -m venv venv && source venv/bin/activate
pip install flwr flwr-datasets torch torchvision scikit-learn pandas numpy matplotlib ray python-openflow

python3 data/inspect_data.py
python3 data/prepare_data.py --n-clients 5

python3 scripts/run_simulation.py --strategy fedavg --rounds 20 --num-malicious 2
python3 scripts/run_simulation.py --strategy trimmed_mean --rounds 20 --num-malicious 2 --beta 0.4
python3 scripts/plot_results.py

# networking phases need root (Mininet) — see network/controller.py for why Ryu isn't used
sudo python3 scripts/network/run_networked_experiment.py --strategy fedavg --num-malicious 2
python3 scripts/network/compare_networked.py --strategy fedavg
```

## Notes / known limitations

- **Ryu is not used as the SDN controller.** It's unmaintained since ~2021 and unrunnable on any
  currently-installable Python — full diagnosis is in `network/controller.py`'s module
  docstring. What's running instead is a from-scratch OpenFlow 1.3 learning-switch controller
  built on the `python-openflow` protocol library plus plain Python sockets.
- IDS-driven SDN actions (e.g. quarantining a host the model flags as malicious) are **not**
  built yet — this repo covers the FL + networking integration, not the closed-loop response.
- The MLP and hyperparameters here are tuned for "get a working, honest comparison," not for
  state-of-the-art detection accuracy.
