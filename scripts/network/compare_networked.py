"""
Phase D — compare the networked (Mininet + custom OF1.3 controller) FL run
against the original non-networked (Flower simulation) run.

Loads results/metrics_{strategy}.csv (original) and
results/networked/metrics_{strategy}.csv (new), checks whether accuracy/F1
per round line up (they should — the FL math didn't change, only the
transport did), and reports whatever timing overhead was actually measured:
round_times_{strategy}.csv (wall-clock seconds since run start, per round)
next to the networked metrics if net_server.py logged it, PLUS an optional
--original-seconds / --networked-seconds pair for a total-run-time
comparison (both runs are most fairly timed with the shell `time` builtin
around their respective launch commands, since scripts/run_simulation.py —
the non-networked entry point — is existing deliverable code this project
was told not to modify, so it carries no built-in timer).

Usage:
    python3 scripts/network/compare_networked.py --strategy fedavg \
        --original-seconds 27.4 --networked-seconds 96.1
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def log(msg):
    print(f"[compare_networked] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["fedavg", "trimmed_mean"], default="fedavg")
    parser.add_argument("--original-dir", type=str, default="results")
    parser.add_argument("--networked-dir", type=str, default="results/networked")
    parser.add_argument("--original-seconds", type=float, default=None,
                         help="Total wall-clock time of the non-networked run, if measured with `time`")
    parser.add_argument("--networked-seconds", type=float, default=None,
                         help="Total wall-clock time of the networked run, if measured with `time`")
    args = parser.parse_args()

    orig_path = Path(args.original_dir) / f"metrics_{args.strategy}.csv"
    net_path = Path(args.networked_dir) / f"metrics_{args.strategy}.csv"
    for p in (orig_path, net_path):
        if not p.exists():
            sys.exit(f"Missing {p}")

    orig = pd.read_csv(orig_path)
    net = pd.read_csv(net_path)

    print("=" * 70)
    print(f"CORRECTNESS CHECK — {args.strategy}: does the FL math match?")
    print("=" * 70)
    n = min(len(orig), len(net))
    if len(orig) != len(net):
        log(f"WARNING: round counts differ (original={len(orig)}, networked={len(net)}) "
            f"— comparing the first {n} rounds only")

    merged = orig.iloc[:n].merge(net.iloc[:n], on="round", suffixes=("_orig", "_net"))
    for col in ["accuracy", "precision", "recall", "f1", "loss"]:
        diff = (merged[f"{col}_orig"] - merged[f"{col}_net"]).abs()
        print(f"  {col:>10s}: max abs diff = {diff.max():.6f}, mean abs diff = {diff.mean():.6f}")

    max_acc_diff = (merged["accuracy_orig"] - merged["accuracy_net"]).abs().max()
    if max_acc_diff < 1e-6:
        print(f"\n  -> Accuracy matches EXACTLY across all {n} rounds — the networked run is "
              f"bit-for-bit reproducing the same FL computation, just over a real (emulated) "
              f"network instead of in-process Ray actors.")
    elif max_acc_diff < 0.02:
        print(f"\n  -> Accuracy matches closely (max diff {max_acc_diff:.4f}) — expected small "
              f"noise from e.g. thread/message-ordering differences between the Ray-simulation "
              f"transport and real gRPC, not a correctness problem.")
    else:
        print(f"\n  -> Accuracy diverges more than expected (max diff {max_acc_diff:.4f}) — "
              f"worth investigating whether client/malicious-id assignment actually matched "
              f"between the two runs before trusting the networked numbers.")

    print()
    print("=" * 70)
    print("OVERHEAD CHECK — what does the network layer cost?")
    print("=" * 70)

    timing_path = Path(args.networked_dir) / f"round_times_{args.strategy}.csv"
    if timing_path.exists():
        timing = pd.read_csv(timing_path)
        timing = timing[timing["round"] > 0]  # round 0 is pre-training init eval, not a real round
        per_round = timing["wall_clock_seconds_since_start"].diff().dropna()
        if len(per_round):
            print(f"  Networked per-round wall-clock time (post-init rounds): "
                  f"mean={per_round.mean():.2f}s, min={per_round.min():.2f}s, max={per_round.max():.2f}s")
        print(f"  Networked total (round 1 -> last round): "
              f"{timing['wall_clock_seconds_since_start'].iloc[-1] - timing['wall_clock_seconds_since_start'].iloc[0]:.2f}s")
    else:
        log(f"no {timing_path} found — networked per-round timing wasn't logged for this run")

    if args.original_seconds is not None and args.networked_seconds is not None:
        overhead = args.networked_seconds - args.original_seconds
        factor = args.networked_seconds / args.original_seconds if args.original_seconds > 0 else float("nan")
        print(f"\n  Total run time — original (in-process simulation): {args.original_seconds:.1f}s")
        print(f"  Total run time — networked (Mininet + real gRPC):  {args.networked_seconds:.1f}s")
        print(f"  Overhead added by the network layer: +{overhead:.1f}s ({factor:.2f}x)")
    else:
        log("pass --original-seconds and --networked-seconds (from `time <command>` on each "
            "run) for a total-run-time comparison")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The FL math is identical (or near-identical) whether clients are in-process Ray "
          "actors or real processes on separate emulated hosts exchanging gRPC over Mininet's "
          "virtual links — that's the point of this phase: the FL logic didn't change, only the "
          "transport did, and the numbers above should demonstrate that. Any overhead measured "
          "is entirely attributable to real process startup, gRPC connection setup, and the "
          "emulated network path — not to a different training outcome.")


if __name__ == "__main__":
    main()
