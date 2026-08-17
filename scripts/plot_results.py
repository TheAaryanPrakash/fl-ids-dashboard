"""
Phase 3 — Results & sanity check.

Loads {results_dir}/metrics_fedavg.csv and metrics_trimmed_mean.csv, plots
accuracy and F1 vs. round for both strategies (same run condition — e.g.
both clean or both under the same poisoning setup), saves
{results_dir}/comparison.png, and prints a short text summary of final
accuracy per strategy and whether the robust strategy actually outperformed
FedAvg.

Usage: python3 scripts/plot_results.py [results_dir] [condition_label]
  e.g. python3 scripts/plot_results.py results "under poisoning (2/5 malicious)"
       python3 scripts/plot_results.py results/baseline "clean baseline"
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STRATEGIES = ["fedavg", "trimmed_mean"]
COLORS = {"fedavg": "#d62728", "trimmed_mean": "#1f77b4"}


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    condition = sys.argv[2] if len(sys.argv) > 2 else ""
    title_suffix = f" ({condition})" if condition else ""

    data = {}
    for strat in STRATEGIES:
        path = f"{results_dir}/metrics_{strat}.csv"
        try:
            data[strat] = pd.read_csv(path)
        except FileNotFoundError:
            sys.exit(f"Missing {path} — run scripts/run_simulation.py --strategy {strat} first")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for strat, df in data.items():
        axes[0].plot(df["round"], df["accuracy"], marker="o", label=strat, color=COLORS[strat])
        axes[1].plot(df["round"], df["f1"], marker="o", label=strat, color=COLORS[strat])

    axes[0].set_title(f"Global model accuracy vs. round{title_suffix}")
    axes[0].set_xlabel("round")
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title(f"Global model F1 vs. round{title_suffix}")
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("F1 (attack class)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = f"{results_dir}/comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot_results] saved -> {out_path}")

    print("\n[plot_results] SUMMARY")
    print("=" * 60)
    finals = {}
    for strat, df in data.items():
        last = df.iloc[-1]
        finals[strat] = last
        print(f"{strat:>14s}: final round={int(last['round'])} "
              f"accuracy={last['accuracy']:.4f} precision={last['precision']:.4f} "
              f"recall={last['recall']:.4f} f1={last['f1']:.4f} loss={last['loss']:.4f}")

    # Round 0 is the identical pre-training random init for both strategies
    # (same seed) — exclude it from "which strategy held up better" checks,
    # since it says nothing about the aggregation strategy itself.
    n_rounds = len(data["fedavg"])
    mid = max(1, n_rounds // 2)
    post_init = {strat: df[df["round"] > 0] for strat, df in data.items()}
    early_avg = {strat: df.iloc[:mid]["accuracy"].mean() for strat, df in post_init.items()}
    rounds_ahead = int((post_init["trimmed_mean"]["accuracy"].to_numpy()
                         > post_init["fedavg"]["accuracy"].to_numpy()).sum())

    print(f"\nAverage accuracy over the first {mid} post-init rounds (early robustness window):")
    for strat, acc in early_avg.items():
        print(f"{strat:>14s}: {acc:.4f}")
    print(f"trimmed_mean had higher accuracy than fedavg in {rounds_ahead}/{len(post_init['fedavg'])} "
          f"post-init rounds.")

    fa_final = finals["fedavg"]["accuracy"]
    tm_final = finals["trimmed_mean"]["accuracy"]

    print()
    if tm_final > fa_final:
        print(f"Final-round result: trimmed_mean ({tm_final:.4f}) outperformed "
              f"fedavg ({fa_final:.4f}) at the final round.")
    else:
        print(f"Final-round result: trimmed_mean ({tm_final:.4f}) did NOT beat "
              f"fedavg ({fa_final:.4f}) at the final round — reporting as observed, "
              f"not forcing the expected outcome.")

    if early_avg["trimmed_mean"] > early_avg["fedavg"]:
        print(f"Early-training result: trimmed_mean averaged {early_avg['trimmed_mean']:.4f} "
              f"accuracy vs. fedavg's {early_avg['fedavg']:.4f} over the first {mid} rounds "
              f"after init.")
    else:
        print(f"Early-training result: fedavg averaged {early_avg['fedavg']:.4f} accuracy vs. "
              f"trimmed_mean's {early_avg['trimmed_mean']:.4f} over the first {mid} rounds "
              f"after init — no early robustness advantage observed here.")


if __name__ == "__main__":
    main()
