from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qst.utils import ensure_dir


METHOD_LABELS = {
    "clean_nn": "Clean NN",
    "adversarial_nn": "Adv. NN",
    "linear_inversion": "Linear inversion",
    "mle": "MLE",
    "bayesian_particle": "Bayesian",
    "compressed_sensing": "Compressed sensing",
    "purification_mle": "Purification MLE",
}

ATTACK_LABELS = {
    "clean": "Clean",
    "random_replacement": "Random repl.",
    "targeted_replacement": "Targeted repl.",
    "worst_replacement": "Worst repl.",
    "frequency_pgd": "Frequency PGD",
}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_method_attack_bars(frame: pd.DataFrame, figures_dir: Path) -> None:
    summary = frame.groupby(["num_qubits", "method", "attack"], as_index=False).agg(
        mean_fidelity=("fidelity", "mean"),
        std_fidelity=("fidelity", "std"),
    )
    for qubits, qframe in summary.groupby("num_qubits"):
        attacks = list(qframe["attack"].drop_duplicates())
        methods = list(qframe["method"].drop_duplicates())
        x = np.arange(len(attacks))
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(attacks)), 5))
        for j, method in enumerate(methods):
            values = []
            errors = []
            for attack in attacks:
                rows = qframe[(qframe["method"] == method) & (qframe["attack"] == attack)]
                if rows.empty:
                    values.append(np.nan)
                    errors.append(0.0)
                else:
                    values.append(float(rows["mean_fidelity"].iloc[0]))
                    errors.append(float(rows["std_fidelity"].fillna(0.0).iloc[0]))
            ax.bar(x + (j - (len(methods) - 1) / 2) * width, values, width, label=METHOD_LABELS.get(method, method), yerr=errors, capsize=2)
        ax.set_title(f"Method comparison under attacks ({qubits} qubit{'s' if qubits > 1 else ''})")
        ax.set_ylabel("Mean fidelity")
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels([ATTACK_LABELS.get(a, a) for a in attacks], rotation=25, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        _save(fig, figures_dir / f"q{qubits}_method_attack_fidelity")


def plot_infidelity_cdf(frame: pd.DataFrame, figures_dir: Path) -> None:
    for (qubits, attack), group in frame.groupby(["num_qubits", "attack"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, mframe in group.groupby("method"):
            values = np.sort(mframe["infidelity"].to_numpy())
            if len(values) == 0:
                continue
            cdf = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, cdf, label=METHOD_LABELS.get(method, method))
        ax.set_title(f"Empirical CDF of infidelity, {ATTACK_LABELS.get(attack, attack)} ({qubits} qubit{'s' if qubits > 1 else ''})")
        ax.set_xlabel("Infidelity")
        ax.set_ylabel("Empirical CDF")
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _save(fig, figures_dir / f"q{qubits}_{attack}_infidelity_cdf")


def plot_alpha_sweep(sweep: pd.DataFrame, figures_dir: Path) -> None:
    summary = sweep.groupby(["num_qubits", "method", "attack", "alpha_requested"], as_index=False).agg(
        mean_fidelity=("fidelity", "mean"),
        std_fidelity=("fidelity", "std"),
    )
    for (qubits, attack), group in summary.groupby(["num_qubits", "attack"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, mframe in group.groupby("method"):
            mframe = mframe.sort_values("alpha_requested")
            x = mframe["alpha_requested"].to_numpy()
            y = mframe["mean_fidelity"].to_numpy()
            e = mframe["std_fidelity"].fillna(0.0).to_numpy()
            lower = np.clip(y - e, 0.0, 1.0)
            upper = np.clip(y + e, 0.0, 1.0)
            ax.plot(x, y, marker="o", label=METHOD_LABELS.get(method, method))
            ax.fill_between(x, lower, upper, alpha=0.12)
        ax.set_title(f"Robustness versus replacement fraction, {ATTACK_LABELS.get(attack, attack)} ({qubits} qubit{'s' if qubits > 1 else ''})")
        ax.set_xlabel("Requested replacement fraction $\\alpha$")
        ax.set_ylabel("Mean fidelity")
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        _save(fig, figures_dir / f"q{qubits}_{attack}_alpha_sweep_methods")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-csv", default="paper_results/evaluation/all_method_evaluation.csv")
    parser.add_argument("--alpha-sweep-csv", default="paper_results/sweeps/all_alpha_sweep_methods.csv")
    parser.add_argument("--figures-dir", default="paper_results/figures")
    args = parser.parse_args()
    figures_dir = ensure_dir(args.figures_dir)
    evaluation_path = Path(args.evaluation_csv)
    if evaluation_path.exists():
        frame = pd.read_csv(evaluation_path)
        plot_method_attack_bars(frame, figures_dir)
        plot_infidelity_cdf(frame, figures_dir)
    else:
        print(f"Skipping evaluation figures; file not found: {evaluation_path}")
    sweep_path = Path(args.alpha_sweep_csv)
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        plot_alpha_sweep(sweep, figures_dir)
    else:
        print(f"Skipping alpha-sweep figures; file not found: {sweep_path}")
    print(f"Saved paper figures to {figures_dir}")


if __name__ == "__main__":
    main()
