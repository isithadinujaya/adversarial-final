from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CLEAN_CONFIGS = [
    "configs/paper/one_qubit_clean.yaml",
    "configs/paper/two_qubit_clean.yaml",
    "configs/paper/three_qubit_clean.yaml",
]
ADV_CONFIGS = [
    "configs/paper/one_qubit_adversarial.yaml",
    "configs/paper/two_qubit_adversarial.yaml",
    "configs/paper/three_qubit_adversarial.yaml",
]
CLEAN_CHECKPOINTS = [
    "outputs/paper/one_qubit_clean_nn_100k/best.pt",
    "outputs/paper/two_qubit_clean_nn_100k/best.pt",
    "outputs/paper/three_qubit_clean_nn_100k/best.pt",
]
ADV_CHECKPOINTS = [
    "outputs/paper/one_qubit_adv_nn_100k/best.pt",
    "outputs/paper/two_qubit_adv_nn_100k/best.pt",
    "outputs/paper/three_qubit_adv_nn_100k/best.pt",
]


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-three-qubit", action="store_true")
    args = parser.parse_args()

    clean_configs = CLEAN_CONFIGS[:2] if args.skip_three_qubit else CLEAN_CONFIGS
    adv_configs = ADV_CONFIGS[:2] if args.skip_three_qubit else ADV_CONFIGS
    clean_checkpoints = CLEAN_CHECKPOINTS[:2] if args.skip_three_qubit else CLEAN_CHECKPOINTS
    adv_checkpoints = ADV_CHECKPOINTS[:2] if args.skip_three_qubit else ADV_CHECKPOINTS

    if args.train:
        for config in clean_configs + adv_configs:
            run([sys.executable, "-m", "scripts.train", "--config", config])

    sample_args = [] if args.max_samples is None else ["--max-samples", str(args.max_samples)]
    if args.evaluate:
        run([
            sys.executable,
            "-m",
            "scripts.evaluate_paper_methods",
            "--clean-configs",
            *clean_configs,
            "--adversarial-configs",
            *adv_configs,
            "--clean-checkpoints",
            *clean_checkpoints,
            "--adversarial-checkpoints",
            *adv_checkpoints,
            "--output-dir",
            "paper_results/evaluation",
            *sample_args,
        ])

    if args.sweep:
        run([
            sys.executable,
            "-m",
            "scripts.sweep_paper_methods",
            "--clean-configs",
            *clean_configs,
            "--adversarial-configs",
            *adv_configs,
            "--clean-checkpoints",
            *clean_checkpoints,
            "--adversarial-checkpoints",
            *adv_checkpoints,
            "--output-dir",
            "paper_results/sweeps",
            *sample_args,
        ])

    if args.figures:
        run([
            sys.executable,
            "-m",
            "scripts.make_paper_figures",
            "--evaluation-csv",
            "paper_results/evaluation/all_method_evaluation.csv",
            "--alpha-sweep-csv",
            "paper_results/sweeps/all_alpha_sweep_methods.csv",
            "--figures-dir",
            "paper_results/figures",
        ])
        run([
            sys.executable,
            "-m",
            "scripts.make_method_figures",
            "--figures-dir",
            "figures/method",
        ])


if __name__ == "__main__":
    main()
