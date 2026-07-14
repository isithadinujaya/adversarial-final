from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from qst.attacks import physical_replacement_attack
from qst.baselines import BASELINE_METHODS, estimate_density
from qst.config import load_config
from qst.data import build_datasets, build_loaders
from qst.evaluator import load_checkpoint
from qst.measurements import PauliCubeMeasurement
from qst.metrics import frobenius_distance, quantum_fidelity, trace_distance
from qst.models import build_model
from qst.utils import ensure_dir, resolve_device, seed_everything, save_json


NN_METHODS = ["clean_nn", "adversarial_nn"]
PHYSICAL_ATTACKS = ["random_replacement", "targeted_replacement", "worst_replacement"]


def _load_model(config, checkpoint, device):
    model = build_model(config).to(device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    return model


def _limited_batches(loader, max_samples: int):
    seen = 0
    for batch in loader:
        if seen >= max_samples:
            break
        remaining = max_samples - seen
        if batch["rho"].shape[0] > remaining:
            batch = {k: v[:remaining] for k, v in batch.items()}
        seen += batch["rho"].shape[0]
        yield batch


def evaluate_alpha_sweep_case(
    *,
    clean_config_path: str,
    adversarial_config_path: str,
    clean_checkpoint: str,
    adversarial_checkpoint: str,
    methods: list[str],
    attacks: list[str],
    output_dir: Path,
    max_samples: int | None,
) -> pd.DataFrame:
    config = load_config(adversarial_config_path)
    clean_config = load_config(clean_config_path)
    if max_samples is not None:
        config.evaluation.max_samples = max_samples
    seed_everything(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    datasets = build_datasets(config)
    loaders = build_loaders(config, datasets)
    measurement = PauliCubeMeasurement(config.experiment.num_qubits, device=device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(config.experiment.seed + 1201)
    clean_model = _load_model(clean_config, clean_checkpoint, device) if "clean_nn" in methods else None
    adv_model = _load_model(config, adversarial_checkpoint, device) if "adversarial_nn" in methods else None

    rows = []
    for alpha in config.evaluation.alpha_grid:
        for attack in attacks:
            for batch in tqdm(_limited_batches(loaders["test"], config.evaluation.max_samples), desc=f"q={config.experiment.num_qubits}:alpha={alpha}:{attack}"):
                rho = batch["rho"].to(device)
                result = physical_replacement_attack(
                    rho,
                    alpha=float(alpha),
                    epsilon_physical=config.attack.epsilon_physical,
                    kind=attack,
                    target_state=config.attack.target_state,
                    target_min_trace_distance=config.attack.target_min_trace_distance,
                    generator=generator,
                )
                frequencies = measurement.sample_frequencies(
                    result.attacked_state,
                    config.data.shots_per_setting,
                    generator=generator,
                )
                predictions = {}
                if clean_model is not None:
                    with torch.no_grad():
                        predictions["clean_nn"] = clean_model(frequencies)
                if adv_model is not None:
                    with torch.no_grad():
                        predictions["adversarial_nn"] = adv_model(frequencies)
                for method in methods:
                    if method in BASELINE_METHODS:
                        predictions[method] = estimate_density(
                            method,
                            frequencies,
                            measurement,
                            config,
                            shots=config.data.shots_per_setting,
                            generator=generator,
                        )
                for method, pred in predictions.items():
                    with torch.no_grad():
                        fid = quantum_fidelity(rho, pred, epsilon=config.loss.fidelity_epsilon)
                        tr = trace_distance(rho, pred)
                        fr = frobenius_distance(rho, pred)
                    for i in range(rho.shape[0]):
                        rows.append({
                            "num_qubits": config.experiment.num_qubits,
                            "method": method,
                            "attack": attack,
                            "alpha_requested": float(alpha),
                            "alpha_effective": float(result.alpha_effective[i].detach().cpu()),
                            "epsilon_physical_actual": float(result.epsilon_actual[i].detach().cpu()),
                            "fidelity": float(fid[i].detach().cpu()),
                            "infidelity": float(1.0 - fid[i].detach().cpu()),
                            "trace_distance": float(tr[i].detach().cpu()),
                            "frobenius_distance": float(fr[i].detach().cpu()),
                        })
    frame = pd.DataFrame(rows)
    ensure_dir(output_dir)
    frame.to_csv(output_dir / f"q{config.experiment.num_qubits}_alpha_sweep_methods.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-configs", nargs="+", required=True)
    parser.add_argument("--adversarial-configs", nargs="+", required=True)
    parser.add_argument("--clean-checkpoints", nargs="+", required=True)
    parser.add_argument("--adversarial-checkpoints", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=[*NN_METHODS, *BASELINE_METHODS])
    parser.add_argument("--attacks", nargs="+", default=PHYSICAL_ATTACKS)
    parser.add_argument("--output-dir", default="paper_results/sweeps")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    lengths = {len(args.clean_configs), len(args.adversarial_configs), len(args.clean_checkpoints), len(args.adversarial_checkpoints)}
    if len(lengths) != 1:
        raise ValueError("All config/checkpoint lists must have the same length.")
    output_dir = ensure_dir(args.output_dir)
    frames = []
    for clean_cfg, adv_cfg, clean_ckpt, adv_ckpt in zip(args.clean_configs, args.adversarial_configs, args.clean_checkpoints, args.adversarial_checkpoints):
        frames.append(evaluate_alpha_sweep_case(
            clean_config_path=clean_cfg,
            adversarial_config_path=adv_cfg,
            clean_checkpoint=clean_ckpt,
            adversarial_checkpoint=adv_ckpt,
            methods=args.methods,
            attacks=args.attacks,
            output_dir=output_dir,
            max_samples=args.max_samples,
        ))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "all_alpha_sweep_methods.csv", index=False)
    summary = combined.groupby(["num_qubits", "method", "attack", "alpha_requested"], as_index=False).agg(
        mean_fidelity=("fidelity", "mean"),
        std_fidelity=("fidelity", "std"),
        mean_trace_distance=("trace_distance", "mean"),
        samples=("fidelity", "size"),
    )
    summary.to_csv(output_dir / "all_alpha_sweep_summary.csv", index=False)
    save_json(output_dir / "sweep_settings.json", vars(args))
    print(f"Saved paper method alpha sweep to {output_dir}")


if __name__ == "__main__":
    main()
