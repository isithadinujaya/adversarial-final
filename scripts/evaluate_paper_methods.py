from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from qst.attacks import frequency_pgd_attack, physical_replacement_attack
from qst.baselines import BASELINE_METHODS, estimate_density
from qst.config import QSTConfig, load_config
from qst.data import build_datasets, build_loaders
from qst.evaluator import load_checkpoint
from qst.measurements import PauliCubeMeasurement
from qst.metrics import frobenius_distance, physicality_metrics, quantum_fidelity, trace_distance
from qst.models import build_model
from qst.utils import ensure_dir, resolve_device, save_json, seed_everything


NN_METHODS = ["clean_nn", "adversarial_nn"]
DEFAULT_ATTACKS = ["clean", "random_replacement", "targeted_replacement", "worst_replacement", "frequency_pgd"]
PHYSICAL_ATTACKS = {"random_replacement", "targeted_replacement", "fixed_replacement", "worst_replacement"}


def _load_nn(config: QSTConfig, checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
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
            batch = {key: value[:remaining] for key, value in batch.items()}
        seen += batch["rho"].shape[0]
        yield batch


def _make_frequencies(
    rho: torch.Tensor,
    attack: str,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
    generator: torch.Generator,
    *,
    shots: int,
    pgd_model: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    clean = measurement.sample_frequencies(rho, shots, generator=generator)
    metadata = {
        "alpha_requested": torch.zeros(rho.shape[0], device=rho.device),
        "alpha_effective": torch.zeros(rho.shape[0], device=rho.device),
        "epsilon_physical_actual": torch.zeros(rho.shape[0], device=rho.device),
    }
    if attack == "clean":
        return clean, metadata
    if attack in PHYSICAL_ATTACKS:
        result = physical_replacement_attack(
            rho,
            alpha=config.evaluation.default_alpha,
            epsilon_physical=config.attack.epsilon_physical,
            kind=attack,
            target_state=config.attack.target_state,
            target_min_trace_distance=config.attack.target_min_trace_distance,
            generator=generator,
        )
        metadata["alpha_requested"] = result.alpha_requested
        metadata["alpha_effective"] = result.alpha_effective
        metadata["epsilon_physical_actual"] = result.epsilon_actual
        return measurement.sample_frequencies(result.attacked_state, shots, generator=generator), metadata
    if attack == "frequency_pgd":
        if pgd_model is None:
            raise ValueError("frequency_pgd requires a neural network pgd_model.")
        return frequency_pgd_attack(
            pgd_model,
            clean,
            rho,
            epsilon=config.evaluation.default_epsilon_frequency,
            num_settings=config.num_settings,
            outcomes_per_setting=config.dimension,
            steps=config.attack.pgd_eval_steps,
            step_size=config.attack.pgd_step_size,
            random_start=config.attack.pgd_random_start,
            generator=generator,
        ), metadata
    raise ValueError(f"Unknown attack: {attack}")


def _append_metric_rows(
    rows: list[dict[str, float | int | str]],
    *,
    method: str,
    attack: str,
    config: QSTConfig,
    batch,
    rho: torch.Tensor,
    prediction: torch.Tensor,
    metadata: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        fidelities = quantum_fidelity(rho, prediction, epsilon=config.loss.fidelity_epsilon)
        traces = trace_distance(rho, prediction)
        frob = frobenius_distance(rho, prediction)
        physical = physicality_metrics(prediction)
    ensemble = batch["ensemble"].cpu().numpy()
    purity_values = batch["purity"].cpu().numpy()
    sample_indices = batch["sample_index"].cpu().numpy()
    for i in range(rho.shape[0]):
        rows.append(
            {
                "method": method,
                "attack": attack,
                "num_qubits": config.experiment.num_qubits,
                "dimension": config.dimension,
                "sample_index": int(sample_indices[i]),
                "ensemble": int(ensemble[i]),
                "purity": float(purity_values[i]),
                "shots": int(config.data.shots_per_setting),
                "alpha_requested": float(metadata["alpha_requested"][i].detach().cpu()),
                "alpha_effective": float(metadata["alpha_effective"][i].detach().cpu()),
                "epsilon_physical_actual": float(metadata["epsilon_physical_actual"][i].detach().cpu()),
                "epsilon_frequency": float(config.evaluation.default_epsilon_frequency if attack == "frequency_pgd" else 0.0),
                "fidelity": float(fidelities[i].detach().cpu()),
                "infidelity": float(1.0 - fidelities[i].detach().cpu()),
                "trace_distance": float(traces[i].detach().cpu()),
                "frobenius_distance": float(frob[i].detach().cpu()),
                "minimum_eigenvalue": float(physical["minimum_eigenvalue"][i].detach().cpu()),
                "trace_error": float(physical["trace_error"][i].detach().cpu()),
                "hermitian_error": float(physical["hermitian_error"][i].detach().cpu()),
            }
        )


def evaluate_case(
    *,
    clean_config_path: str,
    adversarial_config_path: str,
    clean_checkpoint: str,
    adversarial_checkpoint: str,
    methods: list[str],
    attacks: list[str],
    output_dir: Path,
    include_frequency_pgd_for_baselines: bool,
    max_samples: int | None,
) -> pd.DataFrame:
    config = load_config(adversarial_config_path)
    clean_config = load_config(clean_config_path)
    seed_everything(config.experiment.seed)
    if max_samples is not None:
        config.evaluation.max_samples = max_samples
        clean_config.evaluation.max_samples = max_samples
    device = resolve_device(config.experiment.device)
    datasets = build_datasets(config)
    loaders = build_loaders(config, datasets)
    measurement = PauliCubeMeasurement(config.experiment.num_qubits, device=device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(config.experiment.seed + 909)

    clean_model = _load_nn(clean_config, clean_checkpoint, device) if "clean_nn" in methods else None
    adversarial_model = _load_nn(config, adversarial_checkpoint, device) if "adversarial_nn" in methods or "frequency_pgd" in attacks else None

    rows: list[dict[str, float | int | str]] = []
    for attack in attacks:
        for batch in tqdm(_limited_batches(loaders["test"], config.evaluation.max_samples), desc=f"q={config.experiment.num_qubits}:{attack}"):
            rho = batch["rho"].to(device)
            frequencies, metadata = _make_frequencies(
                rho,
                attack,
                measurement,
                config,
                generator,
                shots=config.data.shots_per_setting,
                pgd_model=adversarial_model,
            )
            if "clean_nn" in methods and clean_model is not None:
                with torch.no_grad():
                    prediction = clean_model(frequencies)
                _append_metric_rows(rows, method="clean_nn", attack=attack, config=config, batch=batch, rho=rho, prediction=prediction, metadata=metadata)
            if "adversarial_nn" in methods and adversarial_model is not None:
                with torch.no_grad():
                    prediction = adversarial_model(frequencies)
                _append_metric_rows(rows, method="adversarial_nn", attack=attack, config=config, batch=batch, rho=rho, prediction=prediction, metadata=metadata)

            for method in methods:
                if method not in BASELINE_METHODS:
                    continue
                if attack == "frequency_pgd" and not include_frequency_pgd_for_baselines:
                    continue
                prediction = estimate_density(
                    method,
                    frequencies,
                    measurement,
                    config,
                    shots=config.data.shots_per_setting,
                    generator=generator,
                )
                _append_metric_rows(rows, method=method, attack=attack, config=config, batch=batch, rho=rho, prediction=prediction, metadata=metadata)

    frame = pd.DataFrame(rows)
    ensure_dir(output_dir)
    csv_path = output_dir / f"q{config.experiment.num_qubits}_method_evaluation.csv"
    frame.to_csv(csv_path, index=False)
    summary = (
        frame.groupby(["num_qubits", "method", "attack"], as_index=False)
        .agg(
            mean_fidelity=("fidelity", "mean"),
            std_fidelity=("fidelity", "std"),
            mean_infidelity=("infidelity", "mean"),
            mean_trace_distance=("trace_distance", "mean"),
            mean_frobenius_distance=("frobenius_distance", "mean"),
            samples=("fidelity", "size"),
        )
    )
    summary.to_csv(output_dir / f"q{config.experiment.num_qubits}_method_summary.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-configs", nargs="+", required=True)
    parser.add_argument("--adversarial-configs", nargs="+", required=True)
    parser.add_argument("--clean-checkpoints", nargs="+", required=True)
    parser.add_argument("--adversarial-checkpoints", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", default=[*NN_METHODS, *BASELINE_METHODS])
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--output-dir", default="paper_results/evaluation")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-frequency-pgd-for-baselines", action="store_true")
    args = parser.parse_args()

    lengths = {len(args.clean_configs), len(args.adversarial_configs), len(args.clean_checkpoints), len(args.adversarial_checkpoints)}
    if len(lengths) != 1:
        raise ValueError("clean/adversarial configs and checkpoints must have the same length.")

    output_dir = ensure_dir(args.output_dir)
    frames = []
    for clean_cfg, adv_cfg, clean_ckpt, adv_ckpt in zip(
        args.clean_configs,
        args.adversarial_configs,
        args.clean_checkpoints,
        args.adversarial_checkpoints,
    ):
        frames.append(
            evaluate_case(
                clean_config_path=clean_cfg,
                adversarial_config_path=adv_cfg,
                clean_checkpoint=clean_ckpt,
                adversarial_checkpoint=adv_ckpt,
                methods=args.methods,
                attacks=args.attacks,
                output_dir=output_dir,
                include_frequency_pgd_for_baselines=args.include_frequency_pgd_for_baselines,
                max_samples=args.max_samples,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "all_method_evaluation.csv", index=False)
    summary = (
        combined.groupby(["num_qubits", "method", "attack"], as_index=False)
        .agg(
            mean_fidelity=("fidelity", "mean"),
            std_fidelity=("fidelity", "std"),
            mean_infidelity=("infidelity", "mean"),
            mean_trace_distance=("trace_distance", "mean"),
            mean_frobenius_distance=("frobenius_distance", "mean"),
            samples=("fidelity", "size"),
        )
    )
    summary.to_csv(output_dir / "all_method_summary.csv", index=False)
    save_json(output_dir / "evaluation_settings.json", vars(args))
    print(f"Saved paper method evaluation to {output_dir}")


if __name__ == "__main__":
    main()
