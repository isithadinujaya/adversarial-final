from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from qst.attacks import frequency_pgd_attack, physical_replacement_attack
from qst.config import QSTConfig
from qst.losses import LossOutput, RobustTomographyLoss, _squared_frobenius_per_sample
from qst.measurements import PauliCubeMeasurement
from qst.metrics import quantum_fidelity
from qst.utils import ensure_dir, save_json


@dataclass
class EpochResult:
    total: float
    clean: float
    adversarial: float
    physical: float
    pgd: float
    consistency: float
    physical_consistency: float
    pgd_consistency: float
    clean_fidelity: float
    physical_fidelity: float
    pgd_fidelity: float
    adversarial_fidelity: float


class RobustQSTTrainer:
    """Train either clean-only or staged adversarial MLP tomography models.

    strategy=clean_only
        Trains only on clean finite-shot measurement frequencies. This is the baseline NN.

    strategy=staged_adversarial
        Stage 1: clean reconstruction warm-up.
        Stage 2: balanced adversarial warm-up; each state receives one attack, and
                 every mini-batch contains all attack families with reduced budgets.
        Stage 3: full hierarchical robust training with all physical attacks plus PGD.

    strategy=direct_adversarial
        Skips stages 1 and 2 and starts immediately with the Stage-3 objective.
    """

    def __init__(
        self,
        config: QSTConfig,
        model: nn.Module,
        measurement: PauliCubeMeasurement,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.measurement = measurement.to(device)
        self.device = device
        self.output_dir = ensure_dir(config.experiment.output_dir)
        self.loss_function = RobustTomographyLoss(
            clean_weight=config.loss.clean_weight,
            physical_weight=config.loss.physical_weight,
            pgd_weight=config.loss.pgd_weight,
            consistency_weight=config.loss.consistency_weight,
            physical_max_weight=config.loss.physical_max_weight,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.min_learning_rate,
        )
        self.train_generator = torch.Generator(device=device.type)
        self.train_generator.manual_seed(config.experiment.seed + 404)
        self.validation_seed = config.experiment.seed + 505
        self.history_path = self.output_dir / "history.csv"

    def _training_stage(self, epoch: int) -> str:
        if self.config.training.strategy == "clean_only":
            return "clean"
        if self.config.training.strategy == "direct_adversarial":
            return "hierarchical"
        clean_end = self.config.training.clean_warmup_epochs
        balanced_end = clean_end + self.config.training.balanced_warmup_epochs
        if epoch <= clean_end:
            return "clean"
        if epoch <= balanced_end:
            return "balanced"
        return "hierarchical"

    def _early_stopping_enabled(self, epoch: int) -> bool:
        if self.config.training.strategy != "staged_adversarial":
            return True
        if self.config.training.enable_early_stopping_after_stage <= 1:
            return True
        if self.config.training.enable_early_stopping_after_stage == 2:
            return epoch > self.config.training.clean_warmup_epochs
        return epoch > (
            self.config.training.clean_warmup_epochs
            + self.config.training.balanced_warmup_epochs
        )

    def _uniform_batch(
        self,
        minimum: float,
        maximum: float,
        batch_size: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if minimum == maximum:
            return torch.full(
                (batch_size,), minimum, device=self.device, dtype=torch.float32
            )
        values = torch.rand(
            batch_size, generator=generator, device=self.device, dtype=torch.float32
        )
        return minimum + (maximum - minimum) * values

    def _make_physical_frequencies(
        self,
        rho: torch.Tensor,
        attack_kind: str,
        *,
        generator: torch.Generator,
        alpha_max: float | None = None,
    ) -> torch.Tensor:
        maximum = self.config.attack.alpha_max if alpha_max is None else alpha_max
        alpha = self._uniform_batch(
            self.config.attack.alpha_min,
            maximum,
            rho.shape[0],
            generator=generator,
        )
        result = physical_replacement_attack(
            rho,
            alpha=alpha,
            epsilon_physical=self.config.attack.epsilon_physical,
            kind=attack_kind,
            target_state=self.config.attack.target_state,
            target_min_trace_distance=self.config.attack.target_min_trace_distance,
            generator=generator,
        )
        return self.measurement.sample_frequencies(
            result.attacked_state,
            self.config.data.shots_per_setting,
            generator=generator,
        )

    def _make_pgd_frequencies(
        self,
        rho: torch.Tensor,
        clean_frequencies: torch.Tensor,
        *,
        training: bool,
        generator: torch.Generator,
        epsilon_max: float | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        maximum = (
            self.config.attack.epsilon_frequency_max
            if epsilon_max is None
            else epsilon_max
        )
        epsilon = self._uniform_batch(
            self.config.attack.epsilon_frequency_min,
            maximum,
            rho.shape[0],
            generator=generator,
        )
        if steps is None:
            steps = (
                self.config.attack.pgd_train_steps
                if training
                else self.config.attack.pgd_eval_steps
            )
        return frequency_pgd_attack(
            self.model,
            clean_frequencies,
            rho,
            epsilon=epsilon,
            num_settings=self.config.num_settings,
            outcomes_per_setting=self.config.dimension,
            steps=steps,
            step_size=self.config.attack.pgd_step_size,
            random_start=self.config.attack.pgd_random_start,
            generator=generator,
        )

    def _zero_loss_output(self, clean_loss: torch.Tensor) -> LossOutput:
        zero = clean_loss.new_tensor(0.0)
        return LossOutput(
            total=self.config.loss.clean_weight * clean_loss,
            clean=clean_loss,
            adversarial=zero,
            physical=zero,
            pgd=zero,
            consistency=zero,
            physical_consistency=zero,
            pgd_consistency=zero,
        )

    def _balanced_loss_output(
        self,
        target_rho: torch.Tensor,
        clean_prediction: torch.Tensor,
        adversarial_prediction: torch.Tensor,
        family_ids: torch.Tensor,
    ) -> LossOutput:
        """Stage-2 loss for one balanced attack per state.

        family_ids: 0 for physical attacks, 1 for PGD.
        """
        clean_per = _squared_frobenius_per_sample(target_rho, clean_prediction)
        adv_per = _squared_frobenius_per_sample(target_rho, adversarial_prediction)
        cons_per = _squared_frobenius_per_sample(
            clean_prediction.detach(), adversarial_prediction
        )
        clean = clean_per.mean()
        physical_mask = family_ids == 0
        pgd_mask = family_ids == 1
        physical = adv_per[physical_mask].mean() if physical_mask.any() else clean.new_tensor(0.0)
        pgd = adv_per[pgd_mask].mean() if pgd_mask.any() else clean.new_tensor(0.0)
        physical_cons = cons_per[physical_mask].mean() if physical_mask.any() else clean.new_tensor(0.0)
        pgd_cons = cons_per[pgd_mask].mean() if pgd_mask.any() else clean.new_tensor(0.0)
        adversarial = self.config.loss.physical_weight * physical + self.config.loss.pgd_weight * pgd
        consistency = self.config.loss.physical_weight * physical_cons + self.config.loss.pgd_weight * pgd_cons
        total = self.config.loss.clean_weight * clean + adversarial + self.config.loss.consistency_weight * consistency
        return LossOutput(
            total=total,
            clean=clean,
            adversarial=adversarial,
            physical=physical,
            pgd=pgd,
            consistency=consistency,
            physical_consistency=physical_cons,
            pgd_consistency=pgd_cons,
        )

    def _make_balanced_adversarial_frequencies(
        self,
        rho: torch.Tensor,
        clean_frequencies: torch.Tensor,
        *,
        epoch: int,
        training: bool,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attacks = [*self.config.attack.physical_training_types, "frequency_pgd"]
        batch_size = rho.shape[0]
        assignments = (torch.arange(batch_size, device=self.device) + epoch) % len(attacks)
        family_ids = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        adversarial = clean_frequencies.detach().clone()
        for index, attack_kind in enumerate(attacks):
            mask = assignments == index
            if not mask.any():
                continue
            if attack_kind == "frequency_pgd":
                family_ids[mask] = 1
                adversarial[mask] = self._make_pgd_frequencies(
                    rho[mask],
                    clean_frequencies[mask],
                    training=training,
                    generator=generator,
                    epsilon_max=self.config.training.warmup_epsilon_frequency_max,
                    steps=self.config.training.warmup_pgd_steps,
                )
            else:
                family_ids[mask] = 0
                adversarial[mask] = self._make_physical_frequencies(
                    rho[mask],
                    attack_kind,
                    generator=generator,
                    alpha_max=self.config.training.warmup_alpha_max,
                )
        return adversarial, family_ids

    def _run_epoch(
        self,
        loader: DataLoader,
        *,
        training: bool,
        epoch: int,
    ) -> EpochResult:
        stage = self._training_stage(epoch)
        self.model.train(training)
        totals = {
            "total": 0.0,
            "clean": 0.0,
            "adversarial": 0.0,
            "physical": 0.0,
            "pgd": 0.0,
            "consistency": 0.0,
            "physical_consistency": 0.0,
            "pgd_consistency": 0.0,
            "clean_fidelity": 0.0,
            "physical_fidelity": 0.0,
            "pgd_fidelity": 0.0,
            "adversarial_fidelity": 0.0,
        }
        examples = 0

        if training:
            generator = self.train_generator
        else:
            generator = torch.Generator(device=self.device.type)
            generator.manual_seed(self.validation_seed + epoch)

        iterator = tqdm(loader, leave=False, desc=f"{'train' if training else 'val'}:{stage}")
        for batch in iterator:
            rho = batch["rho"].to(self.device)
            batch_size = rho.shape[0]
            clean_frequencies = self.measurement.sample_frequencies(
                rho,
                self.config.data.shots_per_setting,
                generator=generator,
            )

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            if stage == "clean":
                clean_prediction = self.model(clean_frequencies)
                clean_loss = _squared_frobenius_per_sample(rho, clean_prediction).mean()
                losses = self._zero_loss_output(clean_loss)
                physical_fidelity = clean_prediction.new_tensor(float("nan"))
                pgd_fidelity = clean_prediction.new_tensor(float("nan"))
                adversarial_fidelity = clean_prediction.new_tensor(float("nan"))
            elif stage == "balanced":
                adversarial_frequencies, family_ids = self._make_balanced_adversarial_frequencies(
                    rho,
                    clean_frequencies,
                    epoch=epoch,
                    training=training,
                    generator=generator,
                )
                clean_prediction = self.model(clean_frequencies)
                adversarial_prediction = self.model(adversarial_frequencies)
                losses = self._balanced_loss_output(
                    rho,
                    clean_prediction,
                    adversarial_prediction,
                    family_ids,
                )
                with torch.no_grad():
                    adv_fid_per = quantum_fidelity(
                        rho,
                        adversarial_prediction,
                        epsilon=self.config.loss.fidelity_epsilon,
                    )
                    physical_mask = family_ids == 0
                    pgd_mask = family_ids == 1
                    physical_fidelity = adv_fid_per[physical_mask].mean() if physical_mask.any() else clean_prediction.new_tensor(float("nan"))
                    pgd_fidelity = adv_fid_per[pgd_mask].mean() if pgd_mask.any() else clean_prediction.new_tensor(float("nan"))
                    adversarial_fidelity = adv_fid_per.mean()
            else:  # hierarchical Stage 3
                physical_frequency_sets = [
                    self._make_physical_frequencies(
                        rho,
                        attack_kind,
                        generator=generator,
                    )
                    for attack_kind in self.config.attack.physical_training_types
                ]
                pgd_frequencies = self._make_pgd_frequencies(
                    rho,
                    clean_frequencies,
                    training=training,
                    generator=generator,
                )
                clean_prediction = self.model(clean_frequencies)
                physical_prediction_list = [
                    self.model(frequencies) for frequencies in physical_frequency_sets
                ]
                physical_predictions = torch.stack(physical_prediction_list, dim=1)
                pgd_prediction = self.model(pgd_frequencies)
                losses = self.loss_function(
                    rho,
                    clean_prediction,
                    physical_predictions,
                    pgd_prediction,
                )
                with torch.no_grad():
                    physical_fidelities = torch.stack(
                        [
                            quantum_fidelity(
                                rho,
                                prediction,
                                epsilon=self.config.loss.fidelity_epsilon,
                            )
                            for prediction in physical_prediction_list
                        ],
                        dim=1,
                    )
                    physical_fidelity = physical_fidelities.min(dim=1).values.mean()
                    pgd_fidelity = quantum_fidelity(
                        rho,
                        pgd_prediction,
                        epsilon=self.config.loss.fidelity_epsilon,
                    ).mean()
                    adversarial_fidelity = (
                        self.config.loss.physical_weight * physical_fidelity
                        + self.config.loss.pgd_weight * pgd_fidelity
                    )

            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
                self.optimizer.step()

            with torch.no_grad():
                clean_fidelity = quantum_fidelity(
                    rho,
                    clean_prediction,
                    epsilon=self.config.loss.fidelity_epsilon,
                ).mean()
                values = losses.detached_dict()
                for key in (
                    "total",
                    "clean",
                    "adversarial",
                    "physical",
                    "pgd",
                    "consistency",
                    "physical_consistency",
                    "pgd_consistency",
                ):
                    totals[key] += values[key] * batch_size
                totals["clean_fidelity"] += float(clean_fidelity.detach().cpu()) * batch_size
                for key, tensor_value in [
                    ("physical_fidelity", physical_fidelity),
                    ("pgd_fidelity", pgd_fidelity),
                    ("adversarial_fidelity", adversarial_fidelity),
                ]:
                    value = float(tensor_value.detach().cpu())
                    if math.isnan(value):
                        value = 0.0
                    totals[key] += value * batch_size
                examples += batch_size
            iterator.set_postfix(total=f"{totals['total'] / examples:.4f}")

        return EpochResult(**{key: value / examples for key, value in totals.items()})

    def _save_checkpoint(self, path: Path, epoch: int, best_validation: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_validation": best_validation,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.as_dict(),
            },
            path,
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        save_json(self.output_dir / "resolved_config.json", self.config.as_dict())
        best_validation = math.inf
        patience = 0
        history_rows: list[dict[str, float | int | str]] = []
        start_time = time.time()

        for epoch in range(1, self.config.training.epochs + 1):
            stage = self._training_stage(epoch)
            train_result = self._run_epoch(train_loader, training=True, epoch=epoch)
            validation_result = self._run_epoch(val_loader, training=False, epoch=epoch)
            if self._early_stopping_enabled(epoch):
                self.scheduler.step(validation_result.total)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "stage": stage,
                "learning_rate": learning_rate,
                **{f"train_{k}": v for k, v in train_result.__dict__.items()},
                **{f"val_{k}": v for k, v in validation_result.__dict__.items()},
            }
            history_rows.append(row)
            with self.history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(history_rows)

            self._save_checkpoint(self.output_dir / "last.pt", epoch, best_validation)
            if validation_result.total < best_validation:
                best_validation = validation_result.total
                patience = 0
                self._save_checkpoint(self.output_dir / "best.pt", epoch, best_validation)
            elif self._early_stopping_enabled(epoch):
                patience += 1

            if epoch % self.config.training.log_every == 0:
                print(
                    f"Epoch {epoch:03d} [{stage}] | "
                    f"train={train_result.total:.6f} | "
                    f"val={validation_result.total:.6f} | "
                    f"clean F={validation_result.clean_fidelity:.6f} | "
                    f"physical F={validation_result.physical_fidelity:.6f} | "
                    f"PGD F={validation_result.pgd_fidelity:.6f} | "
                    f"lr={learning_rate:.2e}"
                )
            if self._early_stopping_enabled(epoch) and patience >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        summary = {
            "best_validation_total": best_validation,
            "epochs_completed": len(history_rows),
            "elapsed_seconds": time.time() - start_time,
            "best_checkpoint": str(self.output_dir / "best.pt"),
        }
        save_json(self.output_dir / "training_summary.json", summary)
        return summary
