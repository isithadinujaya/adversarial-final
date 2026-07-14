from __future__ import annotations

from dataclasses import dataclass

import torch

from qst.config import QSTConfig
from qst.measurements import PauliCubeMeasurement
from qst.models.head import raw_cholesky_to_density
from qst.states import StateMixture, sample_density_matrices


BASELINE_METHODS = [
    "linear_inversion",
    "mle",
    "bayesian_particle",
    "compressed_sensing",
    "purification_mle",
]


@dataclass
class EstimatorDiagnostics:
    method: str
    iterations: int


def project_psd_trace(rho: torch.Tensor, *, epsilon: float = 1.0e-12) -> torch.Tensor:
    """Project Hermitian matrices onto PSD trace-one density matrices."""
    dimension = rho.shape[-1]
    rho = (rho + rho.conj().transpose(-1, -2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = eigenvalues.real.clamp_min(0.0)
    trace = eigenvalues.sum(dim=-1, keepdim=True)
    fallback = trace <= epsilon
    normalized = eigenvalues / trace.clamp_min(epsilon)
    if fallback.any():
        uniform = torch.full_like(normalized, 1.0 / dimension)
        normalized = torch.where(fallback, uniform, normalized)
    return (eigenvectors * normalized.unsqueeze(-2)) @ eigenvectors.conj().transpose(-1, -2)


def _measurement_matrix(measurement: PauliCubeMeasurement) -> torch.Tensor:
    projectors = measurement.projectors.reshape(
        measurement.input_dimension,
        measurement.dimension,
        measurement.dimension,
    )
    return projectors.transpose(-1, -2).reshape(measurement.input_dimension, -1)


def linear_inversion_estimator(
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
) -> torch.Tensor:
    """Least-squares inversion followed by PSD trace-one projection."""
    matrix = _measurement_matrix(measurement).to(frequencies.device)
    pseudo_inverse = torch.linalg.pinv(matrix)
    rho_vector = frequencies.to(matrix.dtype) @ pseudo_inverse.transpose(0, 1)
    rho = rho_vector.reshape(frequencies.shape[0], measurement.dimension, measurement.dimension)
    return project_psd_trace(rho)


def _nll_from_frequencies(
    rho: torch.Tensor,
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    *,
    shots: int,
) -> torch.Tensor:
    probabilities = measurement.probabilities(rho).clamp_min(1.0e-9)
    target = measurement.reshape_frequency_vector(frequencies)
    return -(float(shots) * target * probabilities.log()).sum(dim=(-1, -2)).mean()


def mle_estimator(
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
    *,
    shots: int,
) -> torch.Tensor:
    """Batched maximum-likelihood tomography with a full Cholesky parameterization."""
    batch_size = frequencies.shape[0]
    raw = torch.zeros(
        batch_size,
        measurement.dimension**2,
        device=frequencies.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([raw], lr=config.baselines.mle_learning_rate)
    for _ in range(config.baselines.mle_steps):
        optimizer.zero_grad(set_to_none=True)
        rho = raw_cholesky_to_density(raw, measurement.dimension)
        loss = _nll_from_frequencies(rho, frequencies, measurement, shots=shots)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return raw_cholesky_to_density(raw, measurement.dimension)


def _mixture(config: QSTConfig) -> StateMixture:
    return StateMixture(
        pure_fraction=config.data.pure_fraction,
        mixed_fraction=config.data.mixed_fraction,
        depolarized_fraction=config.data.depolarized_fraction,
        visibility_min=config.data.depolarized_visibility_min,
        visibility_max=config.data.depolarized_visibility_max,
        ginibre_rank=config.data.ginibre_rank,
    )


def bayesian_particle_estimator(
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
    *,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Particle posterior-mean Bayesian estimator.

    A finite prior sample is drawn from the same Haar/Ginibre/depolarized family used
    for data generation. The posterior mean is an importance-weighted mixture using
    the multinomial likelihood of the observed frequencies.
    """
    particles_total = config.baselines.bayesian_particles
    chunk = config.baselines.bayesian_particle_chunk
    all_particles: list[torch.Tensor] = []
    all_log_likelihoods: list[torch.Tensor] = []
    frequency_blocks = measurement.reshape_frequency_vector(frequencies)
    mixture = _mixture(config)
    remaining = particles_total
    while remaining > 0:
        current = min(chunk, remaining)
        particles, _ = sample_density_matrices(
            current,
            measurement.dimension,
            mixture,
            device=frequencies.device,
            dtype=measurement.projectors.dtype,
            generator=generator,
        )
        probabilities = measurement.probabilities(particles).clamp_min(1.0e-9)
        log_likelihood = (
            float(shots)
            * frequency_blocks[:, None, :, :]
            * probabilities[None, :, :, :].log()
        ).sum(dim=(-1, -2))
        all_particles.append(particles)
        all_log_likelihoods.append(log_likelihood)
        remaining -= current
    particles = torch.cat(all_particles, dim=0)
    log_likelihoods = torch.cat(all_log_likelihoods, dim=1)
    weights = torch.softmax(log_likelihoods - log_likelihoods.max(dim=1, keepdim=True).values, dim=1)
    rho = torch.einsum("bp,pij->bij", weights.to(particles.dtype), particles)
    return project_psd_trace(rho)


def _eigen_shrink_project(rho: torch.Tensor, shrinkage: float) -> torch.Tensor:
    rho = (rho + rho.conj().transpose(-1, -2)) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    eigenvalues = (eigenvalues.real - shrinkage).clamp_min(0.0)
    trace = eigenvalues.sum(dim=-1, keepdim=True)
    fallback = trace <= 1.0e-12
    normalized = eigenvalues / trace.clamp_min(1.0e-12)
    if fallback.any():
        dimension = rho.shape[-1]
        normalized = torch.where(
            fallback,
            torch.full_like(normalized, 1.0 / dimension),
            normalized,
        )
    return (eigenvectors * normalized.unsqueeze(-2)) @ eigenvectors.conj().transpose(-1, -2)


def compressed_sensing_estimator(
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
) -> torch.Tensor:
    """Low-rank projected-gradient tomography with eigenvalue shrinkage.

    This is a practical compressed-sensing style baseline: it minimizes measurement
    residuals while repeatedly applying PSD trace-one projection and eigenvalue
    soft-thresholding to promote low rank.
    """
    rho = linear_inversion_estimator(frequencies, measurement).detach()
    target = measurement.reshape_frequency_vector(frequencies)
    lr = config.baselines.compressed_sensing_learning_rate
    for _ in range(config.baselines.compressed_sensing_steps):
        variable = rho.detach().clone().requires_grad_(True)
        predicted = measurement.probabilities(variable)
        loss = (predicted - target).square().mean()
        gradient = torch.autograd.grad(loss, variable)[0]
        with torch.no_grad():
            rho = variable - lr * gradient
            rho = _eigen_shrink_project(
                rho,
                config.baselines.compressed_sensing_shrinkage,
            )
    return rho.detach()


def _factor_to_density(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    factor = torch.complex(real, imag)
    rho = factor @ factor.conj().transpose(-1, -2)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).real.sum(dim=-1)
    return rho / trace[:, None, None].clamp_min(1.0e-12)


def purification_mle_estimator(
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
    *,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Likelihood optimization over a low-rank purification factor AA^dagger."""
    batch_size = frequencies.shape[0]
    dimension = measurement.dimension
    rank = config.baselines.purification_rank or max(1, dimension // 2)
    init = torch.randn(
        batch_size,
        dimension,
        rank,
        device=frequencies.device,
        dtype=torch.float32,
        generator=generator,
    ) / (dimension * rank) ** 0.5
    real = torch.nn.Parameter(init.clone())
    imag = torch.nn.Parameter(torch.zeros_like(init))
    optimizer = torch.optim.Adam([real, imag], lr=config.baselines.purification_learning_rate)
    for _ in range(config.baselines.purification_steps):
        optimizer.zero_grad(set_to_none=True)
        rho = _factor_to_density(real, imag)
        loss = _nll_from_frequencies(rho, frequencies, measurement, shots=shots)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return _factor_to_density(real, imag)


def estimate_density(
    method: str,
    frequencies: torch.Tensor,
    measurement: PauliCubeMeasurement,
    config: QSTConfig,
    *,
    shots: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    method = method.lower()
    if method == "linear_inversion":
        return linear_inversion_estimator(frequencies, measurement)
    if method == "mle":
        return mle_estimator(frequencies, measurement, config, shots=shots)
    if method == "bayesian_particle":
        return bayesian_particle_estimator(
            frequencies, measurement, config, shots=shots, generator=generator
        )
    if method == "compressed_sensing":
        return compressed_sensing_estimator(frequencies, measurement, config)
    if method == "purification_mle":
        return purification_mle_estimator(
            frequencies, measurement, config, shots=shots, generator=generator
        )
    raise ValueError(f"Unknown baseline method: {method}")
