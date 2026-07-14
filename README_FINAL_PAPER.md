# Final paper code: adversarial quantum state tomography

This version contains the full paper pipeline requested:

1. **Clean-only neural QST model**: trains an MLP only on clean finite-shot Pauli frequencies, then evaluates robustness under corrupted measurements.
2. **Adversarially trained neural QST model**: trains the same MLP using a three-stage schedule:
   - Stage 1: clean warm-up.
   - Stage 2: balanced adversarial warm-up, where every mini-batch contains physical and PGD attacks.
   - Stage 3: hierarchical robust training with separate physical-family and frequency-PGD losses.
3. **Classical/non-neural baselines** under physical attacks:
   - Linear inversion with PSD trace-one projection.
   - Maximum-likelihood estimation (MLE).
   - Particle Bayesian posterior mean.
   - Compressed-sensing-style low-rank projected tomography.
   - Purification-factor MLE tomography.

## Main mathematical loss for adversarial training

The final adversarial training loss is

\[
L_{\rm total}
= L_{\rm clean}
+ 0.5 L_{\rm physical}
+ 0.5 L_{\rm PGD}
+ 0.1 L_{\rm consistency}.
\]

The physical-family loss is

\[
L_{\rm physical}=0.7L_{\rm phys,max}+0.3L_{\rm phys,avg}.
\]

All reconstruction and consistency terms use squared Frobenius distance.

## Important convention

Frequency-space PGD is a neural-network white-box attack. Classical baselines are evaluated under physical replacement attacks by default. The evaluation script can optionally feed PGD-corrupted frequencies to classical baselines using `--include-frequency-pgd-for-baselines`, but this is not the main fair classical comparison.

## Quick smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q

python -m scripts.train --config configs/paper/smoke_clean.yaml
python -m scripts.train --config configs/paper/smoke_adversarial.yaml

python -m scripts.evaluate_paper_methods \
  --clean-configs configs/paper/smoke_clean.yaml \
  --adversarial-configs configs/paper/smoke_adversarial.yaml \
  --clean-checkpoints outputs/smoke_clean_nn/best.pt \
  --adversarial-checkpoints outputs/smoke_adversarial_nn/best.pt \
  --methods clean_nn adversarial_nn linear_inversion \
  --attacks clean random_replacement \
  --max-samples 2 \
  --output-dir paper_results/smoke_eval

python -m scripts.make_paper_figures \
  --evaluation-csv paper_results/smoke_eval/all_method_evaluation.csv \
  --figures-dir paper_results/smoke_figures
```

## Full paper training

Train clean-only NN models:

```bash
python -m scripts.train --config configs/paper/one_qubit_clean.yaml
python -m scripts.train --config configs/paper/two_qubit_clean.yaml
python -m scripts.train --config configs/paper/three_qubit_clean.yaml
```

Train adversarially trained NN models:

```bash
python -m scripts.train --config configs/paper/one_qubit_adversarial.yaml
python -m scripts.train --config configs/paper/two_qubit_adversarial.yaml
python -m scripts.train --config configs/paper/three_qubit_adversarial.yaml
```

The expected checkpoints are:

```text
outputs/paper/one_qubit_clean_nn_100k/best.pt
outputs/paper/two_qubit_clean_nn_100k/best.pt
outputs/paper/three_qubit_clean_nn_100k/best.pt

outputs/paper/one_qubit_adv_nn_100k/best.pt
outputs/paper/two_qubit_adv_nn_100k/best.pt
outputs/paper/three_qubit_adv_nn_100k/best.pt
```

## Evaluate all models and classical baselines

Start with a small run:

```bash
python -m scripts.evaluate_paper_methods \
  --clean-configs configs/paper/one_qubit_clean.yaml configs/paper/two_qubit_clean.yaml configs/paper/three_qubit_clean.yaml \
  --adversarial-configs configs/paper/one_qubit_adversarial.yaml configs/paper/two_qubit_adversarial.yaml configs/paper/three_qubit_adversarial.yaml \
  --clean-checkpoints outputs/paper/one_qubit_clean_nn_100k/best.pt outputs/paper/two_qubit_clean_nn_100k/best.pt outputs/paper/three_qubit_clean_nn_100k/best.pt \
  --adversarial-checkpoints outputs/paper/one_qubit_adv_nn_100k/best.pt outputs/paper/two_qubit_adv_nn_100k/best.pt outputs/paper/three_qubit_adv_nn_100k/best.pt \
  --max-samples 100 \
  --output-dir paper_results/evaluation
```

Then run the final full evaluation by removing `--max-samples 100`.

## Alpha sweeps for method comparison

```bash
python -m scripts.sweep_paper_methods \
  --clean-configs configs/paper/one_qubit_clean.yaml configs/paper/two_qubit_clean.yaml configs/paper/three_qubit_clean.yaml \
  --adversarial-configs configs/paper/one_qubit_adversarial.yaml configs/paper/two_qubit_adversarial.yaml configs/paper/three_qubit_adversarial.yaml \
  --clean-checkpoints outputs/paper/one_qubit_clean_nn_100k/best.pt outputs/paper/two_qubit_clean_nn_100k/best.pt outputs/paper/three_qubit_clean_nn_100k/best.pt \
  --adversarial-checkpoints outputs/paper/one_qubit_adv_nn_100k/best.pt outputs/paper/two_qubit_adv_nn_100k/best.pt outputs/paper/three_qubit_adv_nn_100k/best.pt \
  --max-samples 100 \
  --output-dir paper_results/sweeps
```

Remove `--max-samples 100` for final figures.

## Generate paper figures

```bash
python -m scripts.make_paper_figures \
  --evaluation-csv paper_results/evaluation/all_method_evaluation.csv \
  --alpha-sweep-csv paper_results/sweeps/all_alpha_sweep_methods.csv \
  --figures-dir paper_results/figures
```

This creates PNG and PDF versions.

## One-command driver

```bash
python -m scripts.run_paper_pipeline --train --evaluate --sweep --figures --max-samples 100
```

For the full final run, remove `--max-samples 100`.

## Notes about classical algorithms

The baseline implementations are designed to be reproducible and fully contained in PyTorch:

- MLE and purification MLE are iterative and slow for three qubits.
- Bayesian tomography uses a particle posterior mean approximation.
- Compressed sensing uses low-rank eigenvalue shrinkage with PSD trace-one projection.

For final paper numbers, increase baseline iterations/particles in the `baselines:` section of the YAML files after confirming that the smoke tests work.
