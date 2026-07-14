# Three-attack final QST paper code

This package keeps the same file structure as the earlier final-paper code, but the attack set is restricted to exactly:

1. `random_replacement`: replaces `m = round(alpha * N)` copies by `m` independently generated random states and averages them.
2. `targeted_replacement`: replaces `m = round(alpha * N)` copies by the same fixed target state, default `target_state: zero`.
3. `frequency_pgd`: performs projected gradient ascent in the frequency vector with an `l_infinity` radius and simplex projection per Pauli setting.

The robust NN still uses the three-stage schedule:

- Stage 1: clean warm-up.
- Stage 2: balanced adversarial warm-up.
- Stage 3: hierarchical physical/PGD training with squared Frobenius loss.

The paper figure script now creates:

- Clean NN vs adversarially trained NN fidelity comparison.
- Adversarial NN vs baseline algorithms fidelity comparison.
- Full method/attack fidelity bars.
- Infidelity empirical CDFs.
- Alpha-sweep fidelity curves for the two physical attacks.

Smoke/test commands:

```bash
pip install -r requirements.txt
pytest -q
python -m scripts.train --config configs/paper/smoke_clean.yaml
python -m scripts.train --config configs/paper/smoke_adversarial.yaml
```
