# Polymer Chemistry Informed Neural Networks (PCINNs) for data-driven modelling of polymerization processes
This repository contains code for the implementation of Polymer Chemistry Informed Neural Networks (PCINNs), a method that combines kinetic models of polymerization processes with neural network training to improve predictive accuracy. The approach mitigates common challenges, such as data limitations and poor extrapolation, by leveraging domain knowledge to enhance model reliability.
More details are described in [the PolymatGIQ paper](https://doi.org/10.1039/D4PY00995A).

Primary implementation notebook: `MMA_PCINN.ipynb`
Upstream repository: https://github.com/PolymatGIQ/PCINN

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

Open `MMA_PCINN.ipynb` in Jupyter or Cursor/VSCode/etc. and run all cells. The notebook trains three models — a baseline NN, a corrected PCINN, and an upgraded PCINNv2 — then plots train/test/Jacobian loss curves and a side-by-side comparison.

The original script `MMA_PCINN.py` is retained for reference.

## Notebook Structure

The notebook contains 27 cells (11 Markdown, 16 Code) organized into these sections:

1. **Dependencies** — `%pip install` with pinned versions.
2. **Imports & environment** — Consolidated imports, opt-in KMP workaround (off by default, and set before heavy numeric imports when enabled).
3. **Reproducibility & device** — `SEED=42`, all RNGs seeded, automatic GPU/CPU selection via `DEVICE`.
4. **Model definitions** — `NNmodel` (paper baseline), `DomainModel` (pretrained theory surrogate).
5. **Data preprocessing** — Numerically safe min-max scaling, Torch scalers on `DEVICE` for training loops.
6. **Jacobian sampling configuration** — Physical-domain uniform samplers for T, [M], [I], time.
7. **Training configuration** — Leave-one-experiment-out split (`TestReaction`), data tensors on `DEVICE`.
8. **Baseline NN training** — Original cell (commented out) + improved cell with canonical grad order, `train()`/`eval()` discipline, `inference_mode()`.
9. **PCINN training** — Original cell (commented out) + improved cell with Torch scalers for Jacobian sampling, proper `detach().requires_grad_(True)`, theory Jacobian `.detach()`, separate jac loss tracking.
10. **Model export** — Saves weights, inference bundle, and JSON metadata; smoke-test reload assertion.
11. **PCINNv2 (upgraded architecture)** — Residual-correction model with physical output heads, ResidualBlock + LayerNorm, optional Fourier features, Xavier init, physical-space Jacobian matching.
12. **Results summary** — Side-by-side loss comparison of all three models.

The original paper-baseline training cells are preserved (commented out) so the paper implementation remains available for comparison.

## Notebook Notes

- Improved training loops fix the epoch-0 skip bug in the original code, use canonical `zero_grad → forward → backward → step` order, and add `model.train()`/`model.eval()` mode discipline.
- Jacobian computation: theory model parameters are frozen (`requires_grad_(False)`) but autograd remains enabled (no `torch.no_grad()`) so `jacrev` works correctly. Theory Jacobians are `.detach()`ed as MSE targets.
- Checkpoint loading tries `weights_only=True` first (PyTorch 2.6+ secure default), falling back to `weights_only=False` on `pickle.UnpicklingError`.
- The `KMP_DUPLICATE_LIB_OK` workaround is disabled by default and only applied before NumPy/PyTorch imports when explicitly enabled.
- Default cross-validation setting is `TestReaction = 8`. To reproduce full leave-one-experiment-out results, rerun with `TestReaction` in `1..8`.

## PCINNv2 Architecture

An experimental upgrade that combines:
- **Residual correction**: `y(x) = theory(x) + delta(concat(x_repr, theory(x)))` where `x_repr` is raw `x` or Fourier features — the NN learns only what the theory model misses.
- **Physical output heads**: X constrained to [0,1] via clamp, M > 0 via multiplicative `exp(delta)` correction.
- **Residual blocks + LayerNorm** for stable gradient flow.
- **Optional Fourier features** (flag-gated via `USE_FOURIER`).
- **Physical-space Jacobian matching** — both theory and PCINNv2 Jacobians compared in physical units.

## Validation

- Restart kernel and run all notebook cells top-to-bottom.
- Three training sections should execute: baseline NN, PCINN, and PCINNv2.
- Each should render inline loss curve plots. The final PCINNv2 cell produces a side-by-side comparison.
- `NNpred`, `EBNNpred`, and `V2pred` should each be shape `(N, 6)` for the held-out reaction.
- The export cell should save files to `exports/` and pass the reload assertion (`max_diff < 1e-6`).

## LICENSE
This repository is licensed under CC BY-NC 4.0.
For more information please refer to the [license section](https://github.com/Ari-S-123/PCINN/blob/main/License.md).
