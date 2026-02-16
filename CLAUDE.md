# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymer Chemistry Informed Neural Networks (PCINNs) — a research repo that combines a data-driven neural network with Jacobian-based regularization from a pretrained kinetic ("theory") model to predict MMA (methyl methacrylate) polymerization outcomes. Paper: https://doi.org/10.1039/D4PY00995A

Licensed under CC BY-NC 4.0. Do not commit proprietary datasets or large generated checkpoints.

## Running

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run all cells. The notebook trains three models (baseline NN, PCINN, PCINNv2), then plots loss curves and a comparison.

Current environment: Python 3.13.9, PyTorch 2.10.0. The original script `MMA_PCINN.py` (paper-aligned with Python 3.9.18, PyTorch 2.0.1) is retained as a reference but is not the primary entry point.

## Repository Structure

- `MMA_PCINN.ipynb`: primary notebook — 27 cells (11 Markdown, 16 Code) covering models, data loading, baseline NN training, PCINN training, model export, and PCINNv2 upgraded architecture.
- `MMA_PCINN.py`: original single-file script (reference only).
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
- `exports/`: directory created at runtime containing saved model weights, inference bundles, and JSON metadata.
- `PCINN_IMPROVEMENT_PLAN.md`: detailed improvement plan with implementation checklist.

## Architecture

The notebook is organized into clearly delineated sections:

1. **Imports & environment** — Consolidated imports (no `_pickle`, no `F` alias), opt-in KMP workaround, reproducibility seeds (`SEED=42`), and `DEVICE` abstraction.
2. **Model definitions** — `NNmodel` (3-layer tanh, 5→128→64→6) is the trainable network used for both baseline and PCINN. `DomainModel` (relu + sigmoid/softplus heads) is the pretrained theory model loaded from `MMA_solution_net.pt`. `PCINNv2` is the upgraded residual-correction model (see below).
3. **Data loading & preprocessing** — Reads `PMMAordered.xlsx`, applies numerically safe min-max scaling (epsilon denominator), log10-transforms weight-average outputs (columns 1–5 of Y). Both NumPy scalers (for one-time preprocessing) and Torch scalers on `DEVICE` (for training loops) are created.
4. **Baseline NN training** — Standard MSE on data with leave-one-experiment-out split (`TestReaction` 1–8). Original cell preserved (commented out); improved cell uses canonical grad order, `train()`/`eval()` discipline, and `inference_mode()`.
5. **PCINN training** — Same data loss plus Jacobian matching: 32 random points per epoch sampled over physical ranges, Jacobian computed via `torch.func.vmap(jacrev(...))`. Theory Jacobian is `.detach()`ed as MSE target; no `torch.no_grad()` wrapping. Original cell preserved (commented out).
6. **Model export** — Saves PCINN weights, inference bundle (with `copy.deepcopy`), and JSON metadata to `exports/`. Includes smoke-test reload assertion.
7. **PCINNv2 training** — Upgraded architecture with residual correction, physical output heads, physical-space Jacobian matching, and comparison plots across all three models.

### PCINNv2 architecture

- **Residual correction**: `y(x) = theory(x) + delta(concat(x_repr, theory(x)))` where `x_repr` is raw inputs or Fourier-encoded inputs.
- **Physical output heads**: X via `clamp(theory_X + delta_X, 0, 1)`, M via `theory_M * exp(delta_M)` (ensures positivity).
- **ResidualBlock + LayerNorm**: Pre-norm residual block (LayerNorm → Linear → Tanh → Linear + skip) for stable gradient flow.
- **FourierFeatures**: Optional positional encoding (flag `USE_FOURIER`, default off) to reduce spectral bias.
- **Xavier initialization**: Matched to tanh activations for all delta network layers.
- **Physical-space Jacobian matching** (Section 6.7 Option A): Both theory and PCINNv2 Jacobians compared in physical units via `forward_physical()`.

### Key data semantics

- **5 inputs**: `[M]`, `[S]`, `[I]`, `temp`, `time` (all scaled 0–1)
- **6 outputs**: `X` (conversion), `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv` (weight outputs are log10-transformed)
- **Jacobian sampling ranges**: T 323–363 K, [M] 0.5–5, [I] 0.005–0.1, time 300–36000 s

### Paper-baseline hyperparameters (preserve unless intentionally changing)

`lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, `totaljacsamples=32`, leave-one-experiment-out with `TestReaction` in 1..8.

### API migration notes (from original script)

Changes made when migrating from the original script to the improved notebook:

1. `F.tanh(...)` → `torch.tanh(...)` (deprecated since PyTorch ~1.2)
2. `F.sigmoid(...)` → `torch.sigmoid(...)` (same deprecation)
3. `F.relu(...)` → `torch.relu(...)`, `F.softplus(...)` → `torch.nn.functional.softplus(...)` (removed `F` import alias)
4. `torch.load('MMA_solution_net.pt')` → `torch.load('MMA_solution_net.pt', weights_only=True, map_location=DEVICE)` (required since PyTorch 2.6)
5. `super(ClassName, self).__init__()` → `super().__init__()` (Python 3 idiom)
6. `_pickle` import removed; catch only `pickle.UnpicklingError`
7. NumPy/Torch interop fixed: Jacobian sampling uses Torch scalers (`SCALERX_MIN`/`SCALERX_DENOM`) instead of NumPy arrays
8. Epoch-0 skip bug fixed in both training loops
9. `optimizer.zero_grad(set_to_none=True)` for minor memory efficiency
10. `torch.inference_mode()` for test evaluation (stricter than `no_grad()`)

### Preservation strategy

Original paper-baseline training cells are preserved as commented-out code (marked `# [ORIGINAL — paper baseline, do not execute]`). Improved cells appear directly below their originals.

## Coding Conventions

- PEP 8 basics: 4-space indent, readable line lengths.
- `snake_case` for functions/variables, `PascalCase` for `nn.Module` classes.
- Type hints on `forward()` methods and utility functions.
- If behavior diverges from the paper setup, call out the deviation explicitly.
- All models instantiated on `DEVICE`; all data tensors on `DEVICE`.
- Use `model.train()` / `model.eval()` discipline in training loops.
