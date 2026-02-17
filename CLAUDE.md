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

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run all cells. The notebook has two workflows: a **fast single-fold** path (default `TestReaction=8`) that trains three models (baseline NN, PCINN, Soft-Anchor PCINN), plots loss curves, exports weights, and an **optional full 8-fold multi-seed CV** section that reproduces the paper's Table 4 evaluation across all folds and multiple random seeds.

Current environment: Python 3.13.9, PyTorch 2.10.0. The original script `MMA_PCINN.py` (paper-aligned with Python 3.9.18, PyTorch 2.0.1) is retained as a reference but is not the primary entry point.

## Repository Structure

- `MMA_PCINN.ipynb`: primary notebook — 30 cells (13 Markdown, 17 Code) covering models, data loading, baseline NN training, PCINN training, Soft-Anchor PCINN, results comparison, full multi-seed cross-validation, and model export.
- `MMA_PCINN.py`: original single-file script (reference only).
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
- `exports/`: directory created at runtime containing saved model weights, inference bundles, and JSON metadata.

## Architecture

The notebook is organized into clearly delineated sections:

1. **Imports & environment** — Consolidated imports (no `_pickle`, no `F` alias), opt-in KMP workaround, reproducibility seeds (`SEED=42`), and `DEVICE` abstraction.
2. **Model definitions** — `NNmodel` (3-layer tanh, 5→128→64→6) is the trainable network used for baseline, PCINN, and Soft-Anchor PCINN. `DomainModel` (relu + sigmoid/softplus heads) is the pretrained theory model loaded from `MMA_solution_net.pt`.
3. **Data loading & preprocessing** — Reads `PMMAordered.xlsx`, applies numerically safe min-max scaling (epsilon denominator), log10-transforms weight-average outputs (columns 1–5 of Y). Both NumPy scalers (for one-time preprocessing) and Torch scalers on `DEVICE` (for training loops) are created. Loads and freezes `Domain_NN`.
4. **Jacobian sampling configuration** — Defines physical-range `torch.distributions.Uniform` samplers for T, [M], [I], and time. `[S]` is derived as `10 - [M]` (not independently sampled).
5. **Training configuration** — Sets `TestReaction` (default `8`) for leave-one-experiment-out split. Converts train/test arrays to tensors on `DEVICE` once. Defines `safe_log` utility for plotting loss curves.
6. **Baseline NN training** — Standard MSE on data with leave-one-experiment-out split (`TestReaction` 1–8). Original cell preserved (commented out); improved cell uses canonical grad order, `train()`/`eval()` discipline, and `inference_mode()`.
7. **PCINN training** — Same data loss plus Jacobian matching: 32 random points per epoch sampled over physical ranges, Jacobian computed via `torch.func.vmap(jacrev(...))`. Theory Jacobian is `.detach()`ed as MSE target; no `torch.no_grad()` wrapping. Original cell preserved (commented out).
8. **Soft-Anchor PCINN training** — Same architecture and Jacobian matching as the PCINN, with an additional soft anchor loss term: `ANCHOR_WEIGHT * MSE(net(x_samples), Domain_NN(x_samples))`. This penalizes deviation from theory predictions at randomly sampled points, providing value-level regularization complementary to the Jacobian's trend-level regularization.
9. **Results summary** — Side-by-side train/test loss curves for all three single-fold models and prints final test losses for the held-out reaction.
10. **Full leave-one-experiment-out CV (multi-seed)** — Optional long-running section. Trains all 3 architectures across 8 folds × `N_SEEDS` (default 5) random seeds (`BASE_SEEDS=[42, 123, 256, 789, 1337]`). Uses helper functions `train_baseline_nn`, `train_pcinn`, `train_sa_pcinn`, and `per_output_mse`. Reports Table 4–style mean ± std MSE per output, per-fold head-to-head comparison, and bar-chart visualizations.
11. **Model export** — Exports all three single-fold models (`baseline_nn`, `pcinn`, `sa_pcinn`) for the current `TestReaction`: per-model weights (`.pt`), inference bundles (with `copy.deepcopy`, scaling params, and metadata), and human-readable JSON metadata. Selects the best model by final test loss and copies its weights/bundle to `best_fold{fold}_*` convenience files. Includes smoke-test reload assertion (`max_diff < 1e-6`).

### DomainModel output semantics (CRITICAL)

`DomainModel` returns `[sigmoid(X), softplus(log10_Mn), softplus(log10_Mw), softplus(log10_Mz), softplus(log10_Mz+1), softplus(log10_Mv)]`. The softplus activation ensures the log10 values are positive (since M > 1 → log10(M) > 0). Columns 1–5 are ALREADY in log-space — they are NOT physical molecular weights. Any code that consumes Domain_NN output must respect this: comparisons with `Ytrainsample` (which also stores log10 M values) are direct, and no additional log10 transform should be applied.

### Key data semantics

- **5 inputs**: `[M]`, `[S]`, `[I]`, `temp`, `time` (all scaled 0–1)
- **6 outputs**: `X` (conversion), `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv` (weight outputs are log10-transformed)
- **Jacobian sampling ranges**: T 323–363 K, [M] 0.5–5, [S] = 10 − [M] (derived, not independent), [I] 0.005–0.1, time 300–36000 s

### Paper-baseline hyperparameters (preserve unless intentionally changing)

`lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, `totaljacsamples=32`, leave-one-experiment-out with `TestReaction` in 1..8.

### Soft-Anchor PCINN hyperparameters

`ANCHOR_WEIGHT=0.05` (default). Same `lr`, `epochs`, `optimizer`, and `totaljacsamples` as the standard PCINN. The anchor loss uses the same 32 randomly sampled points as the Jacobian matching.

### Multi-seed CV hyperparameters

`N_SEEDS=5`, `BASE_SEEDS=[42, 123, 256, 789, 1337]`. Each fold uses `run_seed = base_seed * 10 + fold` for deterministic but unique seeding. Total learned-model runs: `3 architectures × 8 folds × N_SEEDS`. The kinetic model is deterministic and computed once per fold (replicated `N_SEEDS` times for uniform averaging).

### Why not residual correction (PCINNv2)?

A residual-correction architecture (`y = Domain_NN(x) + delta(x)`) was extensively evaluated and found to underperform the standard PCINN in this low-data regime (~50 training points, leave-one-experiment-out CV). Two fundamental issues were identified:

1. **Jacobian matching self-defeats**: For a residual architecture, `J(PCINNv2) = J(Domain_NN) + J(delta)`. Jacobian matching pushes `J(PCINNv2) ≈ J(Domain_NN)`, which forces `J(delta) ≈ 0` — meaning the correction must be approximately constant across all inputs, preventing it from learning spatially varying corrections.

2. **Extrapolation of corrections**: The delta network learns corrections at training points, but in leave-one-out CV the test reaction occupies a different region of input space (different temperature, concentrations). A correction learned at 60–70°C doesn't reliably extrapolate to 80°C. The standard PCINN avoids this because it learns the full mapping constrained by theory gradients across the entire input space, not corrections anchored to specific training regions.

The soft-anchor approach provides value-level regularization (biasing toward theory predictions) at the loss level without the structural constraints that caused these issues.

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
