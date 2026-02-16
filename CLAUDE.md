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

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run all cells. The notebook trains both the baseline NN and the PCINN, then plots train/test loss curves inline.

Current environment: Python 3.13.9, PyTorch 2.10.0. The original script `MMA_PCINN.py` (paper-aligned with Python 3.9.18, PyTorch 2.0.1) is retained as a reference but is not the primary entry point.

## Repository Structure

- `MMA_PCINN.ipynb`: primary notebook — 17 cells (8 Markdown, 9 Code) covering models, data loading, baseline NN training, and PCINN training.
- `MMA_PCINN.py`: original single-file script (reference only).
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.

## Architecture

The notebook is organized into clearly delineated sections:

1. **Model definitions** — `NNmodel` (3-layer tanh, 5→128→64→6) is the trainable network used for both baseline and PCINN. `DomainModel` (relu + sigmoid/softplus heads) is the pretrained theory model loaded from `MMA_solution_net.pt`.
2. **Data loading & preprocessing** — Reads `PMMAordered.xlsx`, applies min-max scaling (vectors in `scalerx_min.npy`/`scalerx_max.npy`), log10-transforms weight-average outputs (columns 1–5 of Y).
3. **Baseline NN training** — Standard MSE on data with leave-one-experiment-out split (`TestReaction` 1–8).
4. **PCINN training** — Same data loss plus Jacobian matching: 32 random points per epoch sampled over physical ranges, Jacobian computed via `torch.func.vmap(jacrev(...))` for both PCINN and theory model, with MSE between the two Jacobians added to the total loss.

### Key data semantics

- **5 inputs**: `[M]`, `[S]`, `[I]`, `temp`, `time` (all scaled 0–1)
- **6 outputs**: `X` (conversion), `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv` (weight outputs are log10-transformed)
- **Jacobian sampling ranges**: T 323–363 K, [M] 0.5–5, [I] 0.005–0.1, time 300–36000 s

### Paper-baseline hyperparameters (preserve unless intentionally changing)

`lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, `totaljacsamples=32`, leave-one-experiment-out with `TestReaction` in 1..8.

### API migration notes (from original script)

Three changes were made when migrating from the original script to the notebook with updated dependencies:

1. `F.tanh(...)` → `torch.tanh(...)` (deprecated since PyTorch ~1.2)
2. `F.sigmoid(...)` → `torch.sigmoid(...)` (same deprecation)
3. `torch.load('MMA_solution_net.pt')` → `torch.load('MMA_solution_net.pt', weights_only=True)` (required since PyTorch 2.6)

## Coding Conventions

- PEP 8 basics: 4-space indent, readable line lengths.
- `snake_case` for functions/variables, `PascalCase` for `nn.Module` classes.
- If behavior diverges from the paper setup, call out the deviation explicitly.
