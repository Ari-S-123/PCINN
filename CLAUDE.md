# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polymer Chemistry Informed Neural Networks (PCINNs) — a single-script research repo that combines a data-driven neural network with Jacobian-based regularization from a pretrained kinetic ("theory") model to predict MMA (methyl methacrylate) polymerization outcomes. Paper: https://doi.org/10.1039/D4PY00995A

Licensed under CC BY-NC 4.0. Do not commit proprietary datasets or large generated checkpoints.

## Running

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch pandas numpy matplotlib openpyxl
python MMA_PCINN.py
```

Paper-aligned environment: Python 3.9.18, PyTorch 2.0.1. No build step, no test suite — `python MMA_PCINN.py` runs both training loops (baseline NN then PCINN) and plots train/test loss curves.

## Architecture

Everything lives in `MMA_PCINN.py`, organized with `# %%` cell markers:

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

## Coding Conventions

- PEP 8 basics: 4-space indent, readable line lengths.
- `snake_case` for functions/variables, `PascalCase` for `nn.Module` classes.
- Keep `# %%` section blocks to preserve script organization.
- If behavior diverges from the paper setup, call out the deviation explicitly.
