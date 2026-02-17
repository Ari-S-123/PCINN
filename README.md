# Polymer Chemistry Informed Neural Networks (PCINNs) for Data-Driven Polymerization Modeling

This repository implements Polymer Chemistry Informed Neural Networks (PCINNs) for methyl methacrylate (MMA) free-radical polymerization prediction. The method combines data-driven training with kinetic-model guidance via Jacobian matching.

Paper: Ballard, N. *Polymer Chemistry Informed Neural Networks.* *Polym. Chem.*, 2024. DOI: [10.1039/D4PY00995A](https://doi.org/10.1039/D4PY00995A)  
Upstream repository: https://github.com/PolymatGIQ/PCINN  
Primary implementation: `MMA_PCINN.ipynb`

The original script `MMA_PCINN.py` is kept for reference only.

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

Open `MMA_PCINN.ipynb` in Jupyter/VS Code and run cells.

## Notebook Workflows

The notebook contains two workflows:

1. **Single-fold training/export workflow (quick path)**
- Uses `TestReaction` to hold out one reaction (default: `TestReaction = 8`).
- Trains three models: Baseline NN, PCINN, Soft-Anchor PCINN.
- Plots loss curves and exports model artifacts.

2. **Full leave-one-experiment-out multi-seed CV (long-running, optional)**
- Runs all 8 folds and multiple seeds (`N_SEEDS = 5` by default).
- Default run count: `3 models × 8 folds × 5 seeds = 120` training runs.
- Produces table-style aggregate metrics and fold-level PCINN vs SA-PCINN comparisons.

## Notebook Structure

`MMA_PCINN.ipynb` currently has **30 cells (13 markdown, 17 code)**:

1. Dependencies and pinned install cell.
2. Imports/environment setup (including optional unsafe KMP gate, off by default).
3. Reproducibility and device setup (`SEED=42`, `DEVICE` auto-select).
4. Model definitions: `NNmodel` and `DomainModel`.
5. Data preprocessing and scaler/domain model loading.
6. Jacobian sampling range configuration.
7. Single-fold split configuration via `TestReaction`.
8. Baseline NN training (paper-original cell preserved as commented reference + improved executable cell).
9. PCINN training (paper-original cell preserved as commented reference + improved executable cell).
10. Soft-Anchor PCINN training.
11. Single-fold loss/curve comparison summary.
12. Full 8-fold multi-seed cross-validation section.
13. Model export with smoke-test reload verification.

## Model Objectives

- **Baseline NN:** data-only MSE.
- **PCINN:** `L_data + L_jac` (Jacobian matching to pretrained theory model).
- **Soft-Anchor PCINN:** `L_data + L_jac + λ_anchor * L_anchor`, where:

$$L_{\text{anchor}} = \text{MSE}(y_{\text{net}}(x_j), y_{\text{theory}}(x_j))$$

using the same sampled points used for Jacobian matching.

## Validation Checklist

Quick path:
- Restart kernel and run through the single-fold sections.
- Confirm all three model sections execute and render loss curves.
- Confirm `NNpred`, `EBNNpred`, and `SApred` each have shape `(N, 6)`.
- Confirm export files are written to `exports/` and smoke-test reload passes (`max_diff < 1e-6`).

Optional full CV:
- Run the multi-seed CV cell and confirm aggregate tables/plots are produced.
- Expect substantially longer runtime than the single-fold workflow.

## License

This repository is licensed under CC BY-NC 4.0. See `License.md`.
