# Repository Guidelines

## Project Structure & Module Organization

- `MMA_PCINN.ipynb`: primary notebook - 30 cells (13 Markdown, 17 Code) covering setup, single-fold Baseline/PCINN/Soft-Anchor training, optional full 8-fold multi-seed CV, and export.
- `MMA_PCINN.py`: original single-file script (reference only, not the primary entry point).
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights loaded by the notebook.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
- `exports/`: runtime-generated directory containing per-model weights (`.pt`), inference bundles, JSON metadata, and convenience `best_fold*.pt` copies.
- `Original PCINN Paper.pdf`: paper reference PDF in the repo.

## Build, Test, and Development Commands

Current environment: Python 3.13.9 with PyTorch 2.10.0.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

On systems with CUDA, follow https://pytorch.org/get-started/locally/ to install the CUDA-enabled PyTorch build instead of the default CPU-only wheel.

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run cells. No separate build step exists.

## Notebook Organization

The notebook has three single-fold training sections (each with an original commented-out cell + improved executable cell below it):

1. **Baseline NN** - data-only MSE training (no Jacobian regularization).
2. **PCINN** - data MSE + Jacobian matching against the pretrained theory model.
3. **Soft-Anchor PCINN** - PCINN objective plus value-level anchor penalty toward theory predictions at sampled points.

Additional notebook sections:
- **Single-fold results summary** - side-by-side loss plots and final test losses for Baseline/PCINN/SA-PCINN.
- **Full leave-one-experiment-out cross-validation (multi-seed)** - optional long-running section that runs 8 folds x `N_SEEDS` for all three learned models and compares against the kinetic model.
- **Model export** - exports all three single-fold models, marks/copies the best one, and smoke-tests reload.

Key infrastructure cells:
- **Reproducibility & device** - `SEED=42`, all RNGs seeded, `DEVICE` auto-detected.
- **KMP workaround gate** - `ENABLE_UNSAFE_KMP_WORKAROUND=False` by default; if enabled, `KMP_DUPLICATE_LIB_OK` is set before heavy numeric imports in a fresh kernel.
- **Torch scalers** - `SCALERX_MIN`, `SCALERX_MAX`, `SCALERX_DENOM` on `DEVICE` for Jacobian sampling.
- **`safe_log()`** - numerically safe log for loss curve plotting.
- **Export cell** - saves weights, inference bundles, JSON metadata, and validates reload.

## Coding Style & Naming Conventions
- Follow PEP 8 basics: 4-space indentation and readable line lengths.
- Use `snake_case` for functions/variables and `PascalCase` for `nn.Module` classes.
- Type hints on `forward()` methods and utility functions (`-> torch.Tensor`, `-> np.ndarray`).
- Preserve paper semantics: 5 inputs (`[M]`, `[S]`, `[I]`, `temp`, `time`) and 6 outputs (`X`, `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv`; molecular-weight targets are trained in log10 space).
- All models instantiated on `DEVICE`; all data tensors on `DEVICE`.
- Use `model.train()` / `model.eval()` discipline in training loops.
- Use `torch.inference_mode()` for evaluation (stricter than `no_grad()`).
- Original paper-baseline cells are preserved (commented out, marked `# [ORIGINAL - paper baseline, do not execute]`).

## Testing Guidelines
Automated tests are not currently included.

Quick validation (recommended for iteration):
- Open `MMA_PCINN.ipynb`, restart the kernel, and run through the single-fold sections (Baseline NN, PCINN, Soft-Anchor PCINN), results summary, and export.
- The results summary cell should render side-by-side comparison plots for all three models.
- The export cell should save files to `exports/` and pass the reload assertion (`max_diff < 1e-6`).
- `NNpred`, `EBNNpred`, and `SApred` should each be shape `(N, 6)` for the held-out reaction.

Extended validation (optional, long-running):
- Run the full multi-seed LOEO CV section; by default this is `3 models x 8 folds x 5 seeds = 120` training runs.
- Confirm aggregate tables/plots are generated for Table-4-style outputs and per-fold PCINN vs SA-PCINN comparisons.

Reproducibility settings to preserve unless intentionally changed:
- `lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, leave-one-experiment-out with `TestReaction` in `1..8`.
- The notebook's single-fold examples currently default to `TestReaction = 8`.
- For PCINN-based methods, keep Jacobian matching against the pretrained theory model with `32` sampled points/epoch over the paper ranges (`T: 323-363 K`, `[M]: 0.5-5`, `[I]: 0.005-0.1`, `time: 300-36000 s`).
- Checkpoint loading should try `torch.load(..., weights_only=True)` first, then fall back to `weights_only=False` only on `pickle.UnpicklingError` (controlled checkpoint provenance).

## Commit & Pull Request Guidelines
Recent history uses short imperative messages (for example, `Update README.md`, `Re-format`).

- Write concise, action-oriented commit subjects.
- In PRs, include summary, changed artifacts, and reproduction commands.
- If behavior diverges from the paper setup, call out the deviation explicitly and include before/after loss plots.

## Security & Configuration Tips
- Do not commit proprietary datasets or unintended large generated checkpoints.
- The `exports/` directory is generated at runtime and should generally not be committed.
- Keep file paths relative to the repository root.
- Ensure contributions remain compatible with the project license in `License.md` (CC BY-NC 4.0).
