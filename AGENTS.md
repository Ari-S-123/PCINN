# Repository Guidelines

## Project Structure & Module Organization

- `MMA_PCINN.ipynb`: primary notebook — 27 cells (11 Markdown, 16 Code) covering models, preprocessing, leave-one-experiment-out split, baseline NN training, PCINN training, model export, and PCINNv2 upgraded architecture with inline loss curve plots and comparison.
- `MMA_PCINN.py`: original single-file script (reference only, not the primary entry point).
- `PCINN_IMPROVEMENT_PLAN.md`: detailed improvement plan with implementation checklist.
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights loaded by the notebook.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
- `exports/`: runtime-generated directory containing saved model weights (`.pt`), inference bundles, and JSON metadata.
- `neural_networks_polymerization (2).pdf`: paper reference for expected methodology.

## Build, Test, and Development Commands

Current environment: Python 3.13.9 with PyTorch 2.10.0.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

On systems with CUDA, follow https://pytorch.org/get-started/locally/ to install the CUDA-enabled PyTorch build instead of the default CPU-only wheel.

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run all cells. No separate build step exists.

## Notebook Organization

The notebook has three training sections (each with an original commented-out cell + improved cell below it):

1. **Baseline NN** — data-only MSE training (no Jacobian regularization).
2. **PCINN** — data MSE + Jacobian matching against the pretrained theory model.
3. **PCINNv2** — upgraded architecture: residual correction (`y = theory(x) + delta(concat(x_repr, theory(x)))`), physical output heads (X in [0,1], M > 0), ResidualBlock + LayerNorm, optional Fourier features, Xavier init, physical-space Jacobian matching.

Key infrastructure cells:
- **Reproducibility & device** — `SEED=42`, all RNGs seeded, `DEVICE` auto-detected.
- **KMP workaround gate** — `ENABLE_UNSAFE_KMP_WORKAROUND=False` by default; if enabled, `KMP_DUPLICATE_LIB_OK` is set before heavy numeric imports in a fresh kernel.
- **Torch scalers** — `SCALERX_MIN`, `SCALERX_MAX`, `SCALERX_DENOM` on `DEVICE` for Jacobian sampling.
- **`safe_log()`** — numerically safe log for loss curve plotting.
- **Export cell** — saves weights, inference bundle, JSON metadata, smoke-test reload.

## Coding Style & Naming Conventions
- Follow PEP 8 basics: 4-space indentation and readable line lengths.
- Use `snake_case` for functions/variables and `PascalCase` for `nn.Module` classes.
- Type hints on `forward()` methods and utility functions (`-> torch.Tensor`, `-> np.ndarray`).
- Preserve paper semantics: 5 inputs (`[M]`, `[S]`, `[I]`, `temp`, `time`) and 6 outputs (`X`, `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv`; weight outputs log10-transformed).
- All models instantiated on `DEVICE`; all data tensors on `DEVICE`.
- Use `model.train()` / `model.eval()` discipline in training loops.
- Use `torch.inference_mode()` for evaluation (stricter than `no_grad()`).
- Original paper-baseline cells are preserved (commented out, marked `# [ORIGINAL — paper baseline, do not execute]`).

## Testing Guidelines
Automated tests are not currently included.

- Minimum validation: open `MMA_PCINN.ipynb`, restart the kernel, and run all cells top-to-bottom. All three training sections (baseline NN, PCINN, PCINNv2) should complete without errors and produce inline loss curve plots.
- The PCINNv2 cell should render a side-by-side comparison plot of all three models.
- The export cell should save files to `exports/` and pass the reload assertion (`max_diff < 1e-6`).
- `NNpred`, `EBNNpred`, and `V2pred` should each be shape `(N, 6)` for the held-out reaction.
- Reproducibility checks should keep paper baselines unless intentionally changed: `lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, leave-one-experiment-out with `TestReaction` in `1..8`.
- For PCINN, keep Jacobian matching against the pretrained theory model with `32` sampled points/epoch over the paper ranges (`T: 323-363 K`, `[M]: 0.5-5`, `[I]: 0.005-0.1`, `time: 300-36000 s`).
- Checkpoint loading should try `torch.load(..., weights_only=True)` first, then fallback to `weights_only=False` only on `pickle.UnpicklingError` (controlled checkpoint provenance).

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
