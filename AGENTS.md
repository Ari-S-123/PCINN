# Repository Guidelines

## Project Structure & Module Organization

- `MMA_PCINN.ipynb`: primary notebook — models, preprocessing, leave-one-experiment-out split, baseline NN and PCINN training with inline loss curve plots.
- `MMA_PCINN.py`: original single-file script (reference only, not the primary entry point).
- `IMPLEMENTATION_PLAN.md`: migration spec used to verify notebook structure, compatibility fixes, and validation requirements.
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights loaded by the notebook.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
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

## Coding Style & Naming Conventions
- Follow PEP 8 basics: 4-space indentation and readable line lengths.
- Use `snake_case` for functions/variables and `PascalCase` for `nn.Module` classes.
- Preserve paper semantics: 5 inputs (`[M]`, `[S]`, `[I]`, `temp`, `time`) and 6 outputs (`X`, `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv`; weight outputs log10-transformed).
- Preserve migration parity with `MMA_PCINN.py`: only compatibility changes are allowed (`torch.tanh`, `torch.sigmoid`, and checkpoint `torch.load(..., weights_only=...)` handling).

## Testing Guidelines
Automated tests are not currently included.

- Minimum validation: open `MMA_PCINN.ipynb`, restart the kernel, and run all cells top-to-bottom. Both training sections should complete without errors and produce inline loss curve plots.
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
- Keep file paths relative to the repository root.
- Ensure contributions remain compatible with the project license in `License.md` (CC BY-NC 4.0).
