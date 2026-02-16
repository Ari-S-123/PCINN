# Repository Guidelines

## Project Structure & Module Organization
This is a single-script research repository.

- `MMA_PCINN.py`: main workflow (models, preprocessing, leave-one-experiment-out split, baseline NN and PCINN training).
- `MMA_solution_net.pt`: pretrained domain ("theory") model weights loaded by the script.
- `PMMAordered.xlsx`: MMA polymerization dataset.
- `scalerx_max.npy`, `scalerx_min.npy`: min/max vectors for 0-1 input scaling.
- `neural_networks_polymerization (2).pdf`: paper reference for expected methodology.

## Build, Test, and Development Commands
Paper-aligned environment is Python `3.9.18` with PyTorch `2.0.1` (newer versions may work but should be validated).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch pandas numpy matplotlib openpyxl
python MMA_PCINN.py
```

`python MMA_PCINN.py` runs both training loops and plots train/test loss. No separate build step exists.

## Coding Style & Naming Conventions
- Follow PEP 8 basics: 4-space indentation and readable line lengths.
- Use `snake_case` for functions/variables and `PascalCase` for `nn.Module` classes.
- Keep section blocks (`# %%`) to preserve the current script organization.
- Preserve paper semantics: 5 inputs (`[M]`, `[S]`, `[I]`, `temp`, `time`) and 6 outputs (`X`, `Mn`, `Mw`, `Mz`, `Mzplus1`, `Mv`; weight outputs log10-transformed).

## Testing Guidelines
Automated tests are not currently included.

- Minimum validation: run `python MMA_PCINN.py` and confirm both training sections run without runtime errors.
- Reproducibility checks should keep paper baselines unless intentionally changed: `lr=3e-4`, `epochs=10000`, `Adam`, `MSELoss`, leave-one-experiment-out with `TestReaction` in `1..8`.
- For PCINN, keep Jacobian matching against the pretrained theory model with `32` sampled points/epoch over the paper ranges (`T: 323-363 K`, `[M]: 0.5-5`, `[I]: 0.005-0.1`, `time: 300-36000 s`).

## Commit & Pull Request Guidelines
Recent history uses short imperative messages (for example, `Update README.md`, `Re-format`).

- Write concise, action-oriented commit subjects.
- In PRs, include summary, changed artifacts, and reproduction commands.
- If behavior diverges from the paper setup, call out the deviation explicitly and include before/after loss plots.

## Security & Configuration Tips
- Do not commit proprietary datasets or unintended large generated checkpoints.
- Keep file paths relative to the repository root.
- Ensure contributions remain compatible with the project license in `License.md` (CC BY-NC 4.0).
