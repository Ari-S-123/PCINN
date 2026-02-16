# Polymer Chemistry Informed Neural Networks (PCINNs) for data-driven modelling of polymerization processes
This repository contains code for the implementation of Polymer Chemistry Informed Neural Networks (PCINNs), a method that combines kinetic models of polymerization processes with neural network training to improve predictive accuracy. The approach mitigates common challenges, such as data limitations and poor extrapolation, by leveraging domain knowledge to enhance model reliability.
More details are described in [our paper](https://doi.org/10.1039/D4PY00995A).

Primary implementation notebook: `MMA_PCINN.ipynb`  
Upstream repository: https://github.com/PolymatGIQ/PCINN

## Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install torch==2.10.0 numpy==2.4.2 pandas==2.2.3 matplotlib==3.10.8 openpyxl==3.1.5 "ipykernel>=6.29,<7"
```

Open `MMA_PCINN.ipynb` in Jupyter or VS Code and run all cells. The notebook trains a baseline neural network and a PCINN, then plots train/test loss curves for both.

The original script `MMA_PCINN.py` is retained as a reference.

## Notebook Notes

- The notebook preserves the original model/training logic from `MMA_PCINN.py`, with only compatibility updates for modern PyTorch:
  - `F.tanh` -> `torch.tanh`
  - `F.sigmoid` -> `torch.sigmoid`
  - `torch.load(..., weights_only=...)` for pretrained checkpoint loading
- Checkpoint loading behavior:
  - It first tries `weights_only=True` (secure default in PyTorch 2.6+).
  - If the checkpoint cannot be read in weights-only mode (`pickle.UnpicklingError`), it falls back to `weights_only=False` because the checkpoint provenance is controlled in this repository.
- Default cross-validation setting is `TestReaction = 8`. To reproduce full leave-one-experiment-out results, rerun with `TestReaction` in `1..8`.

## Validation

- Restart kernel and run all notebook cells top-to-bottom.
- Both training sections should execute and render inline train/test loss plots.
- `NNpred` and `EBNNpred` should each be shape `(N, 6)` for the held-out reaction.

## LICENSE
This repository is licensed under CC BY-NC 4.0.
For more information please refer to the [license section](https://github.com/PolymatGIQ/PCINN/blob/main/License.md).
