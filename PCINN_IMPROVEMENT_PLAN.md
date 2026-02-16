# PCINN Notebook Improvement Plan (Updated)

**Target notebook:** `MMA_PCINN.ipynb`  
**Goal:** Improve code quality, reliability, and PyTorch best-practice alignment **without changing the paper-baseline training setup** unless explicitly marked as an optional deviation.  
**Preservation strategy:** All improved training loops (Baseline NN and PCINN) should be implemented in **new, separate cells** below their original counterparts. The original cells from the paper should be kept intact (but not executed) so that the paper-baseline implementation remains available for direct comparison and reproducibility verification.  
**Baseline constraints to preserve by default:**
- Optimizer: `Adam`
- Loss: `MSELoss`
- `lr = 3e-4`
- `epochs = 10000`
- `totaljacsamples = 32` (Jacobian samples per epoch)
- Leave-one-experiment-out split via `TestReaction ∈ {1..8}`

---

## A. Mistakes / Corrections vs the Original Plan

### A1 — **Do NOT wrap `vmap(jacrev(...))` in `torch.no_grad()`**
- `torch.no_grad()` disables autograd tracking for computations inside the context; by design, outputs will have `requires_grad=False` even if inputs require grad.
- `torch.func.jacrev` uses **reverse-mode autodiff** under the hood; if you disable grad tracking, Jacobian computation will either fail or silently return incorrect behavior.

**Correct pattern for the theory Jacobian:**
- Freeze theory **parameters** (`requires_grad_(False)`), but keep grad mode **enabled** so Jacobians w.r.t. inputs can be computed.
- Then explicitly `.detach()` the theory Jacobian before using it as an MSE target.

### A2 — `_pickle` import is unnecessary (and undesirable)
- `pickle.UnpicklingError` is the same class as `_pickle.UnpicklingError` in CPython.
- Prefer catching `pickle.UnpicklingError` only; do not import `_pickle`.

### A3 — Best-model tracking: **a shallow `state_dict().copy()` is NOT a snapshot**
- `state_dict()` tensors share storage with the module parameters; without cloning/deepcopy, later training updates can mutate the “saved” best weights.
- Use `copy.deepcopy(model.state_dict())` or clone tensors explicitly.

### A4 — `KMP_DUPLICATE_LIB_OK` should not be presented as “Windows-only”
- OpenMP duplication can occur on macOS/Linux/Windows; the env-var is explicitly described as an **unsafe workaround** (can crash or produce incorrect results).
- If you keep it, (1) gate it behind a clear opt-in and (2) set it **before** importing libraries that load OpenMP runtimes.

---

## B. Implementation Plan (Step-by-Step)

### 0) Pre-flight: notebook structure & naming

**Recommendation:** stop referring to “Cell N” in the plan. Jupyter cell numbers shift constantly. Instead, anchor each change under the notebook’s existing markdown headings:

- `## Model Definitions`
- `## Data Preprocessing`
- `## Jacobian Sampling Configuration`
- `## Training Configuration`
- `## Baseline Neural Network Training`
- `## PCINN Training`
- `## Results Summary`

Where you need new material, add a new heading (e.g., `## Reproducibility & Device`).

---

## 1) Imports, environment, reproducibility, device

### 1.1 Consolidate imports (single place)
Move *all* imports to the top import cell (currently the `%matplotlib inline` + imports cell).

**Key decisions:**
- Remove `_pickle`.
- Keep `pickle` only if you keep the `weights_only` fallback logic.
- Add `copy` and `json` because they are used in the export cell.
- Only import `DataLoader` if you actually adopt it later.

Suggested import block:

```python
%matplotlib inline

from __future__ import annotations

import copy
import json
import os
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.func import jacrev, vmap
```

### 1.2 KMP_DUPLICATE_LIB_OK: remove by default; keep only as an explicit opt-in
**Default:** delete `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`.

If you absolutely need it to run on a specific machine, implement:

```python
# UNSAFE WORKAROUND (use only if you understand the risk):
# Can crash or silently produce incorrect results if multiple OpenMP runtimes are loaded.
ENABLE_UNSAFE_KMP_WORKAROUND = False

if ENABLE_UNSAFE_KMP_WORKAROUND:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

**Important:** if enabled, it should be set before importing NumPy/PyTorch in a fresh kernel.

### 1.3 Reproducibility cell (recommended, not “guaranteed determinism”)
Add a cell *immediately after imports*:

```python
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Optional strictness (may slow things down / error on nondeterministic ops):
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False
```

### 1.4 Device abstraction (always)
Add:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
```

---

## 2) Models: documentation, device placement, mode discipline

### 2.0 Modernize `super()` calls
If the notebook uses the legacy form `super(NNmodel, self).__init__()`, replace it with the Python 3 idiom:

```python
super().__init__()
```

This has identical behavior but is clearer and less error-prone.

### 2.1 Add docstrings + type hints
Add docstrings describing:
- input shape `(batch, 5)`
- output shape `(batch, 6)`
- what each output means (X + log10 molar masses)

Add `forward(self, x: torch.Tensor) -> torch.Tensor`.

### 2.2 Instantiate on device and set modes explicitly
- Baseline model: `model = NNmodel().to(DEVICE)`
- PCINN model: `PCINNmodel = NNmodel().to(DEVICE)`
- Theory model: `Domain_NN = DomainModel().to(DEVICE)`

Always call:
- `model.train()` inside training loop
- `model.eval()` inside evaluation blocks

---

### 2.3 Optional model-architecture upgrades (recommended to evaluate via ablations)

Your current **PCINN** predictor is a plain MLP:

- `NNmodel`: `5 → 128 → 64 → 6` with `tanh` activations
- `DomainModel` (theory net): shared trunk + two heads:
  - `X` head: `sigmoid` to constrain conversion in `[0, 1]`
  - `M` head: `softplus` to constrain molar masses to be positive

That baseline is fine for a class project, but **Jacobian-regularized training is unusually sensitive to architecture**. The following upgrades are the most defensible changes (in descending order of expected ROI).

#### 2.3.1 Make PCINN outputs physically consistent (X bounded, M positive) *without sacrificing log-space training*
Right now, `Y[:, 1:]` is trained in `log10` space, but the PCINN network outputs are unconstrained. A more physically coherent design is:

- predict `X` via a **logit head** (unconstrained scalar) → `sigmoid`
- predict `M` via **positive head** → `softplus` (or `exp`) and then compute `log10(M)` for the data-loss only

This gives you:
- physical outputs for Jacobian matching (if your theory model outputs physical masses)
- log-space regression for the observed data (which often stabilizes learning across wide dynamic ranges)

Implementation pattern (recommended):

- Add `forward_physical(x)` → returns `[X, M1..M5]` where `M > 0`
- Keep `forward(x)` returning `[X, log10(M1)..log10(M5)]` (loss-space output)

Then:
- **data loss** uses `model(x)` vs `Y`
- **jacobian loss** uses `model.forward_physical(x)` vs `theory(x)` (both in physical units)

#### 2.3.2 Turn PCINN into a *residual correction* of the theory model (strongly recommended)
Currently, the PCINN is trained *to imitate data*, with an extra Jacobian penalty to match the theory model’s derivatives. A more standard “physics-guided” pattern is:

\[
\hat{y}(x) = y_{\text{theory}}(x) + \Delta_\theta(x)
\]

Where:
- `y_theory(x)` is frozen (your pretrained `Domain_NN`)
- `Δθ(x)` is a **smaller** neural correction model

Advantages:
- makes optimization easier (the NN learns “what the theory misses”)
- reduces function space the correction must represent
- tends to generalize better out-of-distribution

Two common variants:
1. **Residual model:** `y = theory(x) + delta(x)`  
2. **Hybrid-input residual:** `y = theory(x) + delta(concat(x, theory(x)))` (delta sees theory outputs too)

If you do this, you typically reduce `delta` capacity (e.g., `5 → 64 → 64 → 6`) and add mild regularization (weight decay or spectral norm) so the correction stays small unless needed.

#### 2.3.3 Use residual blocks + LayerNorm for deeper, more trainable MLPs (moderate ROI)
If you need more capacity than `128/64`, prefer *depth with residual connections* over simply widening layers. A practical template:

- `Linear → LayerNorm → activation → Linear → LayerNorm → activation`
- add a skip connection (`x + f(x)`) when widths match

This can improve stability and gradient flow in deeper nets.

BatchNorm is usually less attractive here because you are not training with large, stable batch sizes and Jacobian losses can amplify batch-statistics noise; LayerNorm does not rely on running statistics.

#### 2.3.4 If you see “spectral bias” or poor derivative fidelity, evaluate Fourier features or SIREN (experimental)
For low-dimensional inputs (you have 5), MLPs can struggle to learn higher-frequency structure and derivatives. Two architecture options that are commonly used for better derivative behavior:

- **Fourier feature mapping** (a positional encoding layer before the MLP)
- **SIREN / sinusoidal activations** (requires specialized initialization)

These are not “free wins” and should be treated as experiments behind a feature flag.

#### 2.3.5 Initialization must match activations
If you change activations:
- `tanh` → Xavier/Glorot init (with appropriate gain)
- `relu`/`silu` → Kaiming/He init
- `sine` (SIREN) → use the SIREN paper’s initialization recipe

Do *not* rely on default `nn.Linear` init once you start doing Jacobian-sensitive training.

## 3) Data preprocessing: safer scaling + consistent tensor conversion

### 3.1 Make scaling robust
Update `scalefeaturezeroone`:

```python
def scalefeaturezeroone(
    x: np.ndarray,
    scalerxmax: np.ndarray,
    scalerxmin: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Min-max scale features to [0, 1] with numerical safety."""
    denom = np.maximum(scalerxmax - scalerxmin, eps)
    return (x - scalerxmin) / denom
```

### 3.2 Keep NumPy scalers for preprocessing; create Torch scalers for training
```python
scalerx_max_np = np.load("scalerx_max.npy")
scalerx_min_np = np.load("scalerx_min.npy")

SCALERX_MAX = torch.from_numpy(scalerx_max_np).float().to(DEVICE)
SCALERX_MIN = torch.from_numpy(scalerx_min_np).float().to(DEVICE)
SCALERX_DENOM = torch.clamp(SCALERX_MAX - SCALERX_MIN, min=1e-12)
```

Use `*_np` for the one-time `Xdata` preprocessing, and `SCALERX_*` inside the Jacobian sampling loop.

### 3.3 Convert train/test splits once, keep them as tensors
Change the train/test split section to produce tensors **only**:

```python
Xtrainsample = torch.from_numpy(Xdata[Xdata[:, 5] != TestReaction, :5]).float().to(DEVICE)
Ytrainsample = torch.from_numpy(Ydata[Xdata[:, 5] != TestReaction]).float().to(DEVICE)

Xtestsample = torch.from_numpy(Xdata[Xdata[:, 5] == TestReaction, :5]).float().to(DEVICE)
Ytestsample = torch.from_numpy(Ydata[Xdata[:, 5] == TestReaction]).float().to(DEVICE)
```

This removes repeated `torch.from_numpy(...)` inside loops.

---

## 4) Loading the pretrained theory model: safer + cleaner exception handling

### 4.1 Prefer weights-only loading; fall back only if necessary
Keep this, but remove `_pickle`:

```python
Domain_NN = DomainModel().to(DEVICE)

try:
    Domain_NN.load_state_dict(torch.load("MMA_solution_net.pt", weights_only=True, map_location=DEVICE))
except pickle.UnpicklingError:
    Domain_NN.load_state_dict(torch.load("MMA_solution_net.pt", weights_only=False, map_location=DEVICE))

Domain_NN.eval()
for p in Domain_NN.parameters():
    p.requires_grad_(False)
```

---

## 5) Baseline training loop: correctness + clarity (no baseline deviation)
> **Implementation note:** Create this improved loop in a **new cell** below the original baseline training cell. Keep the original cell intact (optionally marked with a `# [ORIGINAL — paper baseline, do not execute]` comment) so the paper-baseline code remains available for comparison.


### 5.1 Remove the “epoch 0 skip”
Use the canonical pattern:

```python
model = NNmodel().to(DEVICE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 10_000
train_losses: list[float] = []
test_losses: list[float] = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    pred = model(Xtrainsample)
    trainloss = loss_function(pred, Ytrainsample)
    trainloss.backward()
    optimizer.step()

    train_losses.append(trainloss.item())

    model.eval()
    with torch.inference_mode():
        test_pred = model(Xtestsample)
        testloss = loss_function(test_pred, Ytestsample).item()
    test_losses.append(testloss)

    if (epoch + 1) % 500 == 0:
        print(f"[Baseline] epoch={epoch+1}/{EPOCHS} train={train_losses[-1]:.6g} test={test_losses[-1]:.6g}")
```

---

## 6) PCINN training loop: **fix Jacobian sampling correctly**
> **Implementation note:** As with the baseline loop (Section 5), create this improved PCINN loop in a **new cell** below the original PCINN training cell. The original cell should remain intact for paper-baseline comparison.


### 6.1 Fix NumPy/Torch interop warning by using Torch scalers
Replace:

```python
sampl = (sampl - scalerx_min) / (scalerx_max - scalerx_min)
```

with:

```python
sampl = (sampl - SCALERX_MIN) / SCALERX_DENOM
```

### 6.2 Ensure Jacobian inputs can be differentiated
Before calling `jacrev`, ensure `sampl` is floating and requires grad:

```python
sampl = sampl.to(DEVICE).float().requires_grad_(True)
```

### 6.3 Compute theory Jacobian **with grad enabled** but frozen params
**Do not** use `torch.no_grad()` here.

```python
jac_theory_sampl = vmap(jacrev(Domain_NN))(sampl)
jac_theory_sampl = jac_theory_sampl.detach()  # target
```

### 6.4 Compute PCINN Jacobian normally (requires grad so jacloss backprops into PCINN params)
```python
jac_sampl = vmap(jacrev(PCINNmodel))(sampl)
jacloss = loss_function(jac_sampl, jac_theory_sampl)
```

### 6.5 Full PCINN loop (baseline-equivalent)
```python
PCINNmodel = NNmodel().to(DEVICE)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(PCINNmodel.parameters(), lr=3e-4)

EPOCHS = 10_000
TOTAL_JAC_SAMPLES = 32

train_losses: list[float] = []
test_losses: list[float] = []
jac_losses: list[float] = []

for epoch in range(EPOCHS):
    PCINNmodel.train()
    optimizer.zero_grad(set_to_none=True)

    pred = PCINNmodel(Xtrainsample)
    trainloss = loss_function(pred, Ytrainsample)

    # --- Jacobian sampling ---
    # NOTE: torch.distributions samplers may produce CPU tensors depending on how they were constructed.
    # We explicitly move the concatenated sample to DEVICE to avoid device-mismatch when normalizing with
    # SCALERX_MIN / SCALERX_DENOM (which live on DEVICE).
    Msample = M_sampler.sample((TOTAL_JAC_SAMPLES, 1))
    Ssample = 10 - Msample
    Isample = I_sampler.sample((TOTAL_JAC_SAMPLES, 1))
    Tsample = T_sampler.sample((TOTAL_JAC_SAMPLES, 1))
    tsample = time_sampler.sample((TOTAL_JAC_SAMPLES, 1))

    sampl = torch.cat((Msample, Ssample, Isample, Tsample, tsample), dim=1).to(DEVICE)
    sampl = (sampl - SCALERX_MIN) / SCALERX_DENOM

    # IMPORTANT (subtle): `requires_grad_()` can only be enabled on a leaf tensor.
    # At this point, `sampl` is a leaf because none of the operands above required grad,
    # so PyTorch did not build an autograd graph. We `detach()` anyway to make leaf-ness explicit.
    sampl = sampl.float().detach().requires_grad_(True)

    # NOTE: `jacrev` / `vmap` are pure functional transforms; reusing the same `sampl` tensor
    # for both theory and PCINN Jacobians is intentional and safe (no shared-state mutation).
    jac_theory_sampl = vmap(jacrev(Domain_NN))(sampl).detach()
    jac_sampl = vmap(jacrev(PCINNmodel))(sampl)

    jacloss = loss_function(jac_sampl, jac_theory_sampl)

    loss = trainloss + jacloss
    loss.backward()
    optimizer.step()

    train_losses.append(trainloss.item())
    jac_losses.append(jacloss.item())

    PCINNmodel.eval()
    with torch.inference_mode():
        test_pred = PCINNmodel(Xtestsample)
        testloss = loss_function(test_pred, Ytestsample).item()
    test_losses.append(testloss)

    if (epoch + 1) % 500 == 0:
        print(
            f"[PCINN] epoch={epoch+1}/{EPOCHS} "
            f"obj={train_losses[-1]:.6g} jac={jac_losses[-1]:.6g} test={test_losses[-1]:.6g}"
        )
```

### 6.6 Optional: mitigate Jacobian memory spikes (NO baseline deviation)
If you run into memory issues, use `chunk_size` for `jacrev`:

```python
jac_theory_sampl = vmap(jacrev(Domain_NN, chunk_size=1))(sampl).detach()
jac_sampl = vmap(jacrev(PCINNmodel, chunk_size=1))(sampl)
```

---

### 6.7 Optional (recommended if you adopt Section 2.3.1/2.3.2): keep Jacobian loss in a *consistent output space*

**Problem to avoid:** if the theory model outputs *physical molar masses* but your data loss trains in `log10(M)`, then comparing Jacobians directly can become inconsistent unless both sides are expressed in the same output space.

Two clean options:

#### Option A — Compare Jacobians in physical space (preferred if theory outputs physical units)
- Ensure `PCINNmodel.forward_physical(x)` returns `[X, M1..M5]` with `M > 0`.
- Compute Jacobians for both:
  - `jac_theory = vmap(jacrev(lambda z: Domain_NN(z)))(sampl)`
  - `jac_pcinn = vmap(jacrev(lambda z: PCINNmodel.forward_physical(z)))(sampl)`
- `jacloss = mse(jac_pcinn, jac_theory)`

Data loss still uses `PCINNmodel(x)` which returns `[X, log10(M)]`.

#### Option B — Compare Jacobians in log space via the chain rule (use if you keep `PCINNmodel(x)` as the only forward)
If the theory model provides `M` in physical units, transform the theory Jacobian to the `log10`-space Jacobian:

For each mass output `Mi`:
\[
\frac{\partial \log_{10}(M_i)}{\partial x} = \frac{1}{M_i \ln(10)} \frac{\partial M_i}{\partial x}
\]

Implementation sketch:
- compute `M_theory = Domain_NN(sampl)[..., 1:]` and clamp with an epsilon
- scale `jac_theory[..., 1:, :]` accordingly before `mse(...)`

This option is more fiddly but avoids changing the model API.

---

## 7) Plotting: numerically safer log-loss curves + readable figures

Replace `np.log(losses)` with:

```python
def safe_log(x: list[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.log(np.clip(arr, eps, None))
```

Then, when plotting, add minimal labels/titles/grid for readability:

```python
plt.figure()
plt.plot(safe_log(train_losses), label="train (log)")
plt.plot(safe_log(test_losses), label="test (log)")
plt.plot(safe_log(jac_losses), label="jac (log)")

plt.xlabel("epoch")
plt.ylabel("log(loss)")
plt.title("PCINN training curves (log scale)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

---
## 8) NEW FINAL CELL: Export the trained PCINN model (and preprocessing artifacts)

### 8.1 What to export (minimum viable for real inference)
If someone loads the model later, they also need:
- `state_dict`
- `scalerx_min` / `scalerx_max` (or torch equivalents)
- the fact that `Y[:, 1:]` is `log10` transformed
- the fold (`TestReaction`) used

### 8.2 Export cell (robust, with verification)

```python
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

fold = int(TestReaction)

weights_path = EXPORT_DIR / f"pcinn_fold{fold}_weights.pt"
bundle_path = EXPORT_DIR / f"pcinn_fold{fold}_bundle.pt"
meta_path = EXPORT_DIR / f"pcinn_fold{fold}_meta.json"

PCINNmodel.eval()

# --- 1) weights only ---
torch.save(PCINNmodel.state_dict(), weights_path)

# --- 2) bundle: everything needed for inference ---
bundle = {
    "model_class": "NNmodel",
    "model_state_dict": copy.deepcopy(PCINNmodel.state_dict()),
    "scalerx_min": scalerx_min_np,
    "scalerx_max": scalerx_max_np,
    "y_log10_applied_to_columns_1_to_end": True,
    "fold": fold,
    "pytorch_version": torch.__version__,
}
torch.save(bundle, bundle_path)

# --- 3) small human-readable metadata ---
meta = {
    "fold": fold,
    "epochs": int(EPOCHS),
    "lr": 3e-4,
    "total_jac_samples": int(TOTAL_JAC_SAMPLES),
    "pytorch_version": torch.__version__,
}
meta_path.write_text(json.dumps(meta, indent=2))

print(f"Saved:\n- {weights_path}\n- {bundle_path}\n- {meta_path}")

# --- 4) smoke test reload ---
reloaded = NNmodel().to(DEVICE)
reloaded.load_state_dict(torch.load(weights_path, weights_only=True, map_location=DEVICE))
reloaded.eval()

with torch.inference_mode():
    a = PCINNmodel(Xtestsample)
    b = reloaded(Xtestsample)
max_diff = (a - b).abs().max().item()
assert max_diff < 1e-6, f"Reload mismatch: max diff={max_diff}"
print(f"Reload OK (max abs diff = {max_diff:.2e})")```

---

## 9) Optional deviations (clearly marked)

### 9.1 Best-model tracking (recommended, but deviation from “final epoch”)
If you choose to export the *best test-loss* weights instead of final epoch:

```python
# `copy` is already imported in the consolidated import block (Section 1.1).
best_state: dict[str, torch.Tensor] | None = None
best_test = float("inf")

...
if testloss < best_test:
    best_test = testloss
    best_state = copy.deepcopy(PCINNmodel.state_dict())
...
PCINNmodel.load_state_dict(best_state)
```

### 9.2 `torch.compile` (high risk / experimental here)
`torch.compile` is not guaranteed to work cleanly with `vmap(jacrev(...))` compositions and can complicate debugging. Treat as a separate experiment.

---

## 10) Implementation checklist

**Imports & Environment (Section 1)**
- [x] Consolidate all imports into a single top-level cell; remove `_pickle` and unused imports.
- [x] Remove `KMP_DUPLICATE_LIB_OK` default; add opt-in switch if needed.
- [x] Add `SEED` + device cell (`DEVICE`) near the top.

**Models (Section 2)**
- [x] Modernize `super()` calls to Python 3 idiom.
- [x] Add docstrings + type hints to `NNmodel`, `DomainModel`, and `scalefeaturezeroone`.
- [x] Move all model instantiations to `DEVICE`.

**Data & Preprocessing (Section 3)**
- [x] Make `scalefeaturezeroone` numerically safe (epsilon denominator).
- [x] Create `SCALERX_MAX`, `SCALERX_MIN`, `SCALERX_DENOM` as Torch tensors on `DEVICE`.
- [x] Convert train/test splits to tensors on `DEVICE` once, not every epoch.

**Theory Model Loading (Section 4)**
- [x] Use `pickle.UnpicklingError` only (not `_pickle`); add `map_location=DEVICE`.
- [x] Freeze theory model: `Domain_NN.eval()` + `requires_grad_(False)` on all parameters.

**Baseline Training Loop (Section 5)**
- [x] Create improved loop in a **new cell** below the original (preserve original for comparison).
- [x] Fix epoch-0 skip; use canonical `zero_grad → forward → backward → step` order.
- [x] Add `model.train()`/`model.eval()` mode discipline.
- [x] Wrap test evaluation in `torch.inference_mode()`.
- [x] Use `.item()` for scalar extraction; remove redundant accumulators.

**PCINN Training Loop (Section 6)**
- [x] Create improved loop in a **new cell** below the original (preserve original for comparison).
- [x] Replace NumPy scalers with Torch `SCALERX_MIN`/`SCALERX_DENOM` in Jacobian sampling.
- [x] Ensure `sampl` is `.float().detach().requires_grad_(True)` before `jacrev`.
- [x] **No** `no_grad()` around Jacobian computation; `.detach()` the theory Jacobian target instead.
- [x] Add `PCINNmodel.train()`/`PCINNmodel.eval()` mode discipline + `inference_mode()` for test.
- [x] Add periodic stdout logging every N epochs.

**Plotting (Section 7)**
- [x] Use `safe_log()` for numerically safe log-loss curves.
- [x] Add axis labels, titles, grid, and legend to all plots.

**Model Export (Section 8)**
- [x] Add export cell: weights-only `.pt`, inference bundle, human-readable JSON metadata.
- [x] Add smoke-test reload with `max_diff < 1e-6` assertion.

**Optional (Sections 2.3, 6.7, 9)**
- [x] Evaluate architecture upgrades (2.3) via ablation if time permits. *(Implemented as PCINNv2: residual correction + physical output heads + ResidualBlock/LayerNorm + optional Fourier features + Xavier init)*
- [x] Verify Jacobian output-space consistency if architecture changes are adopted (6.7). *(PCINNv2 uses physical-space Jacobian matching via `forward_physical()`)*
- [ ] Consider best-model tracking with `copy.deepcopy` (9.1). *(Skipped per user decision — keeping final-epoch weights)*

---

## References (official PyTorch docs)

- `torch.no_grad`: disables gradient calculation; results have `requires_grad=False`.  
  https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
- `torch.func.jacrev`: computes Jacobian via reverse-mode autodiff.  
  https://docs.pytorch.org/docs/2.9/generated/torch.func.jacrev.html
- Leaf vs non-leaf tensors + `requires_grad_()` behavior:  
  https://docs.pytorch.org/tutorials/beginner/understanding_leaf_vs_nonleaf_tutorial.html  
  https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.is_leaf.html
- Serialization + `weights_only=True` best practices and troubleshooting:  
  https://docs.pytorch.org/docs/stable/notes/serialization.html  
  https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html
- Reproducibility / determinism notes:  
  https://docs.pytorch.org/docs/stable/notes/randomness.html

## References (architecture / PINN literature)

- Residual networks (skip connections) for trainability: He et al., *Deep Residual Learning for Image Recognition* (CVPR 2016).
- LayerNorm: Ba et al., *Layer Normalization* (2016) + PyTorch `nn.LayerNorm` docs.
- Fourier feature mappings to reduce spectral bias in low-dimensional regression: Tancik et al., *Fourier Features Let Networks Learn High Frequency Functions* (NeurIPS 2020).
- Sinusoidal Representation Networks (SIREN) for representing signals and derivatives: Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions* (NeurIPS 2020).
- Gradient-enhanced PINNs (adding gradient/Jacobian terms to the loss): Yu et al., *Gradient-enhanced physics-informed neural networks* (CMA 2022).
- Physics-guided neural network “residual model” and “hybrid-input residual model” structures in reaction/flow modeling: RSC *Reaction Chemistry & Engineering* (2025), “Neural tanks-in-series…”.

(Keep raw paper links in the notebook/README if you want; the plan avoids hardcoding them inline.)
