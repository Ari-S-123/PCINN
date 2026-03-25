# Single-Notebook Implementation Plan for New Polymer Kinetics Work
## End-to-End Jupyter Notebook Only — No `src/`, No Multi-File Refactor, One Top Setup Cell

**Audience:** OpenAI Codex coding agent  
**Primary objective:** Create **one new Jupyter notebook** that can be run **top-to-bottom, end-to-end** to ingest the new polymerization data, harmonize it, validate it, perform exploratory diagnostics including PCA, construct leakage-safe grouped splits, and train **teacher-free baseline models**.  
**Critical user constraint:** **Do not create a `src/` directory, package, module tree, helper `.py` files, or any other code organization outside the notebook itself.**  
**Critical notebook constraint:** **All package installation statements and all imports must appear in a single cell at the very top of the notebook.**

---

## 1. Executive Summary

Build exactly **one new notebook**. That notebook must contain **all new work** required for the current phase.

The notebook must:

1. install and import everything from a **single setup cell at the top**;
2. verify the uploaded files exist;
3. ingest all relevant spreadsheets;
4. convert heterogeneous spreadsheet layouts into one canonical long-form dataset;
5. enrich rows with regime and source metadata;
6. normalize units and targets;
7. run quality control and validation checks;
8. build leakage-safe grouped splits;
9. train teacher-free baseline models;
10. include PCA as a **section inside the same notebook**;
11. export cleaned datasets, split assignments, metrics, and figures.

This notebook is **not** a full new PCINN replication. It is the correct intermediate deliverable to make immediate progress **without** generating synthetic theory data or training a new surrogate teacher model.

---

## 2. Prompting and Codex Execution Principles

The notebook build request that Codex receives should stay explicit, constrained, and structured. OpenAI’s official prompting guidance recommends separating broad role guidance from task-specific instructions, using clear structure, and keeping prompts simple and direct; OpenAI’s reasoning-model guidance also favors providing the goal, constraints, and desired output while letting the model do the reasoning internally rather than asking for chain-of-thought. citeturn269561search0turn269561search2

Therefore, this plan is intentionally organized into:

- mission,
- hard constraints,
- required notebook structure,
- implementation steps,
- validation requirements,
- and acceptance criteria.

Codex should be instructed to follow this plan literally unless a minor adjustment is needed to make the notebook run correctly in the current environment.

---

## 3. Absolute Non-Negotiable Constraints

Codex must satisfy **all** of the following.

### 3.1 Single-notebook constraint
- Create **one** new Jupyter notebook only.
- Do **not** create a `src/` directory.
- Do **not** create helper Python modules.
- Do **not** create package files purely for organization.
- Do **not** split the work across multiple notebooks for this phase.
- All helper functions must live **inside the notebook itself**.

### 3.2 Single top setup cell constraint
- The **first code cell** in the notebook must be the **only** cell that installs packages and imports libraries.
- No later cell may perform additional `pip install`, `apt install`, or import blocks unless absolutely forced by runtime issues.
- If a package is optional and unavailable, handle it gracefully in that same top cell.
- The top setup cell should also print package versions for reproducibility where practical.

### 3.3 End-to-end execution constraint
- The notebook must run from top to bottom in order.
- A fresh kernel restart followed by **Run All** should be the expected usage mode.
- Each section must leave behind named variables needed by later sections.
- Avoid hidden state and manual intervention.

### 3.4 Scientific scope constraint
- The notebook must be **teacher-free**.
- Do **not** use the old MMA solution surrogate teacher on the new data.
- Do **not** include Jacobian matching unless a mechanism-matched teacher is actually available.
- The notebook must explicitly state that its purpose is harmonization, QC, exploratory diagnostics, grouped evaluation, and baseline modeling.

### 3.5 Split integrity constraint
- No random row-wise split across points from the same experimental curve.
- Splitting must be grouped at least by `curve_id`.
- Preserve `paper_id`, `source_file`, and `regime_family` wherever possible.

### 3.6 Robustness constraint
- Every ingestion and transformation step must validate assumptions.
- Errors must be explicit, actionable, and localizable.
- Functions must include type annotations and clear docstrings.
- Every important intermediate table should be inspectable in the notebook.

---

## 4. Existing Inputs the Notebook Must Handle

The notebook must account for the current uploaded context.

### 4.1 PCINN context files
- `MMA_PCINN.ipynb`
- `Original PCINN Paper.pdf`
- `PCA_Exploration.ipynb`

These provide context for prior work, variable conventions, and target framing, but the new notebook should not depend on importing code from them.

### 4.2 New spreadsheet inputs
- `MMA_fig1a.xlsx`
- `MMA_fig2a.xlsx`
- `MMA_fig2b_Mn.xlsx`
- `MMA_fig2b_PDI.xlsx`
- `STY_fig1b.xlsx`
- `STY_fig2c.xlsx`
- `STY_fig2d_Mn.xlsx`
- `STY_fig2d_PDI.xlsx`
- `PSBMA NMR.xlsx`

### 4.3 Known heterogeneity in the new data
The notebook must assume the spreadsheets may differ in:

- sheet names,
- row offsets,
- header formatting,
- wide vs long layout,
- units for time,
- `t` vs `t_res`,
- raw conversion vs percent conversion,
- transformed kinetics such as `-ln(1-conversion)`,
- `conversion` vs `Mn`,
- `conversion` vs `PDI`,
- and manually digitized precision noise.

Therefore, the ingestion pipeline must be **schema-aware and defensive**, not naive.

---

## 5. Final Deliverable Definition

Codex must create a notebook with a name like:

- `polymer_new_data_end_to_end.ipynb`

The name can differ slightly, but it must clearly communicate that it is:
- for the new polymer data,
- end-to-end,
- and self-contained.

The notebook must also create an output directory such as:

- `artifacts/polymer_new_data_end_to_end/`

This output directory may be created by notebook code, but **the notebook itself remains the only source file** created for the workflow.

---

## 6. Required Notebook Section Structure

The notebook must contain markdown headings and code cells organized in the following order.

### Section 1 — Title, scope, and explicit limitations
Explain:
- what the notebook does,
- what it does not do,
- why it is teacher-free,
- why all new work is intentionally kept in one notebook for now.

### Section 2 — Single setup cell
This must be the **first code cell**.
It must:
- install required packages if needed,
- import all libraries,
- set notebook-wide display options,
- define reproducibility seeds,
- and print package versions.

### Section 3 — File inventory and path verification
Verify all expected uploaded files exist.
Print a clear inventory table.
Fail early if critical files are missing.

### Section 4 — Configuration block
Define all notebook-wide constants in one place, for example:
- random seed,
- artifact output path,
- expected file paths,
- canonical column names,
- regime labels,
- subset names,
- split parameters,
- model hyperparameters.

### Section 5 — Utility functions
Define all notebook helper functions here.
Examples:
- file-loading helpers,
- header-cleaning helpers,
- wide-to-long converters,
- value coercion and parsing helpers,
- metadata attachers,
- validators,
- plot helpers,
- grouped split helpers,
- baseline model trainers.

All helpers must remain inside this notebook.

### Section 6 — Raw spreadsheet inspection
Load each spreadsheet in a minimally processed form first.
Display enough rows/metadata to understand layout.
Generate a concise inspection summary for each source file.

### Section 7 — Source-specific ingestion
Convert each source spreadsheet into a canonical intermediate DataFrame.
This section may have one sub-block per file or per family of similar files.

### Section 8 — Canonical harmonization
Concatenate all intermediate tables into one unified long-form dataset with a shared schema.

### Section 9 — Metadata enrichment
Attach domain metadata such as:
- monomer,
- regime family,
- measurement type,
- paper/source group,
- target type,
- file provenance,
- condition labels,
- temperature.

### Section 10 — Target normalization and derived variables
Standardize:
- time units,
- conversion representation,
- log transforms if used,
- molecular-weight derived values,
- consistency checks between related variables.

### Section 11 — Quality control and validation
Run all validation checks and produce a QC report.

### Section 12 — Analysis subset construction
Define clean subsets for immediate modeling.
At least one subset should be narrow and coherent enough for teacher-free baselines.

### Section 13 — Grouped split construction
Create leakage-safe CV and/or train/validation/test structures grouped by experimental curve.

### Section 14 — Teacher-free baseline modeling
Train simple baseline models on at least one coherent subset.

### Section 15 — PCA diagnostics inside the same notebook
Run PCA only as a diagnostic section within this notebook.
Do not split PCA into a separate notebook in this phase.

### Section 16 — Final exports and summary
Write cleaned data, split assignments, metrics, and figures to disk.
Summarize what is now ready for later teacher-based work.

---

## 7. Exact Requirements for the Single Top Setup Cell

The top setup cell is critical.

### 7.1 What must be installed/imported there
The top cell should, as needed, install/import the standard stack for this work, likely including:
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `openpyxl`
- `scipy`
- optionally `seaborn` only if truly needed, though plain matplotlib/pandas plotting is acceptable
- optionally `xgboost` only if already available and clearly justified

Do **not** bloat the environment with unnecessary packages.

### 7.2 What else the top cell must do
The top cell should also:
- import `Path` from `pathlib`;
- import `json`, `math`, `re`, `warnings`, `itertools`, `typing`, and any other standard libraries needed;
- set a global random seed for Python and NumPy;
- silence only noncritical warnings;
- configure pandas display options;
- print version information for key packages.

### 7.3 Practical implementation guidance
Because installations may be environment-dependent, the top cell may use guarded notebook-style install commands. The code should be written so that if packages are already installed, the cell still succeeds cleanly.

The top cell should be clearly labeled in markdown immediately above it as:

- **Setup: install packages, import libraries, configure environment**

No other cell in the notebook should repeat package installation or import boilerplate.

---

## 8. Canonical Schema the Notebook Must Build

The notebook must produce one harmonized long-form table with a schema close to the following.

### 8.1 Identifier columns
- `row_id`
- `source_file`
- `source_sheet`
- `paper_id`
- `curve_id`
- `condition_id`
- `regime_family`
- `subset_name`

### 8.2 Experimental-condition columns
- `monomer`
- `temperature_C`
- `time_value_raw`
- `time_unit_raw`
- `time_min`
- `residence_time_min`
- `solvent`
- `initiator`
- `cta`
- `reactor_type`
- `atmosphere`
- `pressure_bar`
- `light_wavelength_nm`
- `light_intensity_mw_cm2`

### 8.3 Target/value columns
- `measurement_type`
- `target_raw`
- `target_raw_unit`
- `conversion_fraction`
- `conversion_percent`
- `neg_log_one_minus_conversion`
- `Mn`
- `PDI`
- `Mw_derived`
- `target_value_final`
- `target_name_final`

### 8.4 Provenance / QC columns
- `digitized_from_figure`
- `notes`
- `ingestion_warning`
- `validation_status`
- `is_duplicate_suspected`
- `is_out_of_range`

Not every column will be available for every row. Missingness is expected, but it must be explicit and interpretable.

---

## 9. Required Ingestion Strategy

Codex must not assume that each spreadsheet has the same layout. The notebook should implement a staged ingestion strategy.

### Step 9.1 — Read workbook metadata safely
For every Excel file:
- enumerate sheet names,
- attempt to load each sheet,
- record raw shape,
- inspect non-empty rows/columns,
- and summarize the likely header row.

### Step 9.2 — Normalize raw headers
Implement helpers to:
- strip whitespace,
- collapse repeated spaces,
- normalize Unicode oddities,
- standardize case where helpful,
- and preserve enough original naming for traceability.

### Step 9.3 — Detect layout families
At minimum, support the following layout patterns:
- wide time-series by temperature,
- long tables already close to canonical,
- conversion-vs-Mn tables,
- conversion-vs-PDI tables,
- NMR conversion tables.

### Step 9.4 — Convert each layout to canonical intermediate form
For each detected layout, create a clean intermediate DataFrame with:
- one row per observation,
- explicit identifiers,
- explicit measurement type,
- explicit condition metadata inferable from file context.

### Step 9.5 — Preserve provenance
Every row must retain:
- source filename,
- source sheet,
- and an interpretable condition label.

---

## 10. Metadata Enrichment Rules

The notebook must attach metadata from file identity and context.

### Minimum regime labels to support
Use labels along the lines of:
- `raft_photo_flow_mma`
- `raft_photo_flow_sty`
- `raft_bulk_mma`
- `raft_bulk_sty`
- `raft_emulsion_sty`
- `self_initiated_high_pressure_sty`
- `conversion_only_other`

The exact naming may vary slightly, but the notebook must not flatten all rows into one generic polymerization bucket.

### Required metadata logic
Codex must infer and/or assign, when justified by file identity:
- monomer family,
- whether the file is kinetic or molecular-weight oriented,
- whether time means elapsed time or residence time,
- whether the source is likely flow, bulk, emulsion, or other,
- whether the row belongs to a modeling-ready subset.

If a value cannot be justified, leave it missing and document that uncertainty.

---

## 11. Target Normalization Rules

This section is central.

### 11.1 Time normalization
Convert all time-like variables to a common numeric basis where possible:
- keep raw time,
- keep raw unit,
- derive `time_min` when valid.

If the quantity is actually residence time, keep both:
- `residence_time_min`
- and, if useful, `time_min`

but do not conflate them silently.

### 11.2 Conversion normalization
Support these forms:
- raw conversion fraction,
- conversion percent,
- `-ln(1-conversion)`.

Where valid, derive missing representations from the others.
For example, if `y = -ln(1-X)`, derive:
- `X = 1 - exp(-y)`.

Validation must ensure the resulting `X` lies in `[0, 1]` up to small numeric tolerance.

### 11.3 Molecular-weight normalization
Where `Mn` and `PDI` are both available, optionally derive:
- `Mw_derived = Mn * PDI`

but only if units and meaning are consistent.

### 11.4 Transform choices for modeling
The notebook should evaluate whether to model:
- raw conversion,
- transformed conversion,
- `log(Mn)`,
- and `log(PDI)` or raw `PDI`.

Document these choices clearly.

---

## 12. Quality Control and Validation Requirements

The notebook must implement explicit QC checks.

### 12.1 File-level QC
- file exists,
- workbook readable,
- sheet count known,
- no silent sheet failures.

### 12.2 Schema-level QC
- required identifier columns populated,
- numeric fields coercible where expected,
- measurement types assigned,
- no catastrophic column collapse after harmonization.

### 12.3 Value-range QC
Check for impossible or suspicious values such as:
- conversion < 0,
- conversion > 1 or > 100 when representation is unclear,
- negative Mn,
- PDI < 1 where inappropriate,
- negative time,
- missing temperature on temperature-indexed curves.

### 12.4 Duplicate / leakage QC
- duplicate rows,
- near-duplicate rows,
- repeated curve identifiers,
- split leakage after grouped splitting.

### 12.5 QC outputs
Produce:
- summary tables,
- warning lists,
- and human-readable markdown notes in the notebook.

Do not hide problems.

---

## 13. Analysis Subset Strategy

Do not force all data into one baseline model immediately.

### Required subset plan
The notebook must define at least:

#### Subset A — Primary modeling subset
A narrow, coherent subset for immediate baseline modeling, preferably one of:
- MMA flow photopolymerization only, or
- styrene flow photopolymerization only.

#### Subset B — Secondary exploratory subset
A slightly broader subset for descriptive analysis only.

#### Subset C — Full harmonized dataset
Used for QC, PCA diagnostics, and inventory, but not necessarily for a single pooled predictive model.

### Why this is required
The new data spans multiple mechanism families. Pooling everything immediately would blur mechanistic differences and create misleading performance estimates.

---

## 14. Grouped Split Construction Requirements

The notebook must implement grouped evaluation, not naive row-wise splitting.

### Minimum requirement
Use `GroupKFold`-style or equivalent grouping by `curve_id`. Group-aware cross-validation is the correct mechanism when multiple rows belong to the same higher-level unit. citeturn269561search3

### Preferred grouping hierarchy
If feasible, preserve multiple grouping levels conceptually:
- `curve_id` for immediate leakage prevention,
- `paper_id` for broader robustness diagnostics,
- `regime_family` for subset filtering.

### Required outputs
The notebook must save:
- per-row fold assignment,
- group membership summary,
- and a leakage check confirming no group appears in both train and validation/test within a split.

---

## 15. Baseline Modeling Requirements

The notebook must include teacher-free baselines only.

### 15.1 Required baseline classes
At minimum, implement:

#### Baseline 1 — Simple curve-level kinetic fit
For conversion-style subsets, fit a simple linear relation on `-ln(1-conversion)` vs time/residence time where appropriate.
This is an interpretability anchor, not just a predictive model.

#### Baseline 2 — Tabular ML baseline
Use one simple tabular regressor such as:
- linear regression,
- ridge regression,
- random forest,
- or gradient boosting,

chosen conservatively and justified.

#### Baseline 3 — Small neural baseline
Implement a modest feedforward neural network only if there is enough clean data in the chosen subset.
Keep it small and well-regularized.

### 15.2 Required targets
Start with targets actually supported by the chosen subset, such as:
- `conversion_fraction`,
- `neg_log_one_minus_conversion`,
- `log(Mn)`,
- `PDI` or `log(PDI)`.

Do not pretend the new data automatically supports the full six-output original PCINN target set.

### 15.3 Required evaluation outputs
Produce:
- fold-wise metrics,
- aggregate metrics,
- parity plots,
- residual plots,
- and curve overlay plots where meaningful.

---

## 16. PCA Requirements Inside the Same Notebook

Yes, PCA should be present, but only as one internal section.

### 16.1 Purpose of PCA here
PCA is for diagnostics, not for the main scientific claim. Use it to examine:
- whether source papers separate strongly,
- whether regime families separate strongly,
- whether the primary modeling subset is compact and coherent,
- whether scaling issues dominate variance.

scikit-learn’s PCA centers data but does **not** scale features automatically, so features must be standardized first when magnitudes differ substantially. scikit-learn’s preprocessing guidance likewise shows scaling can materially change downstream structure and interpretation. citeturn269561search0turn269561search1

### 16.2 Required PCA workflow
- choose a numeric feature view,
- handle missing values explicitly,
- standardize features,
- run PCA,
- plot at least 2D projections,
- color by regime family and source file,
- interpret whether pooling appears hazardous.

### 16.3 Do not overstate PCA
The notebook must not present PCA as proof of mechanistic truth. It is just a structural diagnostic.

---

## 17. Export Requirements

By the end of the notebook, save the following artifacts under the notebook-created output directory.

### Required saved files
- `harmonized_dataset.csv`
- `qc_summary.csv`
- `split_assignments.csv`
- `subset_summary.csv`
- `baseline_metrics.csv`
- figures for key QC and modeling plots
- PCA figures
- a compact JSON or text summary of important notebook outputs

The notebook may also save intermediate canonical tables if helpful, but do not scatter outputs randomly.

---

## 18. Documentation Requirements Inside the Notebook

The notebook must be readable by a human collaborator, not just executable.

### Required documentation style
- Each major section must begin with a markdown explanation.
- Each major code block must have a brief purpose note nearby.
- Functions must include docstrings with parameters, returns, and failure conditions.
- Assumptions must be stated explicitly.
- Known uncertainties from digitized literature extraction must be acknowledged.

---

## 19. Step-by-Step Execution Plan for Codex

Codex should implement the notebook in the following order.

### Step 1
Create the new notebook file.
Do not create `src/` or any separate Python modules.

### Step 2
Write the notebook title/overview markdown section explaining scope and limitations.

### Step 3
Create the single top setup cell.
This cell must contain all package installs, imports, environment configuration, and version printing.

### Step 4
Add a configuration section that defines all paths, constants, labels, seeds, and output locations.

### Step 5
Add file inventory logic that verifies the expected uploaded files are present before proceeding.

### Step 6
Implement notebook-local helper functions for:
- workbook introspection,
- header cleanup,
- parsing wide tables,
- canonical schema conversion,
- validation,
- plotting,
- grouped split generation,
- and baseline fitting.

### Step 7
Inspect each spreadsheet in a raw form and log its structure.
Do not skip this step.

### Step 8
Implement source-specific ingestion blocks and convert each source to an intermediate table.

### Step 9
Concatenate intermediate tables into one canonical harmonized DataFrame.

### Step 10
Attach metadata and normalize targets/units.

### Step 11
Run QC and produce both visual and tabular summaries.

### Step 12
Define analysis subsets, especially one narrow primary subset for immediate baseline work.

### Step 13
Construct grouped splits and verify there is no leakage.

### Step 14
Train teacher-free baselines and save metrics/plots.

### Step 15
Run PCA diagnostics in the same notebook on carefully prepared numeric views.

### Step 16
Export all artifacts and end with a markdown summary of findings and next steps.

---

## 20. Acceptance Criteria

The notebook is acceptable only if **all** of the following are true.

1. There is exactly **one** new notebook for this phase.
2. No `src/` directory or helper module tree was created.
3. All installs and imports live in a **single top code cell**.
4. The notebook runs top-to-bottom with reasonable expectation of success.
5. All uploaded spreadsheets are inventoried and handled explicitly.
6. A canonical harmonized dataset is produced.
7. QC checks run and surface problems clearly.
8. PCA exists inside the notebook, not as a separate first priority deliverable.
9. Group-aware splits are used.
10. Teacher-free baselines are trained on at least one coherent subset.
11. Clean artifacts are exported.
12. The notebook states clearly that this is the immediate bridge toward later teacher-based modeling, not a fake full PCINN recreation.

---

## 21. Final Instruction Block for Codex

Use this as the operational summary when generating the notebook:

> Create one self-contained Jupyter notebook for the new polymer data workflow. Do not create a `src/` directory, helper Python modules, or multiple notebooks for this phase. Put all package installation statements and all imports in a single first code cell at the top of the notebook. Make the notebook runnable end-to-end from a fresh kernel. The notebook must inventory the uploaded files, ingest and harmonize the new spreadsheets into a canonical long-form dataset, enrich metadata, normalize units and targets, run QC, define leakage-safe grouped splits, train teacher-free baseline models on at least one coherent subset, include PCA as a diagnostic section inside the same notebook, and export cleaned data, splits, metrics, and figures. Use robust validation, explicit docstrings, clear markdown explanations, and no hidden assumptions.

