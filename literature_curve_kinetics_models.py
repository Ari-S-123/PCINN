#!/usr/bin/env python3
"""
Standalone kinetics-curve fitting workflow.

This file is intentionally separate from the existing PCINN model code. It only
uses the extracted literature `.xlsx` files in `literature_curve_xlsx/` and
writes outputs into `literature_curve_artifacts/`.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "literature_curve_xlsx"
ARTIFACTS_DIR = ROOT / "literature_curve_artifacts"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"

MPLCONFIGDIR = ARTIFACTS_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
EPOCHS = 4000
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
TEMPERATURE_COLUMN = "Temp (Celsius)"
EXPECTED_Y_COLUMN = "-ln(1-conversion) (expected)"
ACTUAL_Y_COLUMN = "-ln(1-conversion) (actual)"
RESIDUAL_COLUMN = "Error"


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


class CurveMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_axis_label(label: str) -> str:
    return label.replace("\\", "").strip()


def format_temperature(value: float) -> str:
    return f"{value:.2f}"


def axis_label_to_column_name(label: str) -> str:
    label = normalize_axis_label(label)
    replacements = {
        "t_res": "t_res",
        "time(min)": "time_min",
        "time(h)": "time_h",
        "-ln(1-conversion)": "negln_1_minus_conversion",
    }
    if label in replacements:
        return replacements[label]

    sanitized = (
        label.replace("(", "_")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "_")
        .replace(".", "_")
    )
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_").lower()


def parse_curve_workbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    rows: list[dict[str, object]] = []
    for col_idx in range(0, raw.shape[1], 2):
        series_label = float(raw.columns[col_idx])
        x_label = normalize_axis_label(str(raw.iloc[0, col_idx]))
        y_label = normalize_axis_label(str(raw.iloc[0, col_idx + 1]))
        block = raw.iloc[1:, col_idx : col_idx + 2].copy()
        block.columns = ["x_value", "y_value"]
        block = block.dropna()
        # The digitized origin point adds no training signal for these curves.
        block = block[
            ~(np.isclose(block["x_value"], 0.0) & np.isclose(block["y_value"], 0.0))
        ]
        for row in block.itertuples(index=False):
            rows.append(
                {
                    "source_file": path.name,
                    "source_stem": path.stem,
                    "system": path.stem.split("_", 1)[0],
                    "series_label": series_label,
                    "x_label": x_label,
                    "y_label": y_label,
                    "x_value": float(row.x_value),
                    "y_value": float(row.y_value),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["series_label", "x_value"]
    ).reset_index(drop=True)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def train_one_model(df: pd.DataFrame, workbook: Path) -> dict[str, object]:
    x_raw = df[["series_label", "x_value"]].to_numpy(dtype=np.float32)
    y_raw = df[["y_value"]].to_numpy(dtype=np.float32)
    x_label = df["x_label"].iloc[0]
    y_label = df["y_label"].iloc[0]
    x_column = axis_label_to_column_name(x_label)

    x_scaler = Standardizer.fit(x_raw)
    y_scaler = Standardizer.fit(y_raw)

    x = torch.from_numpy(x_scaler.transform(x_raw)).float()
    y = torch.from_numpy(y_scaler.transform(y_raw)).float()

    model = CurveMLP()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()
    losses: list[float] = []

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy()
    pred = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    truth = y_raw.reshape(-1)

    point_predictions = df.copy()
    point_predictions[x_column] = point_predictions.pop("x_value")
    point_predictions[EXPECTED_Y_COLUMN] = point_predictions.pop("y_value")
    point_predictions[ACTUAL_Y_COLUMN] = pred
    point_predictions[RESIDUAL_COLUMN] = (
        point_predictions[ACTUAL_Y_COLUMN] - point_predictions[EXPECTED_Y_COLUMN]
    )
    point_predictions[TEMPERATURE_COLUMN] = point_predictions["series_label"].map(
        format_temperature
    )
    point_predictions = point_predictions[
        [
            "source_file",
            "source_stem",
            "system",
            TEMPERATURE_COLUMN,
            "x_label",
            "y_label",
            x_column,
            EXPECTED_Y_COLUMN,
            ACTUAL_Y_COLUMN,
            RESIDUAL_COLUMN,
        ]
    ]
    point_predictions.to_csv(
        PREDICTIONS_DIR / f"{workbook.stem}_point_predictions.csv", index=False
    )

    dense_rows: list[dict[str, object]] = []
    for series_label, group in df.groupby("series_label", sort=True):
        x_dense = np.linspace(group["x_value"].min(), group["x_value"].max(), 200)
        features = np.column_stack(
            [np.full_like(x_dense, series_label), x_dense]
        ).astype(np.float32)
        with torch.no_grad():
            dense_scaled = model(
                torch.from_numpy(x_scaler.transform(features)).float()
            ).cpu().numpy()
        dense_pred = y_scaler.inverse_transform(dense_scaled).reshape(-1)
        for xv, yv in zip(x_dense, dense_pred):
            dense_rows.append(
                {
                    "source_file": workbook.name,
                    "source_stem": workbook.stem,
                    "system": df["system"].iloc[0],
                    "series_label": float(series_label),
                    TEMPERATURE_COLUMN: format_temperature(float(series_label)),
                    "x_label": x_label,
                    "y_label": y_label,
                    x_column: float(xv),
                    ACTUAL_Y_COLUMN: float(yv),
                }
            )
    dense_predictions = pd.DataFrame(dense_rows)
    dense_predictions_for_csv = dense_predictions[
        [
            "source_file",
            "source_stem",
            "system",
            TEMPERATURE_COLUMN,
            "x_label",
            "y_label",
            x_column,
            ACTUAL_Y_COLUMN,
        ]
    ]
    dense_predictions_for_csv.to_csv(
        PREDICTIONS_DIR / f"{workbook.stem}_dense_predictions.csv", index=False
    )

    metadata = {
        "source_file": workbook.name,
        "system": df["system"].iloc[0],
        "x_label": x_label,
        "y_label": y_label,
        "series_labels": sorted(df["series_label"].unique().tolist()),
        "n_points": int(len(df)),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mae": float(np.mean(np.abs(pred - truth))),
        "r2": r2_score(truth, pred),
    }

    with open(ARTIFACTS_DIR / f"{workbook.stem}_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    torch.save(
        {
            "source_file": workbook.name,
            "model_state_dict": model.state_dict(),
            "x_mean": x_scaler.mean,
            "x_std": x_scaler.std,
            "y_mean": y_scaler.mean,
            "y_std": y_scaler.std,
        },
        MODELS_DIR / f"{workbook.stem}_model.pt",
    )

    make_plot(df, dense_predictions, losses, workbook)
    return metadata


def make_plot(
    df: pd.DataFrame, dense_predictions: pd.DataFrame, losses: list[float], workbook: Path
) -> None:
    series_labels = sorted(df["series_label"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x_column = axis_label_to_column_name(df["x_label"].iloc[0])

    for idx, series_label in enumerate(series_labels):
        color = cmap(idx % 10)
        group = df[df["series_label"] == series_label]
        dense_group = dense_predictions[dense_predictions["series_label"] == series_label]
        axes[0].scatter(
            group["x_value"],
            group["y_value"],
            color=color,
            s=20,
            label=f"{series_label:g} data",
        )
        axes[0].plot(
            dense_group[x_column],
            dense_group[ACTUAL_Y_COLUMN],
            color=color,
            linewidth=2,
            label=f"{series_label:g} model",
        )

    axes[0].set_title(f"{workbook.stem}: literature points vs model fit")
    axes[0].set_xlabel(df["x_label"].iloc[0])
    axes[0].set_ylabel(df["y_label"].iloc[0])
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(losses, color="black")
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Scaled MSE")
    axes[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{workbook.stem}_fit.png", dpi=180)
    plt.close(fig)


def ensure_dirs() -> None:
    for path in [ARTIFACTS_DIR, PREDICTIONS_DIR, PLOTS_DIR, MODELS_DIR, MPLCONFIGDIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_all_models() -> pd.DataFrame:
    set_seed(SEED)
    ensure_dirs()
    workbooks = sorted(DATA_DIR.glob("*.xlsx"))
    if not workbooks:
        raise FileNotFoundError(f"No `.xlsx` files found in {DATA_DIR}")

    summary_rows: list[dict[str, object]] = []
    for workbook in workbooks:
        df = parse_curve_workbook(workbook)
        summary_rows.append(train_one_model(df, workbook))

    summary = pd.DataFrame(summary_rows).sort_values("source_file").reset_index(drop=True)
    summary.to_csv(ARTIFACTS_DIR / "summary_metrics.csv", index=False)
    return summary


def preview_predictions(n_rows_per_series: int = 3) -> dict[str, pd.DataFrame]:
    previews: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(PREDICTIONS_DIR.glob("*_point_predictions.csv")):
        df = pd.read_csv(csv_path)
        df[TEMPERATURE_COLUMN] = df[TEMPERATURE_COLUMN].map(
            lambda value: format_temperature(float(value))
        )
        x_column = axis_label_to_column_name(df["x_label"].iloc[0])
        preview = (
            df.groupby(TEMPERATURE_COLUMN, sort=True, group_keys=False)
            .head(n_rows_per_series)
            .reset_index(drop=True)
        )
        previews[csv_path.name] = preview[
            [
                TEMPERATURE_COLUMN,
                x_column,
                EXPECTED_Y_COLUMN,
                ACTUAL_Y_COLUMN,
                RESIDUAL_COLUMN,
            ]
        ]
    return previews


def main() -> None:
    summary = run_all_models()
    print(summary.to_string(index=False))
    previews = preview_predictions()
    for filename, df in previews.items():
        print(f"\n{filename}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
