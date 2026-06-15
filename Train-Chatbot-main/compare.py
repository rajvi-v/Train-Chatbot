from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_FILES = {
    "WAT->WEY": {
        "KNN":           "model_knn_WAT2WEY.pkl",
        "Random Forest": "model_rf_WAT2WEY.pkl",
        "XGBoost":       "model_xgb_WAT2WEY.pkl",
    },
    "WEY->WAT": {
        "KNN":           "model_knn_WEY2WAT.pkl",
        "Random Forest": "model_rf_WEY2WAT.pkl",
        "XGBoost":       "model_xgb_WEY2WAT.pkl",
    },
}

BIN_EDGES  = [-np.inf, -5, 5, 15, 30, np.inf]
BIN_LABELS = ["Early\n(<-5)", "On time\n(-5–5)", "Slight\n(5–15)",
              "Moderate\n(15–30)", "Severe\n(>30)"]

OUT_DIR = Path("figures_comparison")
OUT_DIR.mkdir(exist_ok=True)

COLORS = {
    "KNN":           "#5bc0de",
    "Random Forest": "#f0ad4e",
    "XGBoost":       "#d9534f",
}

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def load_payload(path):
    
    p = Path(path)
    if not p.exists():
        print(f"  [WARNING] File not found: {path}")
        return None
    return joblib.load(p)


def bin_delays(values):
   
    return pd.cut(values, bins=BIN_EDGES, labels=False).astype(int)


print("=" * 62)
print("Loading model payloads...")
print("=" * 62)

payloads = {}   

for direction, model_map in MODEL_FILES.items():
    payloads[direction] = {}
    for model_name, filepath in model_map.items():
        payload = load_payload(filepath)
        if payload is not None:
            payloads[direction][model_name] = payload
            print(f"  Loaded  {model_name:<15}  {direction}")


print("\n" + "=" * 62)
print("TEST-SET PERFORMANCE SUMMARY")
print("=" * 62)

rows = []
for direction, models in payloads.items():
    for model_name, p in models.items():
        rows.append({
            "Direction":     direction,
            "Model":         model_name,
            "RMSE (min)":    round(p["rmse"], 3),
            "MAE (min)":     round(p["mae"],  3),
            "R²":            round(p["r2"],   3),
        })

df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False))
print()


for direction, models in payloads.items():
    if not models:
        continue

    model_names = list(models.keys())
    x           = np.arange(len(model_names))
    width       = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        f"Model Comparison — {direction}",
        fontsize=13, fontweight="bold", y=1.01
    )

    metrics = [
        ("RMSE (minutes)", "rmse", False),
        ("MAE (minutes)",  "mae",  False),
        ("R²",             "r2",   True),   
    ]

    for ax, (ylabel, key, higher_better) in zip(axes, metrics):
        vals   = [models[m][key] for m in model_names]
        colors = [COLORS.get(m, "#888") for m in model_names]

        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_ylim(0, max(vals) * 1.18)

        best_idx = (np.argmax(vals) if higher_better else np.argmin(vals))
        ax.get_children()[best_idx].set_edgecolor("black")
        ax.get_children()[best_idx].set_linewidth(2)

    plt.tight_layout()
    out = OUT_DIR / f"comparison_metrics_{direction.replace('->','_')}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")

def plot_delay_confusion(y_true, y_pred, model_name, direction, out_dir):
    
    n_bins  = len(BIN_LABELS)
    y_true_b = bin_delays(y_true)
    y_pred_b = bin_delays(y_pred)

    matrix = np.zeros((n_bins, n_bins), dtype=int)
    for actual, predicted in zip(y_true_b, y_pred_b):
        matrix[actual, predicted] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)

    for i in range(n_bins):
        for j in range(n_bins):
            pct    = norm_matrix[i, j]
            count  = matrix[i, j]
            color  = "white" if pct > 0.55 else "black"
            ax.text(
                j, i,
                f"{pct*100:.1f}%\n({count:,})",
                ha="center", va="center",
                fontsize=7.5, color=color
            )

    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels(BIN_LABELS, fontsize=8)
    ax.set_yticklabels(BIN_LABELS, fontsize=8)
    ax.set_xlabel("Predicted Delay Category", fontsize=10)
    ax.set_ylabel("Actual Delay Category",    fontsize=10)
    ax.set_title(
        f"{model_name} — Delay Category Matrix\n({direction})",
        fontsize=11, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion of Actual Category", fontsize=8)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
    )

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    out = out_dir / f"delay_matrix_{safe_name}_{direction.replace('->','_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


print()
for direction, models in payloads.items():
    for model_name, p in models.items():
        y_true = p.get("y_test")
        y_pred = p.get("y_pred")
        if y_true is None or y_pred is None:
            print(f"  [SKIP] {model_name} {direction} — no y_test/y_pred in payload")
            continue
        plot_delay_confusion(
            np.array(y_true),
            np.array(y_pred),
            model_name,
            direction,
            OUT_DIR,
        )


print("\n" + "=" * 62)
print("All outputs saved to:", OUT_DIR)
print("=" * 62)