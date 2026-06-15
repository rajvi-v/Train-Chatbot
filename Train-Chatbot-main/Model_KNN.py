import glob
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STATION_ORDER_WAT2WEY = [
    "WAT", "CLJ", "WOK", "BSK", "WIN", "SHW", "ESL", "SOA",
    "SWG", "SDN", "SOU", "TTN", "ANF", "BEU", "BCU", "SWY",
    "NWM", "HNA", "CHR", "POK", "BMH", "BSM", "PKS", "POO",
    "HAM", "HOL", "WRM", "WOO", "MTN", "DCH", "UPW", "WEY",
]

STATION_ORDER_WEY2WAT = list(reversed(STATION_ORDER_WAT2WEY))

FEATURES = [
    "planned_dep_min",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak",
    "hour",
    "station_num",
    "journey_progress",
    "stations_remaining",
    "current_delay",
    "destination_num",
    "stops_to_dest",
]

K_VALUES = [3, 5, 10, 15, 20]

SAMPLE_FRAC = 0.3
FIGURES_DIR = Path("figures_knn")


# helpers

def to_minutes(t):
    if pd.isnull(t):
        return np.nan
    try:
        parts = str(t).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return np.nan


def build_dataset(df, station_order):
    station_map = {s: i for i, s in enumerate(station_order)}
    n_stations = len(station_order)

    df = df.copy()

    df["planned_arr_min"] = df["planned_arrival_time"].apply(to_minutes)
    df["actual_arr_min"] = df["actual_arrival_time"].apply(to_minutes)
    df["planned_dep_min"] = df["planned_departure_time"].apply(to_minutes)
    df["actual_dep_min"] = df["actual_departure_time"].apply(to_minutes)

    df["current_delay"] = df["actual_dep_min"] - df["planned_dep_min"]

    df["station_num"] = df["location"].map(station_map)
    df = df.dropna(subset=["station_num"])
    df["station_num"] = df["station_num"].astype(int)

    df["date_of_service"] = pd.to_datetime(df["date_of_service"])

    df["day_of_week"] = df["date_of_service"].dt.dayofweek
    df["month"] = df["date_of_service"].dt.month

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["is_peak"] = df["planned_dep_min"].apply(
        lambda x: 1 if pd.notna(x) and (
            420 <= x <= 570 or 990 <= x <= 1140
        ) else 0
    )

    df["hour"] = df["planned_dep_min"] // 60

    df["stations_remaining"] = (n_stations - 1) - df["station_num"]

    df["journey_progress"] = (
        df["station_num"] / (n_stations - 1)
    )

    delay_lookup = df[
        ["rid", "station_num", "actual_arr_min", "planned_arr_min"]
    ].copy()

    delay_lookup["arr_delay"] = (
        delay_lookup["actual_arr_min"] -
        delay_lookup["planned_arr_min"]
    )

    delay_lookup = delay_lookup[
        ["rid", "station_num", "arr_delay"]
    ].dropna()

    print(f"    Generating pairs (sample={int(SAMPLE_FRAC*100)}%)...")

    base_cols = [
        "rid",
        "station_num",
        "planned_dep_min",
        "day_of_week",
        "month",
        "is_weekend",
        "is_peak",
        "hour",
        "stations_remaining",
        "journey_progress",
        "current_delay",
    ]

    df_base = df.dropna(subset=base_cols).copy()

    df_base = df_base[
        (df_base["current_delay"] >= -30) &
        (df_base["current_delay"] <= 200)
    ]

    if SAMPLE_FRAC < 1.0:
        df_base = df_base.sample(
            frac=SAMPLE_FRAC,
            random_state=42
        )

    pairs = df_base.merge(
        delay_lookup.rename(
            columns={
                "station_num": "destination_num",
                "arr_delay": "dest_arr_delay",
            }
        ),
        on="rid",
        how="inner",
    )

    pairs = pairs[
        pairs["destination_num"] > pairs["station_num"]
    ]

    pairs["additional_delay"] = (
        pairs["dest_arr_delay"] -
        pairs["current_delay"]
    )

    pairs["stops_to_dest"] = (
        pairs["destination_num"] -
        pairs["station_num"]
    )

    pairs = pairs.dropna(
        subset=FEATURES + ["additional_delay"]
    )

    pairs = pairs[
        (pairs["additional_delay"] >= -30) &
        (pairs["additional_delay"] <= 200)
    ].copy()

    X = pairs[FEATURES].values
    y = pairs["additional_delay"].values

    return X, y, station_map, pairs


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# plots

def plot_hyperparameter_tuning(results, label, out_dir):

    k_vals = list(results.keys())
    rmse_vals = [results[k]["rmse"] for k in k_vals]

    colors = [
        "#d9534f" if r == min(rmse_vals)
        else "#5bc0de"
        for r in rmse_vals
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        [f"k={k}" for k in k_vals],
        rmse_vals,
        color=colors,
        edgecolor="white",
        width=0.5,
    )

    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)

    ax.set_xlabel("Number of Neighbours (k)")
    ax.set_ylabel("Validation RMSE (minutes)")

    ax.set_title(
        f"KNN — Hyperparameter Tuning ({label})"
    )

    ax.set_ylim(
        min(rmse_vals) * 0.95,
        max(rmse_vals) * 1.05,
    )

    plt.tight_layout()

    path = out_dir / (
        f"knn_hyperparameter_tuning_"
        f"{label.replace('->','_')}.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"    Saved: {path}")


def plot_actual_vs_predicted(
    y_true,
    y_pred,
    label,
    out_dir
):

    idx = np.random.choice(
        len(y_true),
        size=min(5000, len(y_true)),
        replace=False,
    )

    yt, yp = y_true[idx], y_pred[idx]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        yt,
        yp,
        alpha=0.2,
        s=8,
        color="#5bc0de",
        label="Predictions",
    )

    lims = [
        min(yt.min(), yp.min()),
        max(yt.max(), yp.max()),
    ]

    ax.plot(
        lims,
        lims,
        "r--",
        linewidth=1.5,
        label="Perfect prediction",
    )

    ax.set_xlabel(
        "Actual Additional Delay (minutes)"
    )

    ax.set_ylabel(
        "Predicted Additional Delay (minutes)"
    )

    ax.set_title(
        f"KNN — Actual vs Predicted ({label})"
    )

    ax.legend()

    plt.tight_layout()

    path = out_dir / (
        f"knn_actual_vs_predicted_"
        f"{label.replace('->','_')}.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"    Saved: {path}")


def plot_residuals(
    y_true,
    y_pred,
    label,
    out_dir
):

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(
        residuals,
        bins=80,
        color="#5bc0de",
        edgecolor="white",
        alpha=0.85,
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Zero error",
    )

    ax.axvline(
        residuals.mean(),
        color="orange",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean = {residuals.mean():.2f} min",
    )

    ax.set_xlabel(
        "Residual (Actual − Predicted) in minutes"
    )

    ax.set_ylabel("Count")

    ax.set_title(
        f"KNN — Residuals Distribution ({label})"
    )

    ax.legend()

    plt.tight_layout()

    path = out_dir / (
        f"knn_residuals_"
        f"{label.replace('->','_')}.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"    Saved: {path}")


def plot_error_by_stops(
    pairs_test,
    y_true,
    y_pred,
    label,
    out_dir
):

    df_err = pd.DataFrame({
        "stops_to_dest":
            pairs_test["stops_to_dest"].values,
        "abs_error":
            np.abs(y_true - y_pred),
    })

    grouped = (
        df_err.groupby("stops_to_dest")[
            "abs_error"
        ]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        grouped["stops_to_dest"],
        grouped["abs_error"],
        color="#5bc0de",
        edgecolor="white",
    )

    ax.set_xlabel("Stops to Destination")

    ax.set_ylabel(
        "Mean Absolute Error (minutes)"
    )

    ax.set_title(
        f"KNN — Prediction Error by Stops to Destination ({label})"
    )

    plt.tight_layout()

    path = out_dir / (
        f"knn_error_by_stops_"
        f"{label.replace('->','_')}.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"    Saved: {path}")


def plot_k_vs_rmse(results, label, out_dir):

    k_vals = list(results.keys())

    rmse_vals = [
        results[k]["rmse"]
        for k in k_vals
    ]

    mae_vals = [
        results[k]["mae"]
        for k in k_vals
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        k_vals,
        rmse_vals,
        marker="o",
        color="#d9534f",
        label="RMSE",
    )

    ax.plot(
        k_vals,
        mae_vals,
        marker="s",
        color="#5bc0de",
        label="MAE",
    )

    ax.set_xlabel(
        "Number of Neighbours (k)"
    )

    ax.set_ylabel("Error (minutes)")

    ax.set_title(
        f"KNN — Error vs k ({label})"
    )

    ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = out_dir / (
        f"knn_k_vs_error_"
        f"{label.replace('->','_')}.png"
    )

    plt.savefig(path, dpi=150)
    plt.close()

    print(f"    Saved: {path}")


# training

def train_knn_direction(
    label,
    files,
    station_order,
    output_path
):

    print("\n" + "=" * 60)
    print(f"DIRECTION: {label}")
    print("=" * 60)

    out_dir = FIGURES_DIR / label.replace("->", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {len(files)} file(s)...")

    df = pd.concat(
        [pd.read_excel(f) for f in sorted(files)],
        ignore_index=True,
    )

    print(f"Total rows loaded: {len(df):,}")

    X, y, station_map, pairs_df = build_dataset(
        df,
        station_order
    )

    print(
        f"Training examples after pairing: {len(X):,}"
    )

    if len(X) == 0:
        print(f"ERROR: No usable rows for {label}.")
        return

    # Hyperparameter tuning

    print(f"\n{'*'*60}")
    print(
        "Hyperparameter Tuning  "
        "(70% train / 15% val / 15% test)"
    )
    print(f"{'*'*60}")

    X_trainval, X_test, y_trainval, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42,
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.176,
            random_state=42,
        )
    )

    print(
        f"\n  Train:      {len(X_train):,} rows"
        f"  ({100*len(X_train)/len(X):.0f}% of data)"
    )

    print(
        f"  Validation: {len(X_val):,} rows"
        f"  ({100*len(X_val)/len(X):.0f}% of data)"
    )

    print(
        f"  Test:       {len(X_test):,} rows"
        f"  ({100*len(X_test)/len(X):.0f}% of data)"
    )

    print(
        f"\n  {'k':<8} {'Val RMSE':>10}"
        f" {'Val MAE':>10} {'Val R2':>10}"
    )

    print(f"  {'-'*42}")

    results = {}

    for k in K_VALUES:

        model = Pipeline([
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=k,
                    n_jobs=-1,
                ),
            ),
        ])

        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        val_rmse = rmse(y_val, preds)

        val_mae = mean_absolute_error(
            y_val,
            preds
        )

        val_r2 = r2_score(
            y_val,
            preds
        )

        results[k] = {
            "rmse": val_rmse,
            "mae": val_mae,
            "r2": val_r2,
        }

        print(
            f"  {k:<8} "
            f"{val_rmse:>10.2f} "
            f"{val_mae:>10.2f} "
            f"{val_r2:>10.3f}"
        )

    best_k = min(
        results,
        key=lambda k: results[k]["rmse"]
    )

    print(f"\n  Best: k={best_k}")

    print(
        f"  Validation RMSE: "
        f"{results[best_k]['rmse']:.2f} min"
    )

    print("\n  Saving hyperparameter tuning charts...")

    plot_hyperparameter_tuning(
        results,
        label,
        out_dir
    )

    plot_k_vs_rmse(
        results,
        label,
        out_dir
    )

    # Final model

    print(f"\n{'-'*60}")
    print(
        f"Final Model  (80% train / 20% test)"
    )
    print(f"  k={best_k}")
    print(f"{'-'*60}")

    X_train80, X_test20, y_train80, y_test20 = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    _, pairs_test = train_test_split(
        pairs_df,
        test_size=0.20,
        random_state=42,
    )

    print(
        f"\n  Train: {len(X_train80):,} rows"
        f"  |  Test: {len(X_test20):,} rows"
    )

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=best_k,
                n_jobs=-1,
            ),
        ),
    ])

    final_model.fit(X_train80, y_train80)

    final_preds = final_model.predict(X_test20)

    final_rmse = rmse(
        y_test20,
        final_preds
    )

    final_mae = mean_absolute_error(
        y_test20,
        final_preds
    )

    final_r2 = r2_score(
        y_test20,
        final_preds
    )

    print(
        f"\n  Final Test RMSE : "
        f"{final_rmse:.2f} min"
    )

    print(
        f"  Final Test MAE  : "
        f"{final_mae:.2f} min"
    )

    print(
        f"  Final Test R2   : "
        f"{final_r2:.3f}"
    )

    # saving figures

    print("\n  Saving figures...")

    plot_actual_vs_predicted(
        y_test20,
        final_preds,
        label,
        out_dir
    )

    plot_residuals(
        y_test20,
        final_preds,
        label,
        out_dir
    )

    plot_error_by_stops(
        pairs_test,
        y_test20,
        final_preds,
        label,
        out_dir
    )

    print(
        f"  Results recorded — "
        f"RMSE={final_rmse:.2f}, "
        f"MAE={final_mae:.2f}, "
        f"R2={final_r2:.3f}"
    )

    print(
        f"  Figures saved in: {out_dir}/"
    )

    # refit on 100% before saving

    print(
        f"\n  Refitting on 100% of data before saving..."
    )

    final_model.fit(X, y)

    # payload

    payload = {
        "model": final_model,
        "model_name": f"KNN (k={best_k})",
        "direction": label,
        "features": FEATURES,
        "station_map": station_map,
        "station_order": station_order,
        "best_k": best_k,
        "mae": final_mae,
        "rmse": final_rmse,
        "r2": final_r2,
        "tuning_results": results,
        "y_test": y_test20,
        "y_pred": final_preds,
        "pairs_test": pairs_test,
    }

    joblib.dump(payload, output_path)

    print(f"  Saved to: {output_path}")

    return {
        "direction": label,
        "best_k": best_k,
        "rmse": final_rmse,
        "mae": final_mae,
        "r2": final_r2,
    }


# main

FIGURES_DIR.mkdir(exist_ok=True)

all_files = glob.glob("*.xlsx")

if not all_files:
    raise FileNotFoundError(
        "No .xlsx files found in this folder."
    )

wat2wey_files = [
    f for f in all_files
    if "WAT2WEY" in f
]

wey2wat_files = [
    f for f in all_files
    if "WEY2WAT" in f
]

print(
    f"WAT->WEY files ({len(wat2wey_files)}): "
    f"{sorted(wat2wey_files)}"
)

print(
    f"WEY->WAT files ({len(wey2wat_files)}): "
    f"{sorted(wey2wat_files)}"
)

all_results = []

if wat2wey_files:

    r = train_knn_direction(
        label="WAT->WEY",
        files=wat2wey_files,
        station_order=STATION_ORDER_WAT2WEY,
        output_path="model_knn_WAT2WEY.pkl",
    )

    if r:
        all_results.append(r)

if wey2wat_files:

    r = train_knn_direction(
        label="WEY->WAT",
        files=wey2wat_files,
        station_order=STATION_ORDER_WEY2WAT,
        output_path="model_knn_WEY2WAT.pkl",
    )

    if r:
        all_results.append(r)

print("\n" + "=" * 60)
print("KNN RESULTS SUMMARY")
print("=" * 60)

for r in all_results:

    print(
        f"  {r['direction']}  |  "
        f"k={r['best_k']}  |  "
        f"RMSE={r['rmse']:.2f}  "
        f"MAE={r['mae']:.2f}  "
        f"R2={r['r2']:.3f}"
    )

print(f"\nFigures saved in: {FIGURES_DIR}/")
print("Models saved!")