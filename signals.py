# src/signals.py
import pandas as pd
import numpy as np
from pathlib import Path

# try to reuse your features module if present
try:
    from features import load_ticks, add_basic_features, add_micro_features
except Exception:
    # fallback: simple loader if import fails
    def load_ticks(path="data/with_features.csv"):
        return pd.read_csv(path, parse_dates=["timestamp"])
    def add_basic_features(df): return df
    def add_micro_features(df, spread_window=60, vol_window=10, depth_median_window=600):
        # compute minimal micro features inline (if features.py isn't importable)
        df = df.copy()
        if "spread" not in df.columns:
            df["spread"] = df["ask"] - df["bid"]
        if "mid" not in df.columns:
            df["mid"] = (df["ask"] + df["bid"]) / 2.0
        if "ret_1s" not in df.columns:
            df["ret_1s"] = df["mid"].pct_change().fillna(0)
        # imbalance
        denom = (df.get("bid_vol", 0) + df.get("ask_vol", 0)).replace(0, np.nan)
        df["imbalance_top"] = ((df.get("bid_vol",0) - df.get("ask_vol",0)) / denom).fillna(0.0)
        # rolling spread stats
        df[f"spread_mean_{spread_window}s"] = df["spread"].rolling(spread_window, min_periods=1).mean()
        df[f"spread_std_{spread_window}s"] = df["spread"].rolling(spread_window, min_periods=1).std().fillna(0.0)
        eps = 1e-9
        df[f"spread_z_{spread_window}s"] = (df["spread"] - df[f"spread_mean_{spread_window}s"]) / (df[f"spread_std_{spread_window}s"] + eps)
        df[f"vol_{vol_window}s"] = df["ret_1s"].rolling(vol_window, min_periods=1).std().fillna(0.0)
        df["top_depth"] = df.get("bid_vol",0) + df.get("ask_vol",0)
        df[f"depth_med_{depth_median_window}s"] = df["top_depth"].rolling(depth_median_window, min_periods=1).median()
        df["abs_imbalance_top"] = df["imbalance_top"].abs()
        return df

def detect_spread_spike(df, z_threshold=3.0, spread_window=60):
    col = f"spread_z_{spread_window}s"
    if col not in df.columns:
        df = add_micro_features(df, spread_window=spread_window)
    df["spread_spike"] = df[col] > z_threshold
    return df

def detect_liquidity_gap(df, depth_factor=0.3, depth_median_window=600):
    depth_col = "top_depth"
    med_col = f"depth_med_{depth_median_window}s"
    if depth_col not in df.columns or med_col not in df.columns:
        df = add_micro_features(df, depth_median_window=depth_median_window)
    df["liquidity_gap"] = df[depth_col] < (df[med_col] * depth_factor)
    return df

def combine_and_save(df, out_path="data/alerts.csv"):
    df = df.copy()
    df["any_alert"] = df.get("spread_spike", False) | df.get("liquidity_gap", False)
    # keep a compact alerts table for review
    alerts = df[df["any_alert"]].copy()
    keep_cols = ["timestamp","bid","ask","mid","spread","spread_z_60s","imbalance_top","top_depth","spread_spike","liquidity_gap","any_alert"]
    keep_cols = [c for c in keep_cols if c in alerts.columns]
    alerts = alerts[keep_cols]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out_path, index=False)
    print(f"Saved {out_path} — alerts: {len(alerts)}")
    return df, alerts

if __name__ == "__main__":
    # load whichever feature file exists (prefer with_micro_features)
    src_candidates = ["data/with_micro_features.csv", "data/with_features.csv", "data/synthetic_ticks.csv"]
    df = None
    for p in src_candidates:
        try:
            df = load_ticks(p)
            print("Loaded:", p)
            break
        except Exception:
            pass
    if df is None:
        raise FileNotFoundError("Could not find any data/with_micro_features.csv or data/with_features.csv or data/synthetic_ticks.csv")

    # ensure basic and micro features
    df = add_basic_features(df)
    df = add_micro_features(df)

    # detect signals
    df = detect_spread_spike(df, z_threshold=3.0)
    df = detect_liquidity_gap(df, depth_factor=0.3)

    # save alerts
    df, alerts = combine_and_save(df, out_path="data/alerts.csv")
