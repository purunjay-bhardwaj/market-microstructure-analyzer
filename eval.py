# src/eval.py
import pandas as pd
import numpy as np
from pathlib import Path

# loads alerts and original data, computes simple backtest metrics
def load_data():
    # prefer micro features file if present
    candidates = ["data/with_micro_features.csv", "data/with_features.csv", "data/synthetic_ticks.csv"]
    for p in candidates:
        try:
            df = pd.read_csv(p, parse_dates=["timestamp"])
            print("Loaded:", p)
            return df
        except Exception:
            pass
    raise FileNotFoundError("No input CSV found in data/ folder.")

def compute_alerts_and_eval(horizon_seconds=5):
    df = load_data()
    # ensure micro features exist (lazy compute if necessary)
    try:
        from features import add_basic_features, add_micro_features
        df = add_basic_features(df)
        df = add_micro_features(df)
    except Exception:
        # if features import fails, assume signals.py created alerts.csv already
        if Path("data/alerts.csv").exists():
            alerts = pd.read_csv("data/alerts.csv", parse_dates=["timestamp"])
            df = pd.merge(df, alerts, on=["timestamp","bid","ask","mid"], how="left")
            df["any_alert"] = df["any_alert"].fillna(False)
        else:
            raise

    # make sure any_alert exists
    if "any_alert" not in df.columns:
        # fallback: compute simple rule
        df["any_alert"] = (df.get("spread_z_60s", 0) > 3) | (df.get("top_depth", 1e9) < df.get("depth_med_600s", 1e9) * 0.3)

    total = len(df)
    n_alerts = int(df["any_alert"].sum())
    pct_alerts = 100.0 * n_alerts / total if total>0 else 0.0

    # compute mean return after alert at horizon_seconds (use mid price)
    returns = []
    for idx in df.index[df["any_alert"]]:
        tgt_idx = idx + horizon_seconds
        if tgt_idx < len(df):
            mid_now = df.loc[idx, "mid"]
            mid_future = df.loc[tgt_idx, "mid"]
            if pd.notna(mid_now) and pd.notna(mid_future) and mid_now != 0:
                returns.append((mid_future - mid_now) / mid_now)
    mean_return = np.mean(returns) if returns else np.nan
    std_return = np.std(returns) if returns else np.nan

    print("Total rows:", total)
    print("Alerts:", n_alerts, f"({pct_alerts:.3f}%)")
    print(f"Mean return after {horizon_seconds}s (fraction):", mean_return)
    print(f"Mean return after {horizon_seconds}s (bps):", (mean_return * 1e4) if pd.notna(mean_return) else "nan")
    print("Num returns sampled:", len(returns))
    # save a summary CSV
    summary = {
        "total_rows": total,
        "n_alerts": n_alerts,
        "pct_alerts": pct_alerts,
        "mean_return_frac": mean_return,
        "mean_return_bps": (mean_return * 1e4) if pd.notna(mean_return) else None,
        "n_samples": len(returns)
    }
    pd.DataFrame([summary]).to_csv("data/alert_summary.csv", index=False)
    print("Saved data/alert_summary.csv")
    return summary

if __name__ == "__main__":
    compute_alerts_and_eval(horizon_seconds=5)
