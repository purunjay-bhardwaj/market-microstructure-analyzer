# src/features.py
import pandas as pd
import numpy as np

def load_ticks(path="data/synthetic_ticks.csv"):
    """Load the CSV we generated earlier."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df

def add_basic_features(df):
    """Add spread, mid price, 1s return, and a simple VWAP."""
    df = df.copy()
    # spread and mid price
    df["spread"] = df["ask"] - df["bid"]
    df["mid"] = (df["ask"] + df["bid"]) / 2.0

    # 1-second percentage return on mid price
    df["ret_1s"] = df["mid"].pct_change().fillna(0)

    # simple cumulative VWAP (using trade_price * trade_size)
    df["cum_trade_value"] = (df["trade_price"] * df["trade_size"]).cumsum()
    df["cum_trade_size"] = df["trade_size"].cumsum().replace(0, np.nan)
    # use ffill() to avoid FutureWarning
    df["vwap"] = (df["cum_trade_value"] / df["cum_trade_size"]).ffill().fillna(df["mid"])

    # tidy up helper columns
    df.drop(columns=["cum_trade_value", "cum_trade_size"], inplace=True)

    return df

def add_micro_features(df, spread_window=60, vol_window=10, depth_median_window=600):
    """
    Add microstructure features:
    - imbalance_top: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    - spread rolling mean/std and spread_z (z-score)
    - vol_{vol_window}s: rolling std of ret_1s
    - top_depth: bid_vol + ask_vol
    - depth_med_{depth_median_window}s: rolling median of top_depth
    """

    df = df.copy()

    # --- 1) Top-of-book imbalance (simple, range -1..1)
    # if bid_vol + ask_vol is zero (very unlikely), avoid division by zero
    denom = (df["bid_vol"] + df["ask_vol"]).replace(0, np.nan)
    df["imbalance_top"] = (df["bid_vol"] - df["ask_vol"]) / denom
    df["imbalance_top"] = df["imbalance_top"].fillna(0.0)

    # --- 2) Spread rolling statistics (for z-score)
    # rolling mean and std of spread over the last `spread_window` seconds
    df[f"spread_mean_{spread_window}s"] = df["spread"].rolling(spread_window, min_periods=1).mean()
    df[f"spread_std_{spread_window}s"] = df["spread"].rolling(spread_window, min_periods=1).std().fillna(0.0)

    # z-score: how many std devs current spread is above the rolling mean
    # add a tiny epsilon so we don't divide by zero
    eps = 1e-9
    df[f"spread_z_{spread_window}s"] = (df["spread"] - df[f"spread_mean_{spread_window}s"]) / (df[f"spread_std_{spread_window}s"] + eps)

    # --- 3) short-term volatility of mid returns
    df[f"vol_{vol_window}s"] = df["ret_1s"].rolling(vol_window, min_periods=1).std().fillna(0.0)

    # --- 4) top depth: total visible volume at top level
    df["top_depth"] = df["bid_vol"] + df["ask_vol"]

    # rolling median depth used later to detect "depth collapse" (liquidity gap)
    df[f"depth_med_{depth_median_window}s"] = df["top_depth"].rolling(depth_median_window, min_periods=1).median()

    # --- optional: normalized imbalance magnitude (abs)
    df["abs_imbalance_top"] = df["imbalance_top"].abs()

    # IMPORTANT: return the DataFrame (fixes the NoneType bug)
    return df

if __name__ == "__main__":
    # Load raw ticks
    df = load_ticks("data/synthetic_ticks.csv")

    # Add basic features
    df = add_basic_features(df)

    # Add microstructure features
    df = add_micro_features(df)

    # Save output file
    out_path = "data/with_micro_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} — rows: {len(df)}")
    print("Columns:", list(df.columns))
    print(df.head())
