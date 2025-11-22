# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io
import time

st.set_page_config(page_title="Market Microstructure Analyzer", layout="wide")

# --- Helpers: try to reuse your modules if available
USE_FEATURES_MODULE = False
USE_SIGNALS_MODULE = False
try:
    from features import load_ticks, add_basic_features, add_micro_features
    USE_FEATURES_MODULE = True
except Exception:
    USE_FEATURES_MODULE = False

try:
    from signals import detect_spread_spike, detect_liquidity_gap, combine_and_save
    USE_SIGNALS_MODULE = True
except Exception:
    USE_SIGNALS_MODULE = False

# --- Data loading
DATA_DIR = Path("data")
FEATURES_PATHS = [
    DATA_DIR / "with_micro_features.csv",
    DATA_DIR / "with_features.csv",
    DATA_DIR / "synthetic_ticks.csv",
]

@st.cache_data(ttl=30)
def load_features():
    for p in FEATURES_PATHS:
        if p.exists():
            try:
                df = pd.read_csv(p, parse_dates=["timestamp"])
                return df, str(p)
            except Exception:
                pass
    return None, None

df, used_path = load_features()
if df is None:
    st.error("No data found. Run src/features.py to generate data/with_micro_features.csv (or with_features.csv).")
    st.stop()

st.title("Market Microstructure Analyzer — Dashboard")
st.markdown(f"**Loaded:** `{used_path}` — rows: **{len(df):,}**")

# --- Sidebar controls
st.sidebar.header("Controls")
spread_window = st.sidebar.number_input("Spread rolling window (s)", value=60, min_value=1, max_value=3600)
spread_z_thresh = st.sidebar.slider("Spread z-score threshold", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
depth_factor = st.sidebar.slider("Depth collapse factor (fraction of rolling median)", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
vol_window = st.sidebar.number_input("Volatility window (s)", value=10, min_value=1, max_value=600)
horizon_seconds = st.sidebar.number_input("Backtest horizon (s)", value=5, min_value=1, max_value=60)

st.sidebar.markdown("---")
if st.sidebar.button("Recompute features & alerts"):
    # recompute in-place and show a small toast
    with st.spinner("Recomputing micro features and alerts..."):
        if USE_FEATURES_MODULE:
            df2 = add_basic_features(df)
            df2 = add_micro_features(df2, spread_window=spread_window, vol_window=vol_window, depth_median_window=600)
        else:
            # minimal recompute
            df2 = df.copy()
            df2["spread"] = df2["ask"] - df2["bid"]
            df2["mid"] = (df2["ask"] + df2["bid"]) / 2.0
            df2["ret_1s"] = df2["mid"].pct_change().fillna(0)
            df2["cum_trade_value"] = (df2["trade_price"] * df2["trade_size"]).cumsum()
            df2["cum_trade_size"] = df2["trade_size"].cumsum().replace(0, np.nan)
            df2["vwap"] = (df2["cum_trade_value"] / df2["cum_trade_size"]).ffill().fillna(df2["mid"])
            df2.drop(columns=["cum_trade_value","cum_trade_size"], inplace=True, errors="ignore")
            denom = (df2["bid_vol"] + df2["ask_vol"]).replace(0, np.nan)
            df2["imbalance_top"] = ((df2["bid_vol"] - df2["ask_vol"]) / denom).fillna(0.0)
            df2[f"spread_mean_{spread_window}s"] = df2["spread"].rolling(spread_window, min_periods=1).mean()
            df2[f"spread_std_{spread_window}s"] = df2["spread"].rolling(spread_window, min_periods=1).std().fillna(0.0)
            eps = 1e-9
            df2[f"spread_z_{spread_window}s"] = (df2["spread"] - df2[f"spread_mean_{spread_window}s"]) / (df2[f"spread_std_{spread_window}s"] + eps)
            df2[f"vol_{vol_window}s"] = df2["ret_1s"].rolling(vol_window, min_periods=1).std().fillna(0.0)
            df2["top_depth"] = df2["bid_vol"] + df2["ask_vol"]
            df2[f"depth_med_600s"] = df2["top_depth"].rolling(600, min_periods=1).median()
            df = df2
        # update cache by reassigning
        load_features.cache_clear()
        st.success("Recomputed features.")

# --- Compute alerts with current thresholds
def compute_alerts_from_df(df_local, spread_z_threshold=3.0, depth_factor_local=0.3):
    dfx = df_local.copy()
    z_col = f"spread_z_{spread_window}s"
    if z_col not in dfx.columns:
        # compute micro features quickly
        dfx = add_basic_features(dfx) if USE_FEATURES_MODULE else dfx
        dfx = add_micro_features(dfx, spread_window=spread_window, vol_window=vol_window, depth_median_window=600) if USE_FEATURES_MODULE else dfx
    # create spread_spike column based on chosen threshold
    dfx["spread_spike"] = dfx.get(z_col, 0) > spread_z_threshold
    # liquidity gap based on depth_factor * rolling median
    med_col = "depth_med_600s"
    if med_col not in dfx.columns:
        dfx = add_micro_features(dfx, spread_window=spread_window, vol_window=vol_window, depth_median_window=600) if USE_FEATURES_MODULE else dfx
    dfx["liquidity_gap"] = dfx["top_depth"] < (dfx["depth_med_600s"] * depth_factor_local)
    dfx["any_alert"] = dfx["spread_spike"] | dfx["liquidity_gap"]
    return dfx

df_alerts = compute_alerts_from_df(df, spread_z_threshold=spread_z_thresh, depth_factor_local=depth_factor)

# --- Top row with key metrics
col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    st.subheader("Summary")
    total = len(df_alerts)
    n_alerts = int(df_alerts["any_alert"].sum())
    pct_alerts = 100.0 * n_alerts / total if total>0 else 0.0
    st.markdown(f"- **Total ticks:** {total:,}")
    st.markdown(f"- **Alerts:** {n_alerts:,} ({pct_alerts:.3f} %)")
with col2:
    st.subheader("Mean return after alert (horizon)")
    # compute mean return after horizon_seconds
    returns = []
    for idx in df_alerts.index[df_alerts["any_alert"]]:
        tgt_idx = idx + horizon_seconds
        if tgt_idx < len(df_alerts):
            m0 = df_alerts.loc[idx, "mid"]
            mf = df_alerts.loc[tgt_idx, "mid"]
            if pd.notna(m0) and pd.notna(mf) and m0 != 0:
                returns.append((mf - m0) / m0)
    mean_ret = np.nan if not returns else np.mean(returns)
    mean_bps = (mean_ret * 1e4) if pd.notna(mean_ret) else None
    st.metric("Mean return (bps)", f"{mean_bps:.2f}" if mean_bps is not None else "n/a")
with col3:
    st.subheader("Alerts by type")
    st.write(df_alerts[["spread_spike","liquidity_gap"]].sum().to_frame("count"))
with col4:
    st.subheader("Data")
    st.write(f"Loaded file: `{used_path}`")

st.markdown("---")

# --- Charts (two columns)
left, right = st.columns([3,2])

with left:
    st.subheader("Mid Price & Alerts")
    # marker series for alerts
    recent = df_alerts.tail(2000).copy()
    recent_idx = recent["timestamp"]
    # mid price line
    st.line_chart(recent.set_index("timestamp")["mid"])

    st.subheader("Spread z-score (rolling)")
    # ensure z-col exists
    z_col = f"spread_z_{spread_window}s"
    if z_col not in recent.columns:
        recent = add_micro_features(recent) if USE_FEATURES_MODULE else recent
    st.line_chart(recent.set_index("timestamp")[z_col].clip(lower=-10, upper=10).tail(2000))

with right:
    st.subheader("Top Depth & Imbalance (recent)")
    st.line_chart(recent.set_index("timestamp")["top_depth"].tail(500))
    st.line_chart(recent.set_index("timestamp")["imbalance_top"].tail(500))

st.markdown("---")

# --- Alerts table
st.subheader("Recent Alerts")
alerts_df = df_alerts[df_alerts["any_alert"]].copy().sort_values("timestamp", ascending=False).head(200)
if len(alerts_df)==0:
    st.info("No alerts found with current thresholds.")
else:
    st.dataframe(alerts_df[["timestamp","bid","ask","mid","spread", z_col, "imbalance_top","top_depth","spread_spike","liquidity_gap"]])

# --- Download buttons
st.markdown("---")
st.subheader("Download / Export")
csv_buf = df_alerts.to_csv(index=False).encode("utf-8")
st.download_button("Download alerts CSV (current thresholds)", data=csv_buf, file_name="alerts_current.csv", mime="text/csv")

# also save to disk if user wants
if st.button("Save current alerts to data/alerts.csv"):
    Path("data").mkdir(parents=True, exist_ok=True)
    df_alerts.to_csv("data/alerts.csv", index=False)
    st.success("Saved data/alerts.csv")

# --- small help & next steps
st.markdown(
"""
**Notes & next steps**
- Use the sidebar to tune the `spread z-score` and `depth collapse factor` thresholds and press *Recompute features & alerts* if you changed window sizes.
- For deployment, run `streamlit run src/dashboard.py` and expose the host or use Streamlit Cloud.
- To capture a GIF for your GitHub/Overleaf, record a short screencast while interacting with the dashboard.
"""
)

