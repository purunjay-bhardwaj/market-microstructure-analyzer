# Market Microstructure Analyzer

**Repository:** Market Microstructure Analyzer — end-to-end pipeline for tick-level microstructure analysis, alerting, and visualization (Python, Pandas, Streamlit).

---

## One-line summary

End-to-end system that ingests per-second tick snapshots, computes microstructure features (spread, mid, VWAP, order-book imbalance), detects liquidity events (spread z-score spikes and depth collapses), and provides an interactive Streamlit dashboard for visualization and triage.

---

## Contents of this repo

* `src/` — source code

  * `generate_data.py` — synthetic tick + order-book generator
  * `features.py` — feature engineering (spread, mid, VWAP, imbalance, rolling stats)
  * `signals.py` — simple rule-based detectors (spread spike, liquidity gap) and alerts exporter
  * `eval.py` — backtest & summary metrics (percent alerts, mean post-alert returns)
  * `dashboard.py` — Streamlit interactive demo
* `data/` — generated CSVs (local; small sample optionally included)

  * `synthetic_ticks.csv` — generated tick snapshots
  * `with_features.csv` / `with_micro_features.csv` — features
  * `alerts.csv` — flagged alert rows
  * `alert_summary.csv` — one-line evaluation summary
* `requirements.txt` — Python dependencies
* `README.md` — this file
* `Purunjay_Bhardwaj_Resume.pdf` — (optional) your resume; local file path: `/mnt/data/Purunjay_Bhardwaj_Resume.pdf`

---

## Quickstart (run locally)

These commands assume macOS / Linux with Python 3.10+.

```bash
# 1. clone repo (if not already local)
# git clone https://github.com/YOUR_USERNAME/market-microstructure-analyzer.git
# cd market-microstructure-analyzer

# 2. create and activate venv
python3 -m venv venv
source venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. optionally generate synthetic data (2 hours of per-second ticks by default)
python3 src/generate_data.py

# 5. compute features
python3 src/features.py

# 6. detect alerts
python3 src/signals.py

# 7. evaluate alerts and save summary
python3 src/eval.py

# 8. run interactive dashboard
streamlit run src/dashboard.py
```

---

## What to expect (sample outputs)

* `data/alerts.csv` — CSV with flagged rows that satisfied either a spread spike or a liquidity gap.
* `data/alert_summary.csv` — contains: `total_rows`, `n_alerts`, `pct_alerts`, `mean_return_frac`, `mean_return_bps`, `n_samples`.

Use these numbers to craft resume bullets and quantify impact (e.g., alerts ≈ `X%` of ticks; mean post-alert move ≈ `Y` bps within 5s).

---

## How it works (brief)

1. **Ingest**: synthetic (or real) tick snapshots with `bid`, `ask`, `bid_vol`, `ask_vol`, `trade_price`, `trade_size`.
2. **Feature engineering**: compute `spread`, `mid`, `ret_1s`, `vwap`, `imbalance_top`, rolling `spread` mean/std, `spread_z`, `top_depth`, rolling median of `top_depth`.
3. **Signal detection**: two rule-based detectors:

   * **Spread spike**: `spread_z_{60s} > 3`
   * **Liquidity gap**: `top_depth < 0.3 * depth_med_{600s}`
4. **Evaluation**: measure whether flagged events are predictive of short-term price moves by computing mean mid-price return after a `horizon` (default 5s).
5. **Dashboard**: visualize time-series, tune thresholds live, export alerts.

---

## Tips for demo & interview talking points

* Elevator pitch (30s): describe the pipeline, list features, detectors, and the backtest metric (alerts % and mean bps). Keep numbers handy from `data/alert_summary.csv`.
* Demo flow: open dashboard → show summary metrics → zoom to a spike in spread_z → click the alert row → show how the mid price moves shortly after.
* Possible improvements to discuss: ML ranking of alerts, L2 true order-book ingestion, latency reduction, per-symbol calibration, and signal explainability.

---

## Deployment suggestions

* **Streamlit Community Cloud**: easiest. Connect your GitHub repo and point app to `src/dashboard.py`.
* **Docker**: create a lightweight container with Python and run Streamlit for more control.
* **Heroku / Railway / Fly**: other PaaS options — you will need a `requirements.txt` and a small `Procfile` for Streamlit.

---

## License

MIT License — see `LICENSE` file (recommended). You may use this code for project portfolios, interviews, and self-learning. If you share, please attribute.

---

## .gitignore

```
# virtual env
venv/
__pycache__/
.DS_Store

# data
data/*.csv
data/*.npy
data/*.pkl
data/*.parquet

data/

# editor
.vscode/
```

## Contact / Author

Purunjay Bhardwaj — project owner. Resume (local upload): `/mnt/data/Purunjay_Bhardwaj_Resume.pdf`.

---

## Acknowledgements & resources

* Pandas documentation — great for rolling windows and time-series ops
* Streamlit docs — for UI and deployment
* Papers and blog posts on market microstructure (VWAP, order-book imbalance)

---

*Happy coding — ping me if you want me to create a GitHub release, LICENSE file, or a short demo GIF for the README.*
