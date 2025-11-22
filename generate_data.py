import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def make_synthetic_day(symbol="TEST", n_seconds=3600):
    price = 100.0  # start price
    rows = []
    t0 = datetime.now()

    for s in range(n_seconds):
        ts = t0 + timedelta(seconds=s)

        # random walk price
        price += np.random.normal(0, 0.02)

        # bid/ask simulation around mid
        spread = max(0.01, abs(np.random.normal(0.05, 0.02)))
        bid = price - spread/2
        ask = price + spread/2

        # volume simulation
        bid_vol = max(1, np.random.poisson(100))
        ask_vol = max(1, np.random.poisson(100))

        # trade simulation
        trade_size = np.random.exponential(50)
        trade_price = bid if np.random.rand() < 0.5 else ask

        rows.append({
            "timestamp": ts,
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "bid_vol": int(bid_vol),
            "ask_vol": int(ask_vol),
            "trade_price": round(trade_price, 4),
            "trade_size": round(trade_size, 2)
        })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = make_synthetic_day(n_seconds=7200)
    df.to_csv("data/synthetic_ticks.csv", index=False)
    print("Saved data/synthetic_ticks.csv – rows:", len(df))


