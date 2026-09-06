"""Ether-launch safety net: is the ETH-launch Coinbase DDD effect access-specific?

Re-estimate the ETH x segment x post triple for the launch event with the
'treated venue segment' set to (a) Coinbase (reproduce), (b) offshore venues
(placebo venue), and with the treated ASSET replaced by control coins on
Coinbase (placebo asset). If offshore / control placebos also jump, the
Coinbase effect is a broad ETH-launch activity surge, not a US-access channel.
"""
import os
from __future__ import annotations
import warnings, logging
from pathlib import Path
import numpy as np, pandas as pd
from linearmodels.panel import PanelOLS
warnings.simplefilter("ignore"); logging.getLogger("linearmodels").setLevel(logging.ERROR)

ROOT = Path(os.environ.get("CRYPTO1_ROOT", Path(__file__).resolve().parents[1]))
d = pd.read_parquet(ROOT / "results/revision_fixed/daily_revision.parquet")
d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
TURN = "log_rolling_notional_depth_proxy_24h"
OFFSHORE = {"binance", "okx", "bybit", "bitfinex", "poloniex"}

def window(ev, w=30):
    ev = pd.Timestamp(ev)
    s = d[(d["date"] >= ev - pd.Timedelta(days=w)) & (d["date"] <= ev + pd.Timedelta(days=w))].copy()
    s["event_time"] = (s["date"] - ev).dt.days; s["post"] = (s["event_time"] >= 0).astype(int)
    return s

def ddd(win, treated_asset, seg_set, label):
    w = win.copy()
    w["seg"] = w["venue"].isin(seg_set).astype(int)
    w["tr"] = (w["asset"] == treated_asset).astype(int)
    w["tr_post"] = w["tr"] * w["post"]; w["seg_post"] = w["seg"] * w["post"]
    w["tr_seg"] = w["tr"] * w["seg"]; w["ddd"] = w["tr"] * w["seg"] * w["post"]
    w["av"] = w["asset"] + "_" + w["venue"]
    w = w.dropna(subset=[TURN]).set_index(["av", "date"]).sort_index()
    try:
        f = PanelOLS(w[TURN], w[["ddd", "tr_post", "seg_post", "tr_seg"]],
                     entity_effects=True, time_effects=True, drop_absorbed=True,
                     check_rank=False).fit(cov_type="clustered", cluster_entity=True)
        c = float(f.params["ddd"])
        return {"label": label, "coef": c, "pct": (np.exp(c)-1)*100,
                "se": float(f.std_errors["ddd"]), "pval": float(f.pvalues["ddd"]), "nobs": int(f.nobs)}
    except Exception as e:
        return {"label": label, "coef": np.nan, "pct": np.nan, "se": np.nan, "pval": np.nan, "nobs": 0}

win = window("2024-07-23")
rows = [
    ddd(win, "ETH", {"coinbase"}, "ETH x Coinbase (reproduce)"),
    ddd(win, "ETH", {"gemini"},   "ETH x Gemini"),
    ddd(win, "ETH", OFFSHORE,      "ETH x OFFSHORE (placebo venue)"),
    ddd(win, "SOL", {"coinbase"}, "SOL x Coinbase (placebo asset, no ETF)"),
    ddd(win, "DOGE", {"coinbase"},"DOGE x Coinbase (placebo asset, no ETF)"),
    ddd(win, "XRP", {"coinbase"}, "XRP x Coinbase (placebo asset, no ETF)"),
]
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/revision_newcalcs/N3b_eth_launch_safety.csv", index=False)
print("=== ETH LAUNCH (2024-07-23) DDD safety net, 24h turnover ===")
for r in rows:
    star = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 else "*" if r["pval"] < 0.10 else ""
    print(f"  {r['label']:42s} {r['pct']:+6.1f}%  p={r['pval']:.3f} {star}")
print("\nIf OFFSHORE and/or control-asset placebos also jump significantly,")
print("the Coinbase effect is a broad ETH-launch surge, not a US-access channel.")
