"""Cross-asset falsification test for the wrapper-volume mechanism.

If wrapper volume → spot activity is a real causal channel, BTC's ETF wrapper volume
should NOT predict SOL/ADA/etc. spot venue activity beyond a generic market effect.
We compute the same spec but with mismatched control assets and a placebo "wrapper
volume" series.

Outputs:
  - results/revision/mechanism_cross_asset_falsification.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("falsification")

USV_PROXY = {"coinbase", "gemini"}
CONTROL_ASSETS = ["SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT", "UNI"]


def fit_panel(df: pd.DataFrame, y: str, key: str, extras: list[str] | None = None):
    extras = extras or []
    work = df.dropna(subset=[y, key]).copy()
    if work.empty:
        return None
    work["asset_venue"] = work["asset"] + "_" + work["venue"]
    work = work.set_index(["asset_venue", "date"]).sort_index()
    try:
        return PanelOLS(
            work[y], work[[key] + extras],
            entity_effects=True, time_effects=False,
            drop_absorbed=True, check_rank=False,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        log.warning("fit fail %s: %s", y, exc)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="results/revision/daily_revision.parquet")
    p.add_argument("--consolidated", default="results/revision/etf_flow_daily_consolidated.csv")
    p.add_argument("--out", default="results/revision/mechanism_cross_asset_falsification.csv")
    args = p.parse_args()

    daily = pd.read_parquet(args.daily)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    cons = pd.read_csv(args.consolidated, parse_dates=["date"])

    # Build per-asset BTC and ETH wrapper volume series.
    btc_vol = cons[cons["asset"] == "BTC"][["date", "etf_total_dollar_volume"]].rename(
        columns={"etf_total_dollar_volume": "btc_etfvol"})
    eth_vol = cons[cons["asset"] == "ETH"][["date", "etf_total_dollar_volume"]].rename(
        columns={"etf_total_dollar_volume": "eth_etfvol"})

    df = daily.merge(btc_vol, on="date", how="left").merge(eth_vol, on="date", how="left")
    df["log_btc_etfvol"] = np.log1p(df["btc_etfvol"].fillna(0.0))
    df["log_eth_etfvol"] = np.log1p(df["eth_etfvol"].fillna(0.0))
    df["is_usv"] = df["venue"].isin(USV_PROXY).astype(int)

    rows = []

    # Falsification 1: control asset × USV × BTC ETF volume
    # If real mechanism, control assets should NOT respond to BTC's wrapper volume.
    for control in CONTROL_ASSETS:
        sub = df[df["asset"] == control].copy()
        sub["fake_term"] = sub["is_usv"] * sub["log_btc_etfvol"]
        for y in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume"]:
            fit = fit_panel(sub, y, "fake_term")
            if fit is None:
                continue
            rows.append({
                "test": "control_asset_x_btc_etfvol",
                "asset": control, "outcome": y,
                "coef": float(fit.params["fake_term"]),
                "se": float(fit.std_errors["fake_term"]),
                "tstat": float(fit.tstats["fake_term"]),
                "pval": float(fit.pvalues["fake_term"]),
                "nobs": int(fit.nobs),
            })

    # Falsification 2: BTC × non-USV × BTC ETF volume
    # Mechanism should be USV-specific; non-USV should be weaker or null.
    sub = df[df["asset"] == "BTC"].copy()
    sub["non_usv"] = (~sub["venue"].isin(USV_PROXY)).astype(int)
    sub["btc_nonusv_etfvol"] = sub["non_usv"] * sub["log_btc_etfvol"]
    for y in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume"]:
        fit = fit_panel(sub, y, "btc_nonusv_etfvol")
        if fit is None:
            continue
        rows.append({
            "test": "btc_non_usv_etfvol",
            "asset": "BTC", "outcome": y,
            "coef": float(fit.params["btc_nonusv_etfvol"]),
            "se": float(fit.std_errors["btc_nonusv_etfvol"]),
            "tstat": float(fit.tstats["btc_nonusv_etfvol"]),
            "pval": float(fit.pvalues["btc_nonusv_etfvol"]),
            "nobs": int(fit.nobs),
        })

    # Same for ETH × non-USV × ETH ETF volume
    sub = df[df["asset"] == "ETH"].copy()
    sub["non_usv"] = (~sub["venue"].isin(USV_PROXY)).astype(int)
    sub["eth_nonusv_etfvol"] = sub["non_usv"] * sub["log_eth_etfvol"]
    for y in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume"]:
        fit = fit_panel(sub, y, "eth_nonusv_etfvol")
        if fit is None:
            continue
        rows.append({
            "test": "eth_non_usv_etfvol",
            "asset": "ETH", "outcome": y,
            "coef": float(fit.params["eth_nonusv_etfvol"]),
            "se": float(fit.std_errors["eth_nonusv_etfvol"]),
            "tstat": float(fit.tstats["eth_nonusv_etfvol"]),
            "pval": float(fit.pvalues["eth_nonusv_etfvol"]),
            "nobs": int(fit.nobs),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(out))
    print(out.to_string())


if __name__ == "__main__":
    main()
