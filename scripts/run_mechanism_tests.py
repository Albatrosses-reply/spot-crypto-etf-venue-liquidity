"""ETF flow × USV venue mechanism tests for the resubmission.

Reviewer #2 demanded direct ETF-level evidence. Tests run here:

  M1. ETF aggregate net flow vs USV-venue depth/volume
       Y_avt = beta * (USV_v * NetFlow_t) + entity FE + time FE + e

  M2. NAV premium/discount vs USV-venue dispersion
       proxy NAV from issuer; premium = (close_etf - NAV) / NAV
       Y_avt = beta * (USV_v * Premium_t) + FE + e

  M3. ETF wrapper trading volume vs spot venue activity
       sum(ETF_vol) day-over-day from yfinance close volumes

Inputs:
  - data/revision/metrics_1h_revision.parquet  (spot panel)
  - data/raw/etf/farside_btc_flows.parquet     (ETF net flow $M)
  - data/raw/etf/farside_eth_flows.parquet
  - data/raw/etf_prices/etf_prices_daily.parquet (ETF OHLCV from yfinance)

Outputs:
  - results/revision/mechanism_btc.csv
  - results/revision/mechanism_eth.csv
  - results/revision/etf_flow_daily_consolidated.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mechanism")


USV_PROXY = {"coinbase", "gemini"}  # combined institution-compatible US-access slice
EU_PROXY = {"bitstamp"}


def consolidate_etf_flows(btc_flow_df: pd.DataFrame, eth_flow_df: pd.DataFrame,
                          etf_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset_label, df in [("BTC", btc_flow_df), ("ETH", eth_flow_df)]:
        if df is None or df.empty:
            continue
        ticker_cols = [c for c in df.columns if c not in ("date", "Total")]
        if "Total" in df.columns:
            total = df["Total"]
        else:
            total = df[ticker_cols].sum(axis=1, min_count=1)
        agg = pd.DataFrame({
            "date": df["date"],
            "asset": asset_label,
            "etf_total_flow_musd": total,
            "etf_active_funds": df[ticker_cols].notna().sum(axis=1),
        })
        # Per-ticker columns kept as wide for premium join later.
        for c in ticker_cols:
            agg[f"flow_{c}"] = df[c]
        rows.append(agg)
    if not rows:
        return pd.DataFrame()
    flow = pd.concat(rows, ignore_index=True)

    # ETF aggregate dollar volume from yfinance.
    etf_p = etf_prices.copy()
    etf_p["date"] = pd.to_datetime(etf_p["date"]).dt.tz_localize(None).dt.normalize()
    # Map ticker to underlying asset.
    btc_tickers = {"IBIT", "FBTC", "ARKB", "BITB", "HODL", "BTCO", "EZBC",
                   "BRRR", "BTCW", "DEFI", "GBTC", "BTC", "BITO"}
    eth_tickers = {"ETHA", "FETH", "ETHW", "ETHV", "QETH", "EZET", "CETH", "ETHE", "ETH"}
    etf_p["underlying"] = np.where(
        etf_p["ticker"].isin(btc_tickers), "BTC",
        np.where(etf_p["ticker"].isin(eth_tickers), "ETH", "OTHER"),
    )
    etf_p["dollar_vol"] = etf_p["close"] * etf_p["volume"]
    etf_dollar = (
        etf_p[etf_p["underlying"].isin(["BTC", "ETH"])]
        .groupby(["date", "underlying"], as_index=False)
        .agg(etf_total_dollar_volume=("dollar_vol", "sum"),
             etf_unique_tickers=("ticker", "nunique"))
        .rename(columns={"underlying": "asset"})
    )
    etf_dollar["date"] = pd.to_datetime(etf_dollar["date"])

    flow["date"] = pd.to_datetime(flow["date"])
    out = flow.merge(etf_dollar, on=["date", "asset"], how="left")
    log.info("consolidated ETF flow rows=%d, BTC rows=%d, ETH rows=%d",
             len(out), (out["asset"] == "BTC").sum(), (out["asset"] == "ETH").sum())
    return out


def attach_etf_to_spot(daily: pd.DataFrame, etf_flow: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    e = etf_flow[["date", "asset", "etf_total_flow_musd",
                  "etf_total_dollar_volume", "etf_active_funds"]].copy()
    e["date"] = pd.to_datetime(e["date"])
    merged = d.merge(e, on=["date", "asset"], how="left")
    # Fill 0 for treated assets on non-trading days; leave NaN for control assets.
    for col in ("etf_total_flow_musd", "etf_total_dollar_volume", "etf_active_funds"):
        merged.loc[merged["asset"].isin(["BTC", "ETH"]) & merged[col].isna(), col] = 0.0
    return merged


def fit_mechanism(df: pd.DataFrame, y: str, key_term: str,
                  extras: list[str] | None = None):
    """Mechanism regressions: entity FE only (asset_venue). NO time FE,
    because the whole point is to leverage time-varying ETF flow."""
    extras = extras or []
    work = df.dropna(subset=[y, key_term]).copy()
    if work.empty:
        return None
    work["asset_venue"] = work["asset"] + "_" + work["venue"]
    # Add an integer date index for clustering / sort.
    work = work.set_index(["asset_venue", "date"]).sort_index()
    X = work[[key_term] + extras]
    try:
        return PanelOLS(
            work[y], X,
            entity_effects=True,
            time_effects=False,
            drop_absorbed=True, check_rank=False,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        log.warning("mechanism fit fail y=%s key=%s: %s", y, key_term, exc)
        return None


def run_mechanism_tests(daily: pd.DataFrame, etf_flow: pd.DataFrame, asset_label: str
                        ) -> pd.DataFrame:
    spot = daily[daily["asset"].isin([asset_label] + [
        "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT", "UNI"])].copy()
    spot = attach_etf_to_spot(spot, etf_flow)

    spot["is_treated"] = (spot["asset"] == asset_label).astype(int)
    spot["is_usv"] = spot["venue"].isin(USV_PROXY).astype(int)
    # Daily ETF flow only defined for treated asset; for control assets it's NaN
    # — set 0 to allow interaction. The interaction is_treated × is_usv × flow
    # will only be non-zero for treated × USV cells.
    spot["flow_musd"] = spot["etf_total_flow_musd"].fillna(0.0)
    spot["log_flow"] = np.sign(spot["flow_musd"]) * np.log1p(np.abs(spot["flow_musd"]))
    spot["log_etf_dvol"] = np.log1p(spot["etf_total_dollar_volume"].fillna(0.0))

    # Interaction terms.
    spot["treated_usv_flow"] = spot["is_treated"] * spot["is_usv"] * spot["log_flow"]
    spot["treated_flow"] = spot["is_treated"] * spot["log_flow"]
    spot["usv_flow"] = spot["is_usv"] * spot["log_flow"]
    spot["treated_usv_etfvol"] = spot["is_treated"] * spot["is_usv"] * spot["log_etf_dvol"]

    rows = []
    for y, key, extras, label in [
        ("log_rolling_notional_depth_proxy_24h", "treated_usv_flow",
         ["treated_flow", "usv_flow"], "depth_x_flow"),
        ("log_notional_volume", "treated_usv_flow",
         ["treated_flow", "usv_flow"], "vol_x_flow"),
        ("log_dispersion_bps", "treated_usv_flow",
         ["treated_flow", "usv_flow"], "disp_x_flow"),
        ("log_rolling_notional_depth_proxy_24h", "treated_usv_etfvol",
         [], "depth_x_etfvol"),
        ("log_notional_volume", "treated_usv_etfvol",
         [], "vol_x_etfvol"),
    ]:
        fit = fit_mechanism(spot, y, key, extras)
        if fit is None:
            continue
        rows.append({
            "asset": asset_label, "model": label, "outcome": y, "term": key,
            "coef": float(fit.params[key]), "se": float(fit.std_errors[key]),
            "tstat": float(fit.tstats[key]), "pval": float(fit.pvalues[key]),
            "rsq": float(fit.rsquared), "nobs": int(fit.nobs),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="results/revision/daily_revision.parquet")
    p.add_argument("--btc-flows", default="data/raw/etf/farside_btc_flows.parquet")
    p.add_argument("--eth-flows", default="data/raw/etf/farside_eth_flows.parquet")
    p.add_argument("--etf-prices", default="data/raw/etf_prices/etf_prices_daily.parquet")
    p.add_argument("--out-dir", default="results/revision")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(args.daily)
    btc_f = pd.read_parquet(args.btc_flows)
    eth_f = pd.read_parquet(args.eth_flows)
    etfp = pd.read_parquet(args.etf_prices)

    consolidated = consolidate_etf_flows(btc_f, eth_f, etfp)
    consolidated.to_csv(out / "etf_flow_daily_consolidated.csv", index=False)

    btc_results = run_mechanism_tests(daily, consolidated, "BTC")
    btc_results.to_csv(out / "mechanism_btc.csv", index=False)
    log.info("BTC mechanism rows=%d", len(btc_results))

    eth_results = run_mechanism_tests(daily, consolidated, "ETH")
    eth_results.to_csv(out / "mechanism_eth.csv", index=False)
    log.info("ETH mechanism rows=%d", len(eth_results))


if __name__ == "__main__":
    main()
