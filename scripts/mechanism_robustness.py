"""Extended mechanism robustness for the resubmission pivot.

Outputs:
  - results/revision/mechanism_per_venue.csv     (per-USV-venue heterogeneity)
  - results/revision/mechanism_placebo_pre.csv   (pre-ETF era placebo)
  - results/revision/mechanism_iv_first_stage.csv  (flow -> ETF volume first stage)
  - results/revision/mechanism_lagged.csv         (lag-1, lag-2 robustness)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mechanism_robust")


USV_PROXY = {"coinbase", "gemini"}
NON_USV = {"binance", "bybit", "okx", "bitfinex", "bitstamp", "poloniex", "kraken"}
# Use tz-naive (matches daily panel after dt.tz_localize(None) in attach_etf_to_spot).
BTC_LAUNCH = pd.Timestamp("2024-01-11")
ETH_LAUNCH = pd.Timestamp("2024-07-23")


def attach_etf_to_spot(daily: pd.DataFrame, etf_flow: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    e = etf_flow[["date", "asset", "etf_total_flow_musd",
                  "etf_total_dollar_volume", "etf_active_funds"]].copy()
    e["date"] = pd.to_datetime(e["date"])
    merged = d.merge(e, on=["date", "asset"], how="left")
    for col in ("etf_total_flow_musd", "etf_total_dollar_volume", "etf_active_funds"):
        merged.loc[merged["asset"].isin(["BTC", "ETH"]) & merged[col].isna(), col] = 0.0
    return merged


def fit_entity_only(df: pd.DataFrame, y: str, terms: list[str]):
    work = df.dropna(subset=[y] + terms).copy()
    if work.empty:
        return None
    work["asset_venue"] = work["asset"] + "_" + work["venue"]
    work = work.set_index(["asset_venue", "date"]).sort_index()
    try:
        return PanelOLS(
            work[y], work[terms],
            entity_effects=True, time_effects=False,
            drop_absorbed=True, check_rank=False,
        ).fit(cov_type="clustered", cluster_entity=True)
    except Exception as exc:
        log.warning("fit fail y=%s: %s", y, exc)
        return None


def per_venue_mechanism(spot: pd.DataFrame, asset_label: str) -> pd.DataFrame:
    rows = []
    for venue in sorted(USV_PROXY):
        sub = spot[(spot["asset"] == asset_label) & (spot["venue"] == venue)].copy()
        sub["log_etfvol"] = np.log1p(sub["etf_total_dollar_volume"].fillna(0.0))
        sub["log_flow"] = np.sign(sub["etf_total_flow_musd"].fillna(0.0)) * np.log1p(
            np.abs(sub["etf_total_flow_musd"].fillna(0.0))
        )
        # Single-venue regression — entity FE collapses to single, use OLS-by-time approximation.
        sub = sub.dropna(subset=["log_rolling_notional_depth_proxy_24h", "log_etfvol"])
        if len(sub) < 50:
            continue
        # Simple OLS within venue: y = a + b*log_etfvol + e
        x = sub["log_etfvol"].to_numpy()
        for y_col in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume",
                      "log_dispersion_bps"]:
            y = sub[y_col].to_numpy()
            if len(y) < 30:
                continue
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 30:
                continue
            xx = x[mask]; yy = y[mask]
            beta, alpha = np.polyfit(xx, yy, 1)
            yhat = alpha + beta * xx
            ss_res = np.sum((yy - yhat) ** 2)
            ss_tot = np.sum((yy - yy.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            n = len(xx)
            se = np.sqrt(ss_res / max(n - 2, 1) / np.sum((xx - xx.mean()) ** 2)) if n > 2 else np.nan
            tstat = beta / se if (se and se > 0) else np.nan
            from scipy import stats  # type: ignore
            pval = 2 * (1 - stats.t.cdf(abs(tstat), df=max(n - 2, 1))) if not np.isnan(tstat) else np.nan
            rows.append({
                "asset": asset_label, "venue": venue, "outcome": y_col,
                "n": int(n), "beta": float(beta), "se": float(se),
                "tstat": float(tstat), "pval": float(pval), "rsq": float(r2),
            })
    return pd.DataFrame(rows)


def placebo_pre_etf(spot: pd.DataFrame, asset_label: str, launch_date: pd.Timestamp
                    ) -> pd.DataFrame:
    """Run mechanism on a 'placebo' window 365 days BEFORE the ETF launch.
    ETF flow is 0 by construction in the placebo era — substitute lagged BTC/ETH price
    return as the spurious 'flow' to verify the mechanism does not hold pre-ETF.
    """
    pre = spot[(spot["asset"] == asset_label) &
               (spot["date"] >= launch_date - pd.Timedelta(days=365)) &
               (spot["date"] < launch_date)].copy()
    if pre.empty:
        return pd.DataFrame()
    # Use placebo "flow" = absolute log-close change as a stand-in.
    pre = pre.sort_values(["asset", "venue", "date"])
    pre["placebo_flow"] = pre.groupby(["asset", "venue"])["close"].pct_change().abs() * 1e3
    pre["log_placebo"] = np.log1p(pre["placebo_flow"].fillna(0.0))
    pre["is_usv"] = pre["venue"].isin(USV_PROXY).astype(int)
    pre["usv_x_placebo"] = pre["is_usv"] * pre["log_placebo"]

    rows = []
    for y in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume"]:
        fit = fit_entity_only(pre, y, ["usv_x_placebo", "is_usv", "log_placebo"])
        if fit is None or "usv_x_placebo" not in fit.params.index:
            continue
        rows.append({
            "asset": asset_label, "outcome": y,
            "coef": float(fit.params["usv_x_placebo"]),
            "se": float(fit.std_errors["usv_x_placebo"]),
            "tstat": float(fit.tstats["usv_x_placebo"]),
            "pval": float(fit.pvalues["usv_x_placebo"]),
            "nobs": int(fit.nobs),
        })
    return pd.DataFrame(rows)


def lagged_mechanism(spot: pd.DataFrame, asset_label: str, lag_days: int) -> pd.DataFrame:
    work = spot[spot["asset"].isin([asset_label] + [
        "SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT", "UNI"])].copy()
    work = work.sort_values(["asset", "venue", "date"])
    work["etfvol_lag"] = work.groupby(["asset", "venue"])["etf_total_dollar_volume"].shift(lag_days)
    work["log_etfvol_lag"] = np.log1p(work["etfvol_lag"].fillna(0.0))
    work["is_treated"] = (work["asset"] == asset_label).astype(int)
    work["is_usv"] = work["venue"].isin(USV_PROXY).astype(int)
    work["treated_usv_etfvol_lag"] = work["is_treated"] * work["is_usv"] * work["log_etfvol_lag"]
    rows = []
    for y in ["log_rolling_notional_depth_proxy_24h", "log_notional_volume"]:
        fit = fit_entity_only(work, y, ["treated_usv_etfvol_lag"])
        if fit is None or "treated_usv_etfvol_lag" not in fit.params.index:
            continue
        rows.append({
            "asset": asset_label, "lag_days": lag_days, "outcome": y,
            "coef": float(fit.params["treated_usv_etfvol_lag"]),
            "se": float(fit.std_errors["treated_usv_etfvol_lag"]),
            "tstat": float(fit.tstats["treated_usv_etfvol_lag"]),
            "pval": float(fit.pvalues["treated_usv_etfvol_lag"]),
            "nobs": int(fit.nobs),
        })
    return pd.DataFrame(rows)


def first_stage_iv(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Daily ETF flow → ETF dollar volume (per asset)."""
    rows = []
    for asset in ["BTC", "ETH"]:
        sub = consolidated[consolidated["asset"] == asset].dropna(
            subset=["etf_total_flow_musd", "etf_total_dollar_volume"]).copy()
        if sub.empty:
            continue
        x = sub["etf_total_flow_musd"].to_numpy()
        y = np.log1p(sub["etf_total_dollar_volume"].to_numpy())
        beta, alpha = np.polyfit(x, y, 1)
        yhat = alpha + beta * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n = len(x)
        se = np.sqrt(ss_res / max(n - 2, 1) / np.sum((x - x.mean()) ** 2)) if n > 2 else np.nan
        tstat = beta / se if (se and se > 0) else np.nan
        rows.append({
            "asset": asset, "n": int(n), "beta_flow_to_logvol": float(beta),
            "alpha": float(alpha), "se": float(se),
            "tstat": float(tstat), "rsq": float(r2),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="results/revision/daily_revision.parquet")
    p.add_argument("--btc-flows", default="data/raw/etf/farside_btc_flows.parquet")
    p.add_argument("--eth-flows", default="data/raw/etf/farside_eth_flows.parquet")
    p.add_argument("--etf-prices", default="data/raw/etf_prices/etf_prices_daily.parquet")
    p.add_argument("--consolidated", default="results/revision/etf_flow_daily_consolidated.csv")
    p.add_argument("--out-dir", default="results/revision")
    args = p.parse_args()

    out = Path(args.out_dir)
    daily = pd.read_parquet(args.daily)
    consolidated = pd.read_csv(args.consolidated, parse_dates=["date"])

    spot_btc = attach_etf_to_spot(daily, consolidated)
    spot_eth = attach_etf_to_spot(daily, consolidated)

    # 1. Per-venue heterogeneity
    pv_btc = per_venue_mechanism(spot_btc, "BTC")
    pv_eth = per_venue_mechanism(spot_eth, "ETH")
    pd.concat([pv_btc, pv_eth], ignore_index=True).to_csv(
        out / "mechanism_per_venue.csv", index=False)
    log.info("per-venue: BTC %d rows, ETH %d rows", len(pv_btc), len(pv_eth))

    # 2. Pre-ETF placebo
    pl_btc = placebo_pre_etf(spot_btc, "BTC", BTC_LAUNCH)
    pl_eth = placebo_pre_etf(spot_eth, "ETH", ETH_LAUNCH)
    pd.concat([pl_btc, pl_eth], ignore_index=True).to_csv(
        out / "mechanism_placebo_pre.csv", index=False)
    log.info("placebo: BTC %d rows, ETH %d rows", len(pl_btc), len(pl_eth))

    # 3. Lagged mechanism (lag 1, 5)
    lag_frames = []
    for lag in (1, 5):
        for asset in ("BTC", "ETH"):
            spot = attach_etf_to_spot(daily, consolidated)
            df = lagged_mechanism(spot, asset, lag)
            if not df.empty:
                lag_frames.append(df)
    if lag_frames:
        pd.concat(lag_frames, ignore_index=True).to_csv(out / "mechanism_lagged.csv", index=False)
        log.info("lagged frames: %d", len(lag_frames))

    # 4. First-stage IV
    fs = first_stage_iv(consolidated)
    fs.to_csv(out / "mechanism_iv_first_stage.csv", index=False)
    log.info("first stage rows: %d", len(fs))


if __name__ == "__main__":
    main()
