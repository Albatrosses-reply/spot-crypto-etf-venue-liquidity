"""Run revised DID/DDD on the 9-venue panel for the JIFMIM resubmission.

Outputs:
  - results/revision/did_full_grid.csv     (event × outcome × asset/venue × spec)
  - results/revision/ddd_btc_approval.csv  (BTC venue-reallocation DDD with Coinbase as USV)
  - results/revision/event_study_full.csv  (long-window event-time coefs for all 4 main events)
  - results/revision/summary_stats.csv     (Table 2 redo with winsorization)

Identification:
  - Asset-level DID: y_avt = beta * (D_a x Post_t) + alpha_a + lambda_v + delta_t + e
  - Venue DDD: y_avt = ... + theta * (BTC_a x USV_v x Post_t) ...
  - Clustered SE at asset-venue level (HC1 cluster).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("revision_did")


# Two USV proxy variants we want to compare.
USV_VARIANTS = {
    "coinbase_only":   {"coinbase"},
    "gemini_only":     {"gemini"},                       # original paper coding
    "coinbase_gemini": {"coinbase", "gemini"},           # combined
}

EU_PROXY = {"bitstamp"}
TREATED_BTC = "BTC"
TREATED_ETH = "ETH"
CONTROL_BASKET = ["SOL", "XRP", "ADA", "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT", "UNI"]

EVENTS = [
    ("btc_etf_approval",   "2024-01-10", "BTC"),
    ("btc_etf_launch",     "2024-01-11", "BTC"),
    ("eth_rule_approval",  "2024-05-23", "ETH"),
    ("eth_etf_launch",     "2024-07-23", "ETH"),
    ("mica_stablecoin",    "2024-06-30", "MICA"),
    ("mica_full",          "2024-12-30", "MICA"),
]

OUTCOMES = [
    "log_dispersion_bps",
    "log_amihud_illiquidity",
    "log_notional_volume",
    "log_high_low_range_bps",
    "log_rolling_notional_depth_proxy_24h",
    "log_corwin_schultz_spread",
]


def winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    if s.dropna().empty:
        return s
    lo = s.quantile(p)
    hi = s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def to_daily(metrics: pd.DataFrame, sample_filter: str | None = None) -> pd.DataFrame:
    df = metrics.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.floor("D")

    # Quote priority: USD (fiat) > USDT > USDC > FDUSD.
    # For each asset×venue×date, prefer USD if available.
    quote_priority = {"USD": 0, "USDT": 1, "USDC": 2, "FDUSD": 3, "EUR": 4}
    df["_qp"] = df["quote"].map(quote_priority).fillna(9)
    df = df.sort_values(["date", "asset", "venue", "_qp", "timestamp"])

    # Aggregate to daily per asset×venue×quote (preferred quote only).
    agg = (
        df.groupby(["date", "asset", "venue", "_qp", "quote"], as_index=False)
        .agg(
            dispersion_bps=("dispersion_bps", "mean"),
            amihud_illiquidity=("amihud_illiquidity", "mean"),
            notional_volume=("notional_volume", "sum"),
            high_low_range_bps=("high_low_range_bps", "mean"),
            rolling_notional_depth_proxy_24h=("rolling_notional_depth_proxy_24h", "mean"),
            corwin_schultz_spread=("corwin_schultz_spread", "mean"),
            close=("close", "last"),
        )
    )
    # Keep one quote per (date, asset, venue) using priority.
    agg = agg.sort_values(["date", "asset", "venue", "_qp"])
    agg = agg.drop_duplicates(subset=["date", "asset", "venue"], keep="first")
    agg = agg.drop(columns=["_qp"])

    # Compute logs (winsorized).
    for col, log_col in [
        ("dispersion_bps", "log_dispersion_bps"),
        ("amihud_illiquidity", "log_amihud_illiquidity"),
        ("notional_volume", "log_notional_volume"),
        ("high_low_range_bps", "log_high_low_range_bps"),
        ("rolling_notional_depth_proxy_24h", "log_rolling_notional_depth_proxy_24h"),
        ("corwin_schultz_spread", "log_corwin_schultz_spread"),
    ]:
        positive = agg[col].where(agg[col] > 0)
        agg[log_col] = np.log(winsorize(positive, p=0.01))

    log.info("daily aggregated: %d rows, %d venues, %d assets",
             len(agg), agg["venue"].nunique(), agg["asset"].nunique())
    return agg


def make_window(daily: pd.DataFrame, event_date: str, window_days: int = 30) -> pd.DataFrame:
    ev = pd.Timestamp(event_date, tz="UTC")
    lo = ev - pd.Timedelta(days=window_days)
    hi = ev + pd.Timedelta(days=window_days)
    sub = daily[(daily["date"] >= lo) & (daily["date"] <= hi)].copy()
    sub["event_time"] = (sub["date"] - ev).dt.days
    sub["post"] = (sub["event_time"] >= 0).astype(int)
    return sub


def fit_panel_clustered(
    df: pd.DataFrame,
    y: str,
    interaction: str,
    extra_regressors: list[str] | None = None,
):
    """PanelOLS with asset, venue, date FE; cluster by asset×venue."""
    extra_regressors = extra_regressors or []
    work = df.dropna(subset=[y, interaction]).copy()
    if work.empty:
        return None
    work["asset_venue"] = work["asset"] + "_" + work["venue"]
    # Build entity index for PanelOLS (asset, date).
    work = work.set_index(["asset_venue", "date"]).sort_index()
    X = work[[interaction] + extra_regressors]
    # Use PanelOLS with EntityEffects + TimeEffects + venue/asset dummies absorbed.
    # PanelOLS handles asset_venue (entity) FE automatically. Add date FE via TimeEffects.
    try:
        model = PanelOLS(work[y], X, entity_effects=True, time_effects=True, drop_absorbed=True)
        fit = model.fit(cov_type="clustered", cluster_entity=True)
        return fit
    except Exception as exc:
        log.warning("PanelOLS fail y=%s int=%s: %s", y, interaction, exc)
        return None


def build_did_full_grid(daily: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    rows = []
    for ev_id, ev_date, treated in EVENTS:
        win = make_window(daily, ev_date, window_days)
        if win.empty:
            continue
        # Build asset-level DID interactions.
        if treated in ("BTC", "ETH"):
            win["treated_post"] = ((win["asset"] == treated) & (win["post"] == 1)).astype(int)
            int_col = "treated_post"
            spec = f"{treated}_did"
            for y in OUTCOMES:
                fit = fit_panel_clustered(win, y, int_col)
                if fit is None:
                    continue
                rows.append(_extract_row(fit, ev_id, treated, spec, y, int_col, len(win), "asset"))
        # MiCA — venue-based DID.
        elif treated == "MICA":
            win["eu_post"] = (win["venue"].isin(EU_PROXY) & (win["post"] == 1)).astype(int)
            for y in OUTCOMES:
                fit = fit_panel_clustered(win, y, "eu_post")
                if fit is None:
                    continue
                rows.append(_extract_row(fit, ev_id, "EU_VENUE", "mica_eu_did", y, "eu_post", len(win), "venue"))
    return pd.DataFrame(rows)


def build_ddd_btc(daily: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """BTC × USV venue × Post DDD across venue-proxy variants."""
    rows = []
    for variant_name, usv_set in USV_VARIANTS.items():
        for ev_id, ev_date, _ in [e for e in EVENTS if e[0].startswith("btc_")]:
            win = make_window(daily, ev_date, window_days)
            if win.empty:
                continue
            win = win.copy()
            win["btc"] = (win["asset"] == TREATED_BTC).astype(int)
            win["usv"] = win["venue"].isin(usv_set).astype(int)
            win["btc_post"] = win["btc"] * win["post"]
            win["btc_usv"] = win["btc"] * win["usv"]
            win["usv_post"] = win["usv"] * win["post"]
            win["btc_usv_post"] = win["btc"] * win["usv"] * win["post"]
            for y in OUTCOMES:
                fit = fit_panel_clustered(
                    win, y, "btc_usv_post",
                    extra_regressors=["btc_post", "btc_usv", "usv_post"],
                )
                if fit is None:
                    continue
                row = _extract_row(fit, ev_id, "BTCxUSV", f"btc_ddd_{variant_name}",
                                   y, "btc_usv_post", len(win), "ddd")
                row["usv_variant"] = variant_name
                rows.append(row)
    return pd.DataFrame(rows)


def _extract_row(fit, event_id, treated, spec, y, term, nobs, kind):
    coef = float(fit.params[term])
    se = float(fit.std_errors[term])
    tstat = float(fit.tstats[term])
    pval = float(fit.pvalues[term])
    pct_effect = (np.exp(coef) - 1) * 100
    return {
        "event_id": event_id, "treated": treated, "spec": spec,
        "outcome": y, "term": term, "kind": kind,
        "coef": coef, "se": se, "tstat": tstat, "pval": pval,
        "pct_effect": pct_effect, "nobs": int(nobs), "rsq": float(fit.rsquared),
    }


def event_study(daily: pd.DataFrame, ev_id: str, ev_date: str, treated: str,
                outcomes: list[str], window_days: int = 30) -> pd.DataFrame:
    win = make_window(daily, ev_date, window_days)
    if win.empty or treated not in ("BTC", "ETH"):
        return pd.DataFrame()
    rows = []
    for y in outcomes:
        # Build dummies for each event-time t (omit t=-1).
        d = win.dropna(subset=[y]).copy()
        if d.empty:
            continue
        d["asset_venue"] = d["asset"] + "_" + d["venue"]
        # Dummies: D_t = (asset==treated) * I(event_time == t)
        ts = sorted(d["event_time"].unique())
        ts = [t for t in ts if t != -1]
        d["__y"] = d[y]
        regressor_cols = []
        for t in ts:
            col = f"D_{t}"
            d[col] = ((d["asset"] == treated) & (d["event_time"] == t)).astype(int)
            regressor_cols.append(col)
        d = d.set_index(["asset_venue", "date"]).sort_index()
        X = d[regressor_cols]
        try:
            fit = PanelOLS(d["__y"], X, entity_effects=True, time_effects=True,
                           drop_absorbed=True).fit(cov_type="clustered", cluster_entity=True)
        except Exception as exc:
            log.warning("event-study fit fail %s %s: %s", ev_id, y, exc)
            continue
        for t in ts:
            col = f"D_{t}"
            if col not in fit.params.index:
                continue
            rows.append({
                "event_id": ev_id, "treated": treated, "outcome": y,
                "event_time": t, "coef": float(fit.params[col]),
                "se": float(fit.std_errors[col]), "tstat": float(fit.tstats[col]),
                "pval": float(fit.pvalues[col]),
            })
    return pd.DataFrame(rows)


def summary_stats_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [
        "dispersion_bps", "amihud_illiquidity", "notional_volume",
        "high_low_range_bps", "rolling_notional_depth_proxy_24h",
        "corwin_schultz_spread",
    ]:
        s = metrics[col].dropna()
        s_w = winsorize(s, p=0.01)
        rows.append({
            "variable": col, "n_raw": int(len(s)),
            "mean_raw": float(s.mean()), "sd_raw": float(s.std()),
            "p25_raw": float(s.quantile(0.25)),
            "p50_raw": float(s.quantile(0.50)),
            "p75_raw": float(s.quantile(0.75)),
            "pct_zero": float((s == 0).mean()),
            "n_wins": int(len(s_w)),
            "mean_wins": float(s_w.mean()), "sd_wins": float(s_w.std()),
            "p25_wins": float(s_w.quantile(0.25)),
            "p50_wins": float(s_w.quantile(0.50)),
            "p75_wins": float(s_w.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", default="data/revision/metrics_1h_revision.parquet")
    p.add_argument("--out-dir", default="results/revision")
    p.add_argument("--window-days", type=int, default=30)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_parquet(args.metrics)
    log.info("loaded metrics: %d rows", len(metrics))

    daily = to_daily(metrics)
    daily.to_parquet(out / "daily_revision.parquet", index=False)

    summary_stats_table(metrics).to_csv(out / "summary_stats.csv", index=False)
    log.info("wrote summary_stats.csv")

    grid = build_did_full_grid(daily, args.window_days)
    grid.to_csv(out / "did_full_grid.csv", index=False)
    log.info("wrote did_full_grid.csv (%d rows)", len(grid))

    ddd = build_ddd_btc(daily, args.window_days)
    ddd.to_csv(out / "ddd_btc_approval.csv", index=False)
    log.info("wrote ddd_btc_approval.csv (%d rows)", len(ddd))

    es_frames = []
    for ev_id, ev_date, treated in EVENTS:
        if treated in ("BTC", "ETH"):
            es = event_study(daily, ev_id, ev_date, treated, OUTCOMES, args.window_days)
            if not es.empty:
                es_frames.append(es)
    if es_frames:
        es_all = pd.concat(es_frames, ignore_index=True)
        es_all.to_csv(out / "event_study_full.csv", index=False)
        log.info("wrote event_study_full.csv (%d rows)", len(es_all))


if __name__ == "__main__":
    main()
