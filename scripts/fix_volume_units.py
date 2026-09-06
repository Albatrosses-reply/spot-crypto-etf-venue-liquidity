"""Path E — Fix volume-unit bug and rebuild panel.

Diagnosis:
  - 8 of 9 venues report `volume` in BASE currency (BTC, ETH, ...).
    Notional USD = close * volume.
  - Poloniex (CDD source only) has SWAPPED column labels: the column we read
    as "Volume BTC" actually contains the QUOTE (USDT) amount, and vice versa.
    For Poloniex, our existing `volume` column IS already in USD/USDT.
    → Notional USD = volume (no multiplication by close).

We rebuild the panel with a single corrected `notional_volume` column that is
consistent across all venues. Other panel fields (open/high/low/close,
log_return) are unaffected since they are price-based.

Outputs:
  - data/revision/panel_1h_revision_fixed.parquet
  - data/revision/metrics_1h_revision_fixed.parquet
  - data/revision/coverage_summary_fixed.csv
  - data/revision/volume_unit_diagnostic.csv (per-venue sanity table)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fix_volume")


# Venues whose stored `volume` column is in QUOTE currency (already USD/USDT).
QUOTE_VOLUME_VENUES = {"poloniex"}


def fix_panel(panel_path: Path, out_path: Path) -> pd.DataFrame:
    log.info("loading panel from %s", panel_path.name)
    panel = pd.read_parquet(panel_path)
    log.info("rows=%d", len(panel))

    # Recompute notional_volume with venue-specific rule.
    is_quote_vol = panel["venue"].isin(QUOTE_VOLUME_VENUES)
    panel["notional_volume_old"] = panel["notional_volume"]  # keep old for diagnostic
    panel.loc[~is_quote_vol, "notional_volume"] = (
        panel.loc[~is_quote_vol, "close"] * panel.loc[~is_quote_vol, "volume"]
    )
    # For Poloniex: stored volume is already USD/USDT
    panel.loc[is_quote_vol, "notional_volume"] = panel.loc[is_quote_vol, "volume"]

    log.info("notional_volume distribution per venue (median, hourly USD):")
    g = panel.groupby("venue")["notional_volume"].median()
    for v, m in g.sort_values(ascending=False).items():
        log.info("  %-10s %.2e", v, m)

    panel = panel.drop(columns=["notional_volume_old"])
    panel.to_parquet(out_path, index=False)
    log.info("wrote %s (%d rows)", out_path.name, len(panel))
    return panel


def _corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Corwin & Schultz (2012) high-low spread proxy."""
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    h_pair = np.maximum(high[1:], high[:-1])
    l_pair = np.minimum(low[1:], low[:-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (np.log(high[1:] / low[1:])) ** 2 + (np.log(high[:-1] / low[:-1])) ** 2
        gamma = (np.log(h_pair / l_pair)) ** 2
        denom = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)
    out[1:] = spread
    return out


def recompute_metrics(panel: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Recompute Amihud, depth, CS spread, dispersion with corrected notional."""
    work = panel.copy()
    work = work.sort_values(["asset", "venue", "quote", "timestamp"]).reset_index(drop=True)

    keys = ["asset", "venue", "quote"]
    work["simple_return"] = work.groupby(keys, sort=False)["close"].pct_change()
    work["log_return"] = work.groupby(keys, sort=False)["close"].transform(
        lambda s: np.log(s).diff()
    )

    # Realized vol — rolling 24h.
    def _rv(arr):
        a = arr[~np.isnan(arr)]
        return float(np.sqrt(np.sum(a ** 2))) if len(a) >= 2 else np.nan
    work["realized_volatility"] = work.groupby(keys, sort=False)["log_return"].transform(
        lambda s: s.rolling(24, min_periods=2).apply(_rv, raw=True)
    )

    # Robust Amihud (zero-volume cells -> NaN).
    notional = work["notional_volume"].where(work["notional_volume"] > 0)
    work["amihud_illiquidity"] = work["simple_return"].abs() / notional

    # High-low range bps and rolling 24h depth proxy.
    work["high_low_range_bps"] = (
        (work["high"] - work["low"]) / work["close"].replace(0.0, np.nan)
    ) * 10000.0
    work["rolling_notional_depth_proxy_24h"] = work.groupby(keys, sort=False)["notional_volume"].transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )

    # Corwin-Schultz spread (per asset×venue×quote).
    cs_blocks = []
    for k, g in work.groupby(keys, sort=False):
        cs = _corwin_schultz_spread(g["high"].to_numpy(), g["low"].to_numpy())
        cs_blocks.append(pd.Series(cs, index=g.index))
    work["corwin_schultz_spread"] = pd.concat(cs_blocks).sort_index()

    # Cross-venue dispersion (price-based, unchanged from previous version).
    disp = (
        work.groupby(["timestamp", "asset", "quote"], as_index=False)
        .agg(
            venue_count=("venue", "nunique"),
            close_max=("close", "max"),
            close_min=("close", "min"),
            close_mean=("close", "mean"),
        )
    )
    disp["dispersion_bps"] = (
        (disp["close_max"] - disp["close_min"]) / disp["close_mean"].replace(0.0, np.nan)
    ) * 10000.0
    disp = disp[["timestamp", "asset", "quote", "venue_count", "dispersion_bps"]]
    metrics = work.merge(disp, on=["timestamp", "asset", "quote"], how="left")

    metrics.to_parquet(out_path, index=False)
    log.info("wrote %s (%d rows)", out_path.name, len(metrics))
    return metrics


def diagnostic(panel_old: pd.DataFrame, panel_new: pd.DataFrame, out_path: Path):
    """Per-venue median hourly notional, before vs after fix."""
    rows = []
    for v in sorted(panel_new["venue"].unique()):
        old_med = panel_old[panel_old["venue"] == v]["notional_volume"].median()
        new_med = panel_new[panel_new["venue"] == v]["notional_volume"].median()
        rows.append({
            "venue": v,
            "old_median_notional_per_hour": float(old_med),
            "new_median_notional_per_hour": float(new_med),
            "ratio_old_to_new": float(old_med / new_med) if new_med > 0 else np.nan,
            "fix_applied": v in QUOTE_VOLUME_VENUES,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("wrote diagnostic %s", out_path.name)


def coverage(panel: pd.DataFrame, out_path: Path):
    cov = (
        panel.groupby(["asset", "venue", "quote"])
        .agg(rows=("timestamp", "size"),
             t_min=("timestamp", "min"),
             t_max=("timestamp", "max"))
        .reset_index()
    )
    cov.to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-panel", default="data/revision/panel_1h_revision.parquet")
    p.add_argument("--out-dir", default="data/revision")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_path = Path(args.in_panel)
    panel_old = pd.read_parquet(in_path)

    panel_new_path = out_dir / "panel_1h_revision_fixed.parquet"
    panel_new = fix_panel(in_path, panel_new_path)

    metrics_path = out_dir / "metrics_1h_revision_fixed.parquet"
    recompute_metrics(panel_new, metrics_path)

    diag_path = out_dir / "volume_unit_diagnostic.csv"
    diagnostic(panel_old, panel_new, diag_path)

    cov_path = out_dir / "coverage_summary_fixed.csv"
    coverage(panel_new, cov_path)


if __name__ == "__main__":
    main()
