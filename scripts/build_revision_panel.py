"""Build the revised 9-venue panel for the JIFMIM resubmission.

Inputs:
  - data/raw/hardened/{venue}/ohlcv_{asset}_1h.parquet   (5 venues from prior run)
  - data/raw/ccxt/ohlcv_{asset}_{venue}_{quote}_1h.parquet (new venues)

Outputs:
  - data/revision/panel_1h_revision.parquet         (long OHLCV panel + notional_volume)
  - data/revision/metrics_1h_revision.parquet       (per-asset-venue-time metrics)
  - data/revision/coverage_summary.csv              (asset × venue × quote coverage)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("revision_panel")


REQUIRED = ["timestamp", "asset", "venue", "quote", "interval",
            "open", "high", "low", "close", "volume"]

# Venues we re-fetched via CCXT — for these we PREFER ccxt data over hardened
# (since CCXT covers multiple quotes and is more current).
CCXT_VENUES_TO_USE = {"coinbase", "kraken", "okx", "bybit", "binance"}
# Kraken now sourced from CSV bulk import (Q4 2025 dump) and stored as kraken parquets.
# Venues only available in hardened set (CCXT does not cover or we didn't add).
HARDENED_ONLY = {"bitfinex", "bitstamp", "gemini", "poloniex"}


def read_one(p: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        log.warning("read fail %s: %s", p.name, exc)
        return None
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        log.warning("schema mismatch %s missing %s", p.name, miss)
        return None
    df = df[REQUIRED].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["asset"] = df["asset"].astype(str).str.upper()
    df["venue"] = df["venue"].astype(str).str.lower()
    df["quote"] = df["quote"].astype(str).str.upper()
    df["interval"] = df["interval"].astype(str)
    return df


def load_hardened(root: Path) -> pd.DataFrame:
    frames = []
    for vdir in sorted(root.iterdir()):
        if not vdir.is_dir() or vdir.name not in HARDENED_ONLY:
            continue
        for f in sorted(vdir.glob("ohlcv_*_1h.parquet")):
            df = read_one(f)
            if df is None or df.empty:
                continue
            # Force venue from directory name to be safe.
            df["venue"] = vdir.name
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED)
    pool = pd.concat(frames, ignore_index=True)
    log.info("hardened venues=%d rows=%d", pool["venue"].nunique(), len(pool))
    return pool


def load_ccxt(root: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(root.glob("ohlcv_*_*_*_1h.parquet")):
        df = read_one(f)
        if df is None or df.empty:
            continue
        if (df["venue"].iloc[0] not in CCXT_VENUES_TO_USE):
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=REQUIRED)
    pool = pd.concat(frames, ignore_index=True)
    log.info("ccxt venues=%d rows=%d", pool["venue"].nunique(), len(pool))
    return pool


def assemble_panel(hardened_root: Path, ccxt_root: Path) -> pd.DataFrame:
    h = load_hardened(hardened_root)
    c = load_ccxt(ccxt_root)
    panel = pd.concat([h, c], ignore_index=True)
    panel = panel.sort_values(["timestamp", "asset", "venue", "quote"])
    # Dedup: same asset×venue×quote×timestamp — keep last (newer fetch wins).
    panel = panel.drop_duplicates(subset=["timestamp", "asset", "venue", "quote"], keep="last")
    panel = panel.reset_index(drop=True)
    panel["notional_volume"] = panel["close"] * panel["volume"]
    log.info("merged panel: %d rows, %d venues, %d assets, quotes=%s",
             len(panel), panel["venue"].nunique(), panel["asset"].nunique(),
             sorted(panel["quote"].unique()))
    return panel


# -------------------------------------------------------------------- metrics


def _autocorr(x: pd.Series, lag: int = 1) -> float:
    if len(x) <= lag:
        return np.nan
    return float(x.autocorr(lag=lag))


def _variance_ratio(prices: pd.Series, h: int = 2) -> float:
    r1 = prices.pct_change().dropna()
    rh = prices.pct_change(periods=h).dropna()
    if len(r1) < 2 or len(rh) < 2:
        return np.nan
    denom = h * r1.var(ddof=1)
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(rh.var(ddof=1) / denom)


def _realized_vol(arr: np.ndarray) -> float:
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan
    return float(np.sqrt(np.sum(np.square(arr))))


def _corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """High-low based spread proxy of Corwin & Schultz (2012).

    Returns a sequence of paired-day spreads aligned to t (uses bars t-1 and t).
    Output length matches input length, first value = NaN.
    """
    n = len(high)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    # gamma and beta as in C-S (2012). Use 2 consecutive bars.
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


def compute_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel.copy()
    work = work.sort_values(["asset", "venue", "quote", "timestamp"]).reset_index(drop=True)

    keys = ["asset", "venue", "quote"]
    work["simple_return"] = work.groupby(keys, sort=False)["close"].pct_change()
    work["log_return"] = work.groupby(keys, sort=False)["close"].transform(
        lambda s: np.log(s).diff()
    )

    # Realized vol — rolling 24-hour log-return RV.
    work["realized_volatility"] = work.groupby(keys, sort=False)["log_return"].transform(
        lambda s: s.rolling(24, min_periods=2).apply(_realized_vol, raw=True)
    )

    # Amihud illiquidity — exclude zero-volume bars by setting denom NaN.
    notional = work["notional_volume"].where(work["notional_volume"] > 0)
    work["amihud_illiquidity"] = work["simple_return"].abs() / notional

    # High-low range bps.
    work["high_low_range_bps"] = (
        (work["high"] - work["low"]) / work["close"].replace(0.0, np.nan)
    ) * 10000.0

    # Rolling 24h notional depth proxy.
    work["rolling_notional_depth_proxy_24h"] = work.groupby(keys, sort=False)["notional_volume"].transform(
        lambda s: s.rolling(24, min_periods=1).sum()
    )

    # Corwin-Schultz spread proxy (per asset×venue×quote).
    cs_blocks = []
    for k, g in work.groupby(keys, sort=False):
        cs = _corwin_schultz_spread(g["high"].to_numpy(), g["low"].to_numpy())
        cs_blocks.append(pd.Series(cs, index=g.index))
    work["corwin_schultz_spread"] = pd.concat(cs_blocks).sort_index()

    # Cross-venue dispersion (per asset×timestamp×quote).
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
    log.info("metrics computed: %d rows", len(metrics))
    return metrics


# -------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hardened-root", default="data/raw/hardened")
    p.add_argument("--ccxt-root", default="data/raw/ccxt")
    p.add_argument("--out-dir", default="data/revision")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = assemble_panel(Path(args.hardened_root), Path(args.ccxt_root))
    panel.to_parquet(out_dir / "panel_1h_revision.parquet", index=False)
    log.info("wrote %s", (out_dir / "panel_1h_revision.parquet").name)

    coverage = (
        panel.groupby(["asset", "venue", "quote"])
        .agg(
            rows=("timestamp", "size"),
            t_min=("timestamp", "min"),
            t_max=("timestamp", "max"),
        )
        .reset_index()
    )
    coverage.to_csv(out_dir / "coverage_summary.csv", index=False)

    metrics = compute_metrics(panel)
    metrics.to_parquet(out_dir / "metrics_1h_revision.parquet", index=False)
    log.info("wrote %s", (out_dir / "metrics_1h_revision.parquet").name)


if __name__ == "__main__":
    main()
