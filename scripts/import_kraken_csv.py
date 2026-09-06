"""Import Kraken CSV bulk dump into our parquet schema.

Kraken bulk format (no header):
    unix_seconds, open, high, low, close, volume, count

File naming: <KRAKEN_TICKER>_<INTERVAL_MINUTES>.csv
For 1h: <TICKER>_60.csv

Mapping Kraken legacy tickers to our asset codes:
    XBT -> BTC
    XDG -> DOGE
    others (ETH, SOL, XRP, ADA, LTC, BCH, AVAX, LINK, DOT, UNI) match.

Output: data/raw/ccxt/ohlcv_<asset>_kraken_<quote>_1h.parquet
(reusing ccxt directory so build_revision_panel picks them up automatically).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kraken_import")


# Kraken legacy ticker -> our asset code
TICKER_MAP = {
    "XBT":  "BTC",
    "XDG":  "DOGE",
    "ETH":  "ETH",
    "SOL":  "SOL",
    "XRP":  "XRP",
    "ADA":  "ADA",
    "LTC":  "LTC",
    "BCH":  "BCH",
    "AVAX": "AVAX",
    "LINK": "LINK",
    "DOT":  "DOT",
    "UNI":  "UNI",
}

QUOTES = ["USD", "USDT", "USDC"]


def parse_filename(name: str) -> tuple[str, str, int] | None:
    """Return (kraken_ticker, quote, interval_min) or None."""
    if not name.endswith(".csv"):
        return None
    stem = name[:-4]
    if "_" not in stem:
        return None
    pair, interval = stem.rsplit("_", 1)
    try:
        interval_min = int(interval)
    except ValueError:
        return None
    # Try each known quote suffix.
    for q in QUOTES:
        if pair.endswith(q):
            asset = pair[: -len(q)]
            return asset, q, interval_min
    return None


def import_one(path: Path, asset: str, quote: str,
               start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            path, header=None,
            names=["unix", "open", "high", "low", "close", "volume", "count"],
            dtype={"unix": "int64", "open": "float64", "high": "float64",
                   "low": "float64", "close": "float64", "volume": "float64",
                   "count": "int64"},
        )
    except Exception as exc:
        log.warning("read fail %s: %s", path.name, exc)
        return None
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["unix"], unit="s", utc=True)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    if df.empty:
        return None
    df["asset"] = asset
    df["venue"] = "kraken"
    df["quote"] = quote
    df["interval"] = "1h"
    return df[["timestamp", "asset", "venue", "quote", "interval",
               "open", "high", "low", "close", "volume"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kraken-dir", default=os.environ.get("KRAKEN_DIR", "data/raw/kraken/master_q4"))
    p.add_argument("--out-dir", default="data/raw/ccxt")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-03-09")
    p.add_argument("--interval-min", type=int, default=60)
    args = p.parse_args()

    src = Path(args.kraken_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    written = 0
    skipped = 0
    for ticker, asset in TICKER_MAP.items():
        for quote in QUOTES:
            fname = f"{ticker}{quote}_{args.interval_min}.csv"
            path = src / fname
            if not path.exists():
                continue
            df = import_one(path, asset, quote, start_ts, end_ts)
            if df is None or df.empty:
                log.info("[%s/%s] %s -> empty in window", asset, quote, fname)
                skipped += 1
                continue
            out_path = out / f"ohlcv_{asset.lower()}_kraken_{quote.lower()}_1h.parquet"
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                           out_path, compression="zstd")
            log.info("[%s/%s] rows=%d -> %s", asset, quote, len(df), out_path.name)
            written += 1

    log.info("import complete: wrote=%d skipped=%d", written, skipped)


if __name__ == "__main__":
    main()
