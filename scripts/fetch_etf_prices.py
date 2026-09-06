"""Fetch ETF daily OHLCV via yfinance for spot BTC/ETH ETFs.

Output: data/raw/etf_prices/etf_prices_daily.parquet (long format)
        data/raw/etf_prices/etf_meta.csv (per-ETF metadata)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("etf_prices")


BTC_ETFS = [
    ("IBIT", "iShares Bitcoin Trust", "BlackRock", "spot_btc", "2024-01-11"),
    ("FBTC", "Fidelity Wise Origin Bitcoin Fund", "Fidelity", "spot_btc", "2024-01-11"),
    ("ARKB", "ARK 21Shares Bitcoin", "ARK/21Shares", "spot_btc", "2024-01-11"),
    ("BITB", "Bitwise Bitcoin", "Bitwise", "spot_btc", "2024-01-11"),
    ("HODL", "VanEck Bitcoin", "VanEck", "spot_btc", "2024-01-11"),
    ("BTCO", "Invesco Galaxy Bitcoin", "Invesco/Galaxy", "spot_btc", "2024-01-11"),
    ("EZBC", "Franklin Bitcoin", "Franklin Templeton", "spot_btc", "2024-01-11"),
    ("BRRR", "Valkyrie Bitcoin", "Valkyrie", "spot_btc", "2024-01-11"),
    ("BTCW", "WisdomTree Bitcoin", "WisdomTree", "spot_btc", "2024-01-11"),
    ("DEFI", "Hashdex Bitcoin", "Hashdex", "spot_btc", "2024-01-11"),
    ("GBTC", "Grayscale Bitcoin Trust ETF", "Grayscale", "spot_btc", "2013-09-25"),
    ("BTC",  "Grayscale Bitcoin Mini Trust", "Grayscale", "spot_btc", "2024-07-31"),
    # Reference futures-based:
    ("BITO", "ProShares Bitcoin Strategy", "ProShares", "futures_btc", "2021-10-19"),
]
ETH_ETFS = [
    ("ETHA", "iShares Ethereum Trust", "BlackRock", "spot_eth", "2024-07-23"),
    ("FETH", "Fidelity Ethereum Fund", "Fidelity", "spot_eth", "2024-07-23"),
    ("ETHW", "Bitwise Ethereum", "Bitwise", "spot_eth", "2024-07-23"),
    ("ETHV", "VanEck Ethereum", "VanEck", "spot_eth", "2024-07-23"),
    ("QETH", "Invesco Galaxy Ethereum", "Invesco/Galaxy", "spot_eth", "2024-07-23"),
    ("EZET", "Franklin Ethereum", "Franklin Templeton", "spot_eth", "2024-07-23"),
    ("CETH", "21Shares Core Ethereum", "21Shares", "spot_eth", "2024-07-23"),
    ("ETHE", "Grayscale Ethereum Trust ETF", "Grayscale", "spot_eth", "2017-12-14"),
    ("ETH",  "Grayscale Ethereum Mini Trust", "Grayscale", "spot_eth", "2024-07-23"),
]
ALL_ETFS = BTC_ETFS + ETH_ETFS


def fetch_one(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                         progress=False, threads=False)
    except Exception as exc:
        log.error("[%s] yf err: %s", ticker, exc)
        return None
    if df is None or df.empty:
        log.warning("[%s] empty", ticker)
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    df["ticker"] = ticker
    return df[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/etf_prices")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-03-15")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    meta_rows = []
    for ticker, name, issuer, kind, launch in ALL_ETFS:
        df = fetch_one(ticker, args.start, args.end)
        if df is None:
            continue
        frames.append(df)
        meta_rows.append({
            "ticker": ticker, "name": name, "issuer": issuer,
            "kind": kind, "launch": launch, "rows": len(df),
        })
        log.info("[%s] rows=%d", ticker, len(df))

    if not frames:
        raise SystemExit("No ETF data fetched.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    out_pq = out_dir / "etf_prices_daily.parquet"
    out_csv = out_dir / "etf_prices_daily.csv"
    panel.to_parquet(out_pq, index=False)
    panel.to_csv(out_csv, index=False)
    log.info("panel rows=%d, tickers=%d -> %s", len(panel), panel["ticker"].nunique(), out_pq)

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "etf_meta.csv", index=False)
    log.info("meta saved: %d ETFs", len(meta))


if __name__ == "__main__":
    main()
