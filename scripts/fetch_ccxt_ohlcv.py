"""Fetch 1h OHLCV from Coinbase, Kraken, OKX, Bybit (and Binance reference).

Output: data/raw/ccxt/ohlcv_{asset}_{venue}_{quote}_1h.parquet
"""
from __future__ import annotations

import asyncio
import argparse
import logging
from pathlib import Path

import ccxt.async_support as ccxt_async
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ccxt_fetch")


ASSETS = [
    "BTC", "ETH", "SOL", "XRP", "ADA",
    "DOGE", "LTC", "BCH", "AVAX", "LINK", "DOT", "UNI",
]

VENUES = {
    # venue_id : (ccxt_class_name, list_of_quotes_to_try_in_priority_order)
    "coinbase": ("coinbase", ["USD", "USDT", "USDC"]),
    "kraken":   ("kraken",   ["USD", "USDT"]),
    "okx":      ("okx",      ["USDT", "USDC"]),
    "bybit":    ("bybit",    ["USDT", "USDC"]),
    "binance":  ("binance",  ["USDT", "USDC", "FDUSD"]),
}

# Per-venue safe pagination limit; CCXT handles per-exchange max but we cap.
VENUE_LIMIT = {
    "coinbase": 300,
    "kraken":   720,
    "okx":      300,
    "bybit":    1000,
    "binance":  1000,
}

# Rate limit (sec) between paged requests per exchange instance.
VENUE_THROTTLE = {
    "coinbase": 0.40,
    "kraken":   1.10,
    "okx":      0.30,
    "bybit":    0.30,
    "binance":  0.30,
}


def to_ms(date_str: str) -> int:
    return int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)


async def fetch_one_pair(
    exchange,
    venue: str,
    asset: str,
    quote: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    out_dir: Path,
) -> bool:
    """Fetch the full history for asset/quote on this venue, save parquet.

    Returns True if data was written, False if symbol unsupported / empty.
    """
    symbol = f"{asset}/{quote}"
    if symbol not in exchange.markets:
        return False

    limit = VENUE_LIMIT.get(venue, 500)
    throttle = VENUE_THROTTLE.get(venue, 0.5)
    step_ms = 60 * 60 * 1000  # 1h

    rows: list[list[float]] = []
    cursor = start_ms
    total = 0
    while cursor < end_ms:
        try:
            batch = await exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        except Exception as exc:
            log.warning("[%s] %s @ %s err=%s — backing off", venue, symbol, cursor, exc)
            await asyncio.sleep(2.0)
            try:
                batch = await exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
            except Exception as exc2:
                log.error("[%s] %s @ %s persistent err=%s — abort pair", venue, symbol, cursor, exc2)
                return False

        if not batch:
            break

        # Drop bars beyond end_ms.
        batch = [b for b in batch if start_ms <= b[0] <= end_ms]
        if not batch:
            cursor += step_ms * limit
            continue

        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            cursor += step_ms
        else:
            cursor = last_ts + step_ms
        total += len(batch)

        if len(batch) < limit // 2 and last_ts > end_ms - step_ms * 12:
            break
        await asyncio.sleep(throttle)

    if not rows:
        return False

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["asset"] = asset
    df["venue"] = venue
    df["quote"] = quote
    df["interval"] = timeframe

    out_path = out_dir / f"ohlcv_{asset.lower()}_{venue}_{quote.lower()}_{timeframe}.parquet"
    table = pa.Table.from_pandas(
        df[["timestamp", "asset", "venue", "quote", "interval", "open", "high", "low", "close", "volume"]],
        preserve_index=False,
    )
    pq.write_table(table, out_path, compression="zstd")
    log.info("[%s] %s rows=%d -> %s", venue, symbol, len(df), out_path.name)
    return True


async def fetch_venue(
    venue: str,
    ccxt_cls: str,
    quotes: list[str],
    assets: list[str],
    timeframe: str,
    start_ms: int,
    end_ms: int,
    out_dir: Path,
):
    cls = getattr(ccxt_async, ccxt_cls)
    exchange = cls({
        "enableRateLimit": True,
        "timeout": 30000,
    })
    try:
        await exchange.load_markets()
        log.info("[%s] markets loaded n=%d", venue, len(exchange.markets))
        for asset in assets:
            written = False
            for quote in quotes:
                ok = await fetch_one_pair(
                    exchange, venue, asset, quote, timeframe, start_ms, end_ms, out_dir,
                )
                if ok:
                    written = True
                    # On Coinbase/Kraken USD is preferred — keep USDT too if exists.
                    # We let both write for downstream choice.
            if not written:
                log.warning("[%s] %s — no quote pair available among %s", venue, asset, quotes)
    finally:
        await exchange.close()


async def main_async(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)
    venues = args.venues if args.venues else list(VENUES.keys())
    assets = args.assets if args.assets else ASSETS

    log.info("CCXT fetcher: venues=%s assets=%s window=%s..%s",
             venues, len(assets), args.start, args.end)

    # Run venues in parallel — independent exchange clients.
    coros = []
    for v in venues:
        if v not in VENUES:
            log.warning("Unknown venue %s — skip", v)
            continue
        ccxt_cls, quotes = VENUES[v]
        coros.append(
            fetch_venue(v, ccxt_cls, quotes, assets, args.timeframe, start_ms, end_ms, out_dir)
        )
    await asyncio.gather(*coros, return_exceptions=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/ccxt")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-03-09")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--venues", nargs="*")
    p.add_argument("--assets", nargs="*")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
