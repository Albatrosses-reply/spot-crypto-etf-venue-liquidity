"""Scrape Farside Investors daily net flow tables for BTC and ETH spot ETFs.

Output: data/raw/etf/farside_btc_flows.csv, farside_eth_flows.csv
        data/raw/etf/farside_btc_flows.parquet, farside_eth_flows.parquet

Farside table layout (text):
    Date | IBIT | FBTC | BITB | ARKB | BTCO | EZBC | BRRR | HODL | BTCW |
    GBTC | BTC  | Total

Values are in USD millions. Negative numbers in parentheses. Empty -> NaN.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import cloudscraper  # type: ignore
from bs4 import BeautifulSoup  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("etf_flows")


URLS = {
    "btc": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
    "eth": "https://farside.co.uk/ethereum-etf-flow-all-data/",
}

UA = {"User-Agent": "Mozilla/5.0 (research; academic ETF flow analysis)"}


def parse_value(raw: str) -> float | None:
    s = raw.strip()
    if not s or s in ("-", "—", "–"):
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("$", "").strip()
    if not s or s == "-":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse_date(raw: str) -> pd.Timestamp | None:
    s = raw.strip()
    # Farside uses formats like "10 Jan 2024" or "Jan 10 2024".
    for fmt in ("%d %b %Y", "%b %d %Y", "%Y-%m-%d", "%d-%b-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except (ValueError, TypeError):
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, errors="coerce"))
    except Exception:
        return None


def fetch_table(asset: str, url: str) -> pd.DataFrame:
    log.info("[%s] GET %s", asset, url)
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )
    r = scraper.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Farside main table is the largest <table> on the page.
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("No <table> on Farside page")
    biggest = max(tables, key=lambda t: len(t.find_all("tr")))

    rows = biggest.find_all("tr")
    if len(rows) < 3:
        raise RuntimeError("Farside table appears empty")

    # Find header row that contains ticker symbols (skip empty / metadata rows).
    header = None
    for i, tr in enumerate(rows[:5]):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        non_empty = [c for c in cells if c]
        # Heuristic: header row has multiple ALL-CAPS ticker tokens.
        ticker_like = sum(1 for c in non_empty if c.isupper() and 2 <= len(c) <= 5)
        if ticker_like >= 3:
            header = cells
            break
    if header is None:
        # Fallback: use first row.
        header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    header = [c if c else f"col{i}" for i, c in enumerate(header)]
    log.info("[%s] header=%s", asset, header)

    body_rows = []
    skip_labels = {"fee", "total", "average", "minimum", "maximum", ""}
    for tr in rows:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells or len(cells) < 2:
            continue
        first = cells[0].strip().lower()
        if first in skip_labels:
            continue
        date = parse_date(cells[0])
        if date is None:
            continue
        record: dict[str, object] = {"date": date}
        for h, v in zip(header[1:], cells[1:]):
            record[h] = parse_value(v)
        body_rows.append(record)

    if not body_rows:
        raise RuntimeError("No body rows parsed from Farside table")

    df = pd.DataFrame(body_rows)
    # Drop NaT dates (summary footer rows that may have parsed as ambiguous text).
    df = df[df["date"].notna()].copy()
    # Drop fully-duplicate rows (Farside header layout sometimes echoes first row).
    df = df.drop_duplicates(subset="date", keep="first")
    # Drop columns named like cols0..colN that are clearly placeholder.
    placeholder = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("col")]
    df = df.drop(columns=placeholder, errors="ignore")
    df = df.sort_values("date").reset_index(drop=True)
    log.info("[%s] rows=%d cols=%d (incl date)", asset, len(df), df.shape[1])
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/raw/etf")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset, url in URLS.items():
        try:
            df = fetch_table(asset, url)
        except Exception as exc:
            log.error("[%s] failed: %s", asset, exc)
            continue
        csv_path = out_dir / f"farside_{asset}_flows.csv"
        pq_path = out_dir / f"farside_{asset}_flows.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)
        log.info("[%s] saved -> %s, %s", asset, csv_path.name, pq_path.name)


if __name__ == "__main__":
    main()
