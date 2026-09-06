"""Commonality robustness: window variants, exclusions, placebo, plots.

Outputs:
  - results/revision/commonality_window_robustness.csv
  - results/revision/commonality_exclusion_robustness.csv
  - results/revision/commonality_placebo.csv
  - results/revision/figures/commonality_timeseries.png
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_commonality import (
    EVENTS, build_return_panel, rolling_commonality, event_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("commonality_robust")


def window_robustness(daily: pd.DataFrame) -> pd.DataFrame:
    wide = build_return_panel(daily)
    rows = []
    for w in (15, 30, 60):
        log.info("window=%dd", w)
        comm = rolling_commonality(wide, window=w)
        summ = event_summary(comm, window_days=30)
        summ["roll_window"] = w
        rows.append(summ)
    return pd.concat(rows, ignore_index=True)


def exclusion_robustness(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop one venue or one asset at a time; check headline event delta."""
    rows = []
    for drop_kind, drop_value in [
        ("venue", "binance"), ("venue", "coinbase"), ("venue", "kraken"),
        ("venue", "bybit"), ("venue", "okx"), ("venue", "bitstamp"),
        ("asset", "BTC"), ("asset", "ETH"), ("asset", "SOL"), ("asset", "DOGE"),
    ]:
        sub = daily[daily[drop_kind] != drop_value].copy()
        wide = build_return_panel(sub)
        comm = rolling_commonality(wide, window=30)
        summ = event_summary(comm, window_days=30)
        summ["dropped"] = f"{drop_kind}={drop_value}"
        rows.append(summ)
    return pd.concat(rows, ignore_index=True)


def placebo_dates(daily: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Random placebo dates in the pre-ETF era; expect no commonality jump."""
    wide = build_return_panel(daily)
    comm = rolling_commonality(wide, window=30)
    pre_etf_start = pd.Timestamp("2023-04-01")
    pre_etf_end   = pd.Timestamp("2023-12-15")
    rng = np.random.default_rng(seed)
    candidates = comm[(comm["date"] >= pre_etf_start) & (comm["date"] <= pre_etf_end)]
    candidates = candidates.dropna(subset=["C_pairwise"])
    if candidates.empty:
        return pd.DataFrame()
    sample = candidates.sample(n=min(n, len(candidates)), random_state=seed)
    rows = []
    for _, r in sample.iterrows():
        ev = r["date"]
        sub = comm[(comm["date"] >= ev - pd.Timedelta(days=30)) &
                   (comm["date"] <= ev + pd.Timedelta(days=30))]
        pre = sub[sub["date"] < ev]
        post = sub[sub["date"] >= ev]
        if pre.empty or post.empty:
            continue
        for col in ["C_pairwise", "C_pca"]:
            pre_m = float(pre[col].mean())
            post_m = float(post[col].mean())
            pre_s = float(pre[col].std(ddof=1))
            post_s = float(post[col].std(ddof=1))
            n_pre, n_post = int(pre[col].count()), int(post[col].count())
            if n_pre > 1 and n_post > 1 and (pre_s > 0 or post_s > 0):
                pooled = np.sqrt(pre_s ** 2 / n_pre + post_s ** 2 / n_post)
                t = (post_m - pre_m) / max(pooled, 1e-12)
            else:
                t = np.nan
            rows.append({
                "placebo_date": str(ev.date()), "measure": col,
                "pre_mean": pre_m, "post_mean": post_m,
                "delta": post_m - pre_m, "welch_t": float(t) if t == t else np.nan,
            })
    return pd.DataFrame(rows)


def plot_timeseries(daily: pd.DataFrame, out_path: Path):
    wide = build_return_panel(daily)
    comm = rolling_commonality(wide, window=30)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(comm["date"], comm["C_pairwise"], label="Avg pairwise correlation", lw=1.6)
    ax.plot(comm["date"], comm["C_pca"], label="First PCA factor share", lw=1.2, alpha=0.7)
    colors = {"btc": "tab:blue", "eth": "tab:purple", "mica": "tab:red"}
    for ev_id, ev_date in EVENTS:
        ev = pd.Timestamp(ev_date)
        if ev_id.startswith("btc"):
            c = colors["btc"]
        elif ev_id.startswith("eth"):
            c = colors["eth"]
        else:
            c = colors["mica"]
        ax.axvline(ev, color=c, linestyle=":", alpha=0.55, lw=1.0)
        ax.text(ev, 0.04, ev_id.replace("_", "\n"),
                rotation=90, va="bottom", ha="right", fontsize=7,
                color=c, transform=ax.get_xaxis_transform())
    ax.set_ylabel("Cross-series commonality")
    ax.set_xlabel("Date")
    ax.set_title("Cross-asset, cross-venue return commonality "
                 "(12 assets × 9 venues, 30-day rolling)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="results/revision/daily_revision.parquet")
    p.add_argument("--out-dir", default="results/revision")
    args = p.parse_args()

    daily = pd.read_parquet(args.daily)
    out = Path(args.out_dir)

    log.info("running window robustness...")
    wr = window_robustness(daily)
    wr.to_csv(out / "commonality_window_robustness.csv", index=False)

    log.info("running exclusion robustness (10 drops × full pipeline)...")
    er = exclusion_robustness(daily)
    er.to_csv(out / "commonality_exclusion_robustness.csv", index=False)

    log.info("running placebo dates...")
    pl = placebo_dates(daily, n=20)
    pl.to_csv(out / "commonality_placebo.csv", index=False)

    log.info("plotting time series...")
    plot_timeseries(daily, out / "figures" / "commonality_timeseries.png")
    log.info("done.")


if __name__ == "__main__":
    main()
