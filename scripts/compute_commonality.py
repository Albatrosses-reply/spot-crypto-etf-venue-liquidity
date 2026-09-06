"""Compute cross-asset, cross-venue return commonality for the resubmission pivot.

Methodology follows Brauneis, Mestel, Riordan & Theissen (2021, JFM) and
Karolyi, Lee & Van Dijk (2012, JFE).

Daily commonality at time t is the average pairwise correlation among
N return series computed over a rolling window [t-w, t-1].

We compute two variants for robustness:
  C1_t = average pairwise Pearson correlation
  C2_t = first-PCA-factor variance share

Outputs:
  - results/revision/commonality_daily.parquet  (date, C1, C2, N_used)
  - results/revision/commonality_event_summary.csv (pre/post means around each event)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("commonality")


EVENTS = [
    ("btc_etf_approval",   "2024-01-10"),
    ("btc_etf_launch",     "2024-01-11"),
    ("eth_rule_approval",  "2024-05-23"),
    ("eth_etf_launch",     "2024-07-23"),
    ("mica_stablecoin",    "2024-06-30"),
    ("mica_full",          "2024-12-30"),
]


def build_return_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Wide panel of daily log returns indexed by date with (asset, venue)
    column tuples. We use the previously aggregated daily panel where one
    quote per (asset, venue, date) has already been selected by priority."""
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    d = d.sort_values(["asset", "venue", "date"])
    d["log_close"] = np.log(d["close"].clip(lower=1e-12))
    d["log_return"] = d.groupby(["asset", "venue"])["log_close"].diff()
    wide = d.pivot_table(
        index="date", columns=["asset", "venue"], values="log_return", aggfunc="mean"
    )
    wide.columns = [f"{a}_{v}" for (a, v) in wide.columns]
    return wide


def commonality_pairwise(window_returns: np.ndarray) -> tuple[float, int]:
    """Average pairwise correlation of a returns matrix."""
    if window_returns.shape[0] < 5:
        return np.nan, 0
    # Drop columns that are entirely NaN or constant.
    valid = []
    for j in range(window_returns.shape[1]):
        col = window_returns[:, j]
        if np.isnan(col).all() or np.nanstd(col) == 0:
            continue
        valid.append(j)
    if len(valid) < 3:
        return np.nan, len(valid)
    sub = window_returns[:, valid]
    # Use numpy corrcoef on the columns; handle NaN by pairwise complete.
    n = sub.shape[1]
    rho = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            xi = sub[:, i]; xj = sub[:, j]
            mask = ~np.isnan(xi) & ~np.isnan(xj)
            if mask.sum() < 5:
                continue
            xi_m = xi[mask]; xj_m = xj[mask]
            sd_i = xi_m.std(); sd_j = xj_m.std()
            if sd_i == 0 or sd_j == 0:
                continue
            rho[i, j] = np.mean(((xi_m - xi_m.mean()) / sd_i) * ((xj_m - xj_m.mean()) / sd_j))
    iu = np.triu_indices(n, k=1)
    vals = rho[iu]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, len(valid)
    return float(np.mean(vals)), len(valid)


def commonality_pca(window_returns: np.ndarray) -> tuple[float, int]:
    """Variance share of first principal component on a returns matrix."""
    if window_returns.shape[0] < 5:
        return np.nan, 0
    valid = []
    for j in range(window_returns.shape[1]):
        col = window_returns[:, j]
        if np.isnan(col).all() or np.nanstd(col) == 0:
            continue
        valid.append(j)
    if len(valid) < 3:
        return np.nan, len(valid)
    sub = window_returns[:, valid]
    # Fill remaining NaN with column means before PCA (necessary for stable eigendecomp).
    col_mean = np.nanmean(sub, axis=0)
    inds = np.where(np.isnan(sub))
    sub[inds] = np.take(col_mean, inds[1])
    # Standardize columns.
    std = sub.std(axis=0)
    std[std == 0] = 1.0
    z = (sub - sub.mean(axis=0)) / std
    cov = np.cov(z.T)
    if cov.shape[0] < 2:
        return np.nan, len(valid)
    try:
        eig = np.linalg.eigvalsh(cov)
    except Exception:
        return np.nan, len(valid)
    eig = np.sort(eig)[::-1]
    total = float(eig.sum())
    if total <= 0:
        return np.nan, len(valid)
    return float(eig[0] / total), len(valid)


def rolling_commonality(returns_wide: pd.DataFrame, window: int = 30
                        ) -> pd.DataFrame:
    arr = returns_wide.to_numpy(dtype=float)
    dates = returns_wide.index.to_numpy()
    n_t = arr.shape[0]
    out_pairwise = np.full(n_t, np.nan)
    out_pca = np.full(n_t, np.nan)
    out_n = np.zeros(n_t, dtype=int)
    for t in range(window, n_t):
        win = arr[t - window:t, :]
        c1, n1 = commonality_pairwise(win.copy())
        c2, n2 = commonality_pca(win.copy())
        out_pairwise[t] = c1
        out_pca[t] = c2
        out_n[t] = max(n1, n2)
    return pd.DataFrame({
        "date": dates,
        "C_pairwise": out_pairwise,
        "C_pca": out_pca,
        "N_series": out_n,
    })


def event_summary(comm: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    rows = []
    for ev_id, ev_date in EVENTS:
        ev = pd.Timestamp(ev_date)
        pre_lo = ev - pd.Timedelta(days=window_days)
        post_hi = ev + pd.Timedelta(days=window_days)
        sub = comm[(comm["date"] >= pre_lo) & (comm["date"] <= post_hi)]
        pre = sub[sub["date"] < ev]
        post = sub[sub["date"] >= ev]
        if pre.empty or post.empty:
            continue
        for col in ["C_pairwise", "C_pca"]:
            pre_mean = float(pre[col].mean())
            post_mean = float(post[col].mean())
            pre_sd = float(pre[col].std(ddof=1))
            post_sd = float(post[col].std(ddof=1))
            n_pre = int(pre[col].count())
            n_post = int(post[col].count())
            # Welch t-test approximation
            if n_pre > 1 and n_post > 1 and (pre_sd > 0 or post_sd > 0):
                pooled = np.sqrt(pre_sd ** 2 / n_pre + post_sd ** 2 / n_post)
                t = (post_mean - pre_mean) / max(pooled, 1e-12)
            else:
                t = np.nan
            rows.append({
                "event_id": ev_id, "event_date": ev_date, "measure": col,
                "pre_mean": pre_mean, "post_mean": post_mean,
                "delta": post_mean - pre_mean,
                "pre_sd": pre_sd, "post_sd": post_sd,
                "n_pre": n_pre, "n_post": n_post, "welch_t": float(t) if t == t else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--daily", default="results/revision/daily_revision.parquet")
    p.add_argument("--out-dir", default="results/revision")
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(args.daily)
    log.info("daily panel: %d rows", len(daily))

    wide = build_return_panel(daily)
    log.info("return panel: dates=%d, series=%d", wide.shape[0], wide.shape[1])

    comm = rolling_commonality(wide, window=args.window)
    comm.to_parquet(out / "commonality_daily.parquet", index=False)
    log.info("commonality computed: %d days", len(comm))

    summary = event_summary(comm, window_days=30)
    summary.to_csv(out / "commonality_event_summary.csv", index=False)
    log.info("event summary written: %d rows", len(summary))

    # Also dump quick aggregate summary for the user.
    print(summary.to_string())


if __name__ == "__main__":
    main()
