"""Regime-shift identification for cross-venue commonality, robust to autocorrelation.

The naive Welch t-test on rolling commonality has 75% false-positive rate at random
pre-ETF dates due to high autocorrelation in the rolling series. Two cleaner tests:

  (1) Era-level comparison: pre-ETF (Apr-Dec 2023) vs post-ETF (Feb 2024 - Mar 2026)
      Tests whether the commonality regime *changed*, not whether it jumped at a
      single date.

  (2) Empirical p-value via permutation: for each event, how does the observed delta
      compare to a distribution of delta values obtained by shuffling the date axis?

  (3) Quandt-Andrews structural-break test: identify the data-driven break point and
      check whether it lies near our event dates.

  (4) Newey-West HAC SE for the Welch t-test (autocorrelation-corrected).

Outputs:
  - results/revision/commonality_regime_shift.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("regime_shift")


PRE_ETF_START = pd.Timestamp("2023-04-01")
PRE_ETF_END   = pd.Timestamp("2023-12-15")
POST_ETF_START = pd.Timestamp("2024-02-15")
POST_ETF_END   = pd.Timestamp("2026-03-09")

EVENTS = [
    ("btc_etf_approval",   "2024-01-10"),
    ("btc_etf_launch",     "2024-01-11"),
    ("eth_rule_approval",  "2024-05-23"),
    ("eth_etf_launch",     "2024-07-23"),
    ("mica_stablecoin",    "2024-06-30"),
    ("mica_full",          "2024-12-30"),
]


def newey_west_se(series: pd.Series, lags: int = 5) -> float:
    """Newey-West HAC standard error of the mean for autocorrelated series."""
    x = series.dropna().to_numpy()
    n = len(x)
    if n < 5:
        return np.nan
    mean = x.mean()
    e = x - mean
    gamma = [np.dot(e, e) / n]
    for k in range(1, min(lags, n - 1) + 1):
        gamma.append(np.dot(e[k:], e[:-k]) / n)
    var_nw = gamma[0] + 2 * sum(
        (1 - k / (lags + 1)) * gamma[k] for k in range(1, len(gamma))
    )
    var_mean = var_nw / n
    return float(np.sqrt(max(var_mean, 0.0)))


def era_test(comm: pd.DataFrame) -> pd.DataFrame:
    """Pre-ETF vs post-ETF era mean comparison with HAC SE."""
    rows = []
    for col in ("C_pairwise", "C_pca"):
        pre = comm[(comm["date"] >= PRE_ETF_START) & (comm["date"] <= PRE_ETF_END)][col]
        post = comm[(comm["date"] >= POST_ETF_START) & (comm["date"] <= POST_ETF_END)][col]
        m_pre = float(pre.mean())
        m_post = float(post.mean())
        se_pre = newey_west_se(pre, lags=20)
        se_post = newey_west_se(post, lags=20)
        delta = m_post - m_pre
        se_diff = float(np.sqrt(se_pre ** 2 + se_post ** 2))
        t = delta / max(se_diff, 1e-12) if se_diff and se_diff > 0 else np.nan
        rows.append({
            "test": "era_pre_vs_post", "measure": col,
            "pre_mean": m_pre, "post_mean": m_post, "delta": delta,
            "pre_n": int(pre.count()), "post_n": int(post.count()),
            "pre_nw_se": float(se_pre) if not np.isnan(se_pre) else np.nan,
            "post_nw_se": float(se_post) if not np.isnan(se_post) else np.nan,
            "nw_t": float(t) if t == t else np.nan,
        })
    return pd.DataFrame(rows)


def empirical_pvalue(comm: pd.DataFrame, n_perm: int = 1000, seed: int = 42) -> pd.DataFrame:
    """For each EVENT, compute empirical p-value of observed Δ against
    permutation distribution of random-date Δs (with the same window definition)."""
    rng = np.random.default_rng(seed)
    rows = []
    valid_dates = comm["date"].to_numpy()
    for ev_id, ev_date in EVENTS:
        ev = pd.Timestamp(ev_date)
        sub = comm[(comm["date"] >= ev - pd.Timedelta(days=30)) &
                   (comm["date"] <= ev + pd.Timedelta(days=30))]
        for col in ("C_pairwise", "C_pca"):
            pre = sub[sub["date"] < ev][col]
            post = sub[sub["date"] >= ev][col]
            if pre.empty or post.empty:
                continue
            obs_delta = float(post.mean() - pre.mean())

            # Permutation: pick random dates within the pre-ETF era and compute
            # the same statistic; BTC and ETH events are dense in 2024 so we
            # restrict permutations to 2023-04-01..2023-12-15.
            valid = comm[(comm["date"] >= PRE_ETF_START) & (comm["date"] <= PRE_ETF_END)]
            valid = valid.dropna(subset=[col])
            valid_arr = valid[col].to_numpy()
            valid_dt = valid["date"].to_numpy()
            if len(valid_dt) < 60:
                continue
            perm_deltas = []
            for _ in range(n_perm):
                idx = rng.integers(30, len(valid_dt) - 30)
                perm_pre = valid_arr[idx - 30:idx]
                perm_post = valid_arr[idx:idx + 30]
                if len(perm_pre) == 0 or len(perm_post) == 0:
                    continue
                perm_deltas.append(float(perm_post.mean() - perm_pre.mean()))
            if not perm_deltas:
                continue
            perm_arr = np.array(perm_deltas)
            # Two-sided empirical p
            p_two = float((np.abs(perm_arr) >= abs(obs_delta)).mean())
            p_right = float((perm_arr >= obs_delta).mean())
            rows.append({
                "test": "empirical_p", "event_id": ev_id, "event_date": ev_date,
                "measure": col, "obs_delta": obs_delta,
                "perm_mean": float(perm_arr.mean()), "perm_sd": float(perm_arr.std()),
                "perm_p95": float(np.percentile(np.abs(perm_arr), 95)),
                "p_two_sided": p_two, "p_right_tail": p_right,
                "n_perm": int(len(perm_arr)),
            })
    return pd.DataFrame(rows)


def quandt_andrews(comm: pd.DataFrame) -> pd.DataFrame:
    """Find the data-driven break point that maximizes |delta| over a sliding
    breakpoint k (with at least 60 days on either side). Compare to event dates."""
    rows = []
    for col in ("C_pairwise", "C_pca"):
        x = comm[col].dropna().reset_index(drop=True)
        d = comm.dropna(subset=[col])["date"].reset_index(drop=True)
        n = len(x)
        if n < 200:
            continue
        best_t, best_idx, best_delta = -np.inf, -1, 0.0
        for k in range(60, n - 60):
            pre = x[:k]; post = x[k:]
            mu_pre, mu_post = pre.mean(), post.mean()
            sd_pre, sd_post = pre.std(ddof=1), post.std(ddof=1)
            if sd_pre <= 0 or sd_post <= 0:
                continue
            pooled = np.sqrt(sd_pre ** 2 / len(pre) + sd_post ** 2 / len(post))
            t = (mu_post - mu_pre) / pooled
            if abs(t) > abs(best_t):
                best_t = float(t)
                best_idx = k
                best_delta = float(mu_post - mu_pre)
        rows.append({
            "test": "quandt_andrews",
            "measure": col,
            "best_break_date": str(d.iloc[best_idx].date()),
            "best_t": best_t,
            "best_delta": best_delta,
            "best_idx": int(best_idx),
            "n": int(n),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commonality", default="results/revision/commonality_daily.parquet")
    p.add_argument("--out", default="results/revision/commonality_regime_shift.csv")
    args = p.parse_args()

    comm = pd.read_parquet(args.commonality)
    comm["date"] = pd.to_datetime(comm["date"]).dt.tz_localize(None).dt.normalize()
    log.info("commonality loaded: %d days", len(comm))

    era = era_test(comm)
    emp = empirical_pvalue(comm, n_perm=1000)
    qa = quandt_andrews(comm)

    out = pd.concat([era, emp, qa], ignore_index=True)
    out.to_csv(args.out, index=False)
    log.info("wrote %s (%d rows)", args.out, len(out))

    print("=== ERA TEST ===")
    print(era.to_string())
    print()
    print("=== EMPIRICAL P (BTC events) ===")
    print(emp[emp.event_id.str.startswith("btc_")].to_string())
    print()
    print("=== EMPIRICAL P (all events) ===")
    print(emp[["event_id","measure","obs_delta","perm_p95","p_two_sided"]].to_string())
    print()
    print("=== QUANDT-ANDREWS ===")
    print(qa.to_string())


if __name__ == "__main__":
    main()
