"""Reviewer-response new calculations (N1-N7) for FRL-D-26-02641.

Runs on the SAME 9-venue daily panel that produced the published tables
(results/revision_fixed/daily_revision.parquet) so numbers are comparable.

N1 Coinbase minimum detectable effect (power)                 -> R1.1
N2 Two-way clustering + randomization inference on the DDD    -> R2.6
N3 Ether venue-proxy DDD (Gemini/Coinbase/union)              -> R2.9
N4 US-vs-offshore differential in ONE pooled regression       -> R2.11
N5 Residualized falsification regressor + 2024-window robustness -> R1.4/R2.10/R1.5
N6 Commonality block-permutation / circular-shift             -> R2.12
N7 Event-study of the BTC US-access vs offshore cell (pre-trend) -> R1.2/R2.5

Outputs: results/revision_newcalcs/*.csv  and  fig_eventstudy_venue.png
"""
import os
from __future__ import annotations
import warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

warnings.simplefilter("ignore")
logging.getLogger("linearmodels").setLevel(logging.ERROR)
np.random.seed(20260704)

ROOT = Path(os.environ.get("CRYPTO1_ROOT", Path(__file__).resolve().parents[1]))
DAILY = ROOT / "results/revision_fixed/daily_revision.parquet"
ETF   = ROOT / "results/revision_fixed/etf_flow_daily_consolidated.csv"
COMMON= ROOT / "results/revision/commonality_daily.parquet"
OUT   = ROOT / "results/revision_newcalcs"; OUT.mkdir(parents=True, exist_ok=True)

USV = {"gemini_only": {"gemini"}, "coinbase_only": {"coinbase"},
       "coinbase_gemini": {"coinbase", "gemini"}}
TURN = "log_rolling_notional_depth_proxy_24h"   # the "24h turnover" outcome
VOL  = "log_notional_volume"
CONTROLS = ["SOL", "DOGE", "LTC", "BCH", "UNI"]
Z90, Z80 = 1.959963985, 0.8416212336            # z_{.975}, z_{.80}

def pct(c): return (np.exp(c) - 1) * 100

def load_daily():
    d = pd.read_parquet(DAILY)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    return d

def window(d, ev, w=30):
    ev = pd.Timestamp(ev)
    s = d[(d["date"] >= ev - pd.Timedelta(days=w)) & (d["date"] <= ev + pd.Timedelta(days=w))].copy()
    s["event_time"] = (s["date"] - ev).dt.days
    s["post"] = (s["event_time"] >= 0).astype(int)
    return s

def fit_ddd(win, y, usv_set, twoway=False):
    w = win.copy()
    w["usv"] = w["venue"].isin(usv_set).astype(int)
    w["btc"] = (w["asset"] == w["_treated"]).astype(int)
    w["btc_post"] = w["btc"] * w["post"]
    w["usv_post"] = w["usv"] * w["post"]
    w["btc_usv"]  = w["btc"] * w["usv"]
    w["ddd"]      = w["btc"] * w["usv"] * w["post"]
    w["av"] = w["asset"] + "_" + w["venue"]
    w = w.dropna(subset=[y]).set_index(["av", "date"]).sort_index()
    X = w[["ddd", "btc_post", "usv_post", "btc_usv"]]
    m = PanelOLS(w[y], X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False)
    kw = dict(cov_type="clustered", cluster_entity=True)
    if twoway: kw["cluster_time"] = True
    f = m.fit(**kw)
    return float(f.params["ddd"]), float(f.std_errors["ddd"]), float(f.pvalues["ddd"]), int(f.nobs)

# ---------------------------------------------------------------- reproduction
d = load_daily()
appro = window(d, "2024-01-10"); appro["_treated"] = "BTC"
rep = {}
for name, s in USV.items():
    c, se, p, n = fit_ddd(appro, TURN, s)
    rep[name] = (c, se, p, n)
print("=== REPRODUCTION (BTC approval, 24h turnover) ===")
for k, (c, se, p, n) in rep.items():
    print(f"  {k:16s} coef={c:+.4f} se={se:.4f} p={p:.3f}  [{pct(c):+.1f}%]  n={n}")
print("  published: gemini +0.3147(0.1117), coinbase +0.1192(0.1165)")

# ---------------------------------------------------------------- N1 MDE
cb_c, cb_se, cb_p, _ = rep["coinbase_only"]
gem_c = rep["gemini_only"][0]
mde_log = (Z90 + Z80) * cb_se
n1 = pd.DataFrame([{
    "spec": "coinbase_only turnover, BTC approval",
    "coef": cb_c, "pct_effect": pct(cb_c), "se": cb_se, "pval": cb_p,
    "mde_log_80pct": mde_log, "mde_pct_80pct": pct(mde_log),
    "gemini_point_pct": pct(gem_c),
    "note": "test can only detect effects >= MDE at 80% power; MDE ~ the Gemini estimate",
}])
n1.to_csv(OUT / "N1_coinbase_mde.csv", index=False)
print(f"\n=== N1 MDE ===\n  Coinbase SE={cb_se:.4f} -> 80% MDE = {mde_log:.3f} log = {pct(mde_log):+.1f}% "
      f"(Gemini point = {pct(gem_c):+.1f}%)")

# ---------------------------------------------------------------- N3 Ether DDD
print("\n=== N3 Ether venue-proxy DDD ===")
n3rows = []
for ev_id, ev in [("eth_rule_approval", "2024-05-23"), ("eth_etf_launch", "2024-07-23")]:
    win = window(d, ev); win["_treated"] = "ETH"
    for name, s in USV.items():
        for y, lbl in [(TURN, "turnover"), (VOL, "notional")]:
            try:
                c, se, p, n = fit_ddd(win, y, s)
                n3rows.append({"event": ev_id, "proxy": name, "outcome": lbl,
                               "coef": c, "pct": pct(c), "se": se, "pval": p, "nobs": n})
            except Exception as e:
                n3rows.append({"event": ev_id, "proxy": name, "outcome": lbl, "coef": np.nan,
                               "pct": np.nan, "se": np.nan, "pval": np.nan, "nobs": 0})
n3 = pd.DataFrame(n3rows); n3.to_csv(OUT / "N3_ether_ddd.csv", index=False)
for _, r in n3[n3.outcome == "turnover"].iterrows():
    print(f"  {r.event:17s} {r.proxy:16s} turnover {r.pct:+6.1f}%  p={r.pval:.3f}")

# ---------------------------------------------------------------- ETF-volume merge
cons = pd.read_csv(ETF, parse_dates=["date"])
cons["date"] = cons["date"].dt.normalize()
btcv = cons[cons.asset == "BTC"][["date", "etf_total_dollar_volume"]].rename(columns={"etf_total_dollar_volume": "btcv"})
dd = d.merge(btcv, on="date", how="left")
dd["log_btcv"] = np.log1p(dd["btcv"].fillna(0.0))
dd["is_usv"] = dd["venue"].isin({"coinbase", "gemini"}).astype(int)

def fit_entity(df, y, key):
    w = df.dropna(subset=[y, key]).copy()
    w["av"] = w["asset"] + "_" + w["venue"]
    w = w.set_index(["av", "date"]).sort_index()
    f = PanelOLS(w[y], w[[key]], entity_effects=True, time_effects=False,
                 drop_absorbed=True, check_rank=False).fit(cov_type="clustered", cluster_entity=True)
    return f

# ---------------------------------------------------------------- N4 US vs offshore
print("\n=== N4 US-vs-offshore differential (single pooled regression, BTC) ===")
b = dd[dd.asset == "BTC"].copy()
b["etfvol"] = b["log_btcv"]
b["etfvol_us"] = b["log_btcv"] * b["is_usv"]      # differential term
b["av"] = b["asset"] + "_" + b["venue"]
bw = b.dropna(subset=[TURN]).set_index(["av", "date"]).sort_index()
f4 = PanelOLS(bw[TURN], bw[["etfvol", "etfvol_us"]], entity_effects=True, time_effects=False,
              drop_absorbed=True, check_rank=False).fit(cov_type="clustered", cluster_entity=True)
off = float(f4.params["etfvol"]); diff = float(f4.params["etfvol_us"])
n4 = pd.DataFrame([{
    "outcome": "turnover",
    "offshore_slope": off, "offshore_pct": pct(off),
    "us_minus_offshore_diff": diff, "us_minus_offshore_pct": pct(off + diff) - pct(off),
    "diff_se": float(f4.std_errors["etfvol_us"]), "diff_p": float(f4.pvalues["etfvol_us"]),
    "us_slope": off + diff, "us_pct": pct(off + diff),
}])
n4.to_csv(OUT / "N4_us_vs_offshore.csv", index=False)
print(f"  offshore slope {pct(off):+.1f}% | US-access slope {pct(off+diff):+.1f}% | "
      f"difference p={float(f4.pvalues['etfvol_us']):.3f}  (H0: no US premium)")

# ---------------------------------------------------------------- N5 residualize + 2024 window
print("\n=== N5 residualized regressor + 2024-only window ===")
# date-level aggregate controls from the panel itself
agg = dd.groupby("date").agg(agg_notional=("notional_volume", "sum")).reset_index()
agg["log_agg_vol"] = np.log1p(agg["agg_notional"])
btc_px = dd[dd.asset == "BTC"].groupby("date")["close"].median().rename("btc_px").reset_index()
btc_px["btc_ret"] = np.log(btc_px["btc_px"]).diff()
btc_px["btc_rv"] = btc_px["btc_ret"].rolling(7, min_periods=3).std()
ctrl = agg.merge(btc_px[["date", "btc_ret", "btc_rv"]], on="date", how="left")
import statsmodels.api as sm
alld = pd.DataFrame({"date": sorted(dd["date"].unique())})
dl = alld.merge(btcv, on="date", how="left")
dl["log_btcv"] = np.log1p(dl["btcv"].fillna(0.0))
dl = dl.merge(ctrl, on="date", how="left").dropna(subset=["log_agg_vol", "btc_ret", "btc_rv"])
Xr = sm.add_constant(dl[["log_agg_vol", "btc_ret", "btc_rv"]])
dl["resid_btcv"] = sm.OLS(dl["log_btcv"], Xr).fit().resid
dd2 = dd.merge(dl[["date", "resid_btcv"]], on="date", how="left")

def falsif(df, regressor, control, label):
    sub = df[df.asset == control].copy()
    sub["term"] = sub["is_usv"] * sub[regressor]
    try:
        f = fit_entity(sub, TURN, "term")
        return {"variant": label, "asset": control, "coef": float(f.params["term"]),
                "pct": pct(float(f.params["term"])), "pval": float(f.pvalues["term"]), "nobs": int(f.nobs)}
    except Exception as e:
        print(f"    [falsif fail {label}/{control}] {type(e).__name__}: {e}")
        return {"variant": label, "asset": control, "coef": np.nan, "pct": np.nan, "pval": np.nan, "nobs": 0}

n5rows = []
# (a) residualized regressor, full sample
for a in ["BTC"] + CONTROLS:
    n5rows.append(falsif(dd2.dropna(subset=["resid_btcv"]), "resid_btcv", a, "residualized_full"))
# (b) 2024-only window, raw regressor (pre SOL/LTC/DOGE ETFs)
dd_2024 = dd[(dd.date >= "2024-01-01") & (dd.date <= "2024-12-31")].copy()
for a in ["BTC"] + CONTROLS:
    n5rows.append(falsif(dd_2024, "log_btcv", a, "raw_2024only"))
# (c) raw full sample (replicate published, for comparison)
for a in ["BTC"] + CONTROLS:
    n5rows.append(falsif(dd, "log_btcv", a, "raw_full_published"))
n5 = pd.DataFrame(n5rows); n5.to_csv(OUT / "N5_falsification_robustness.csv", index=False)
for v in ["raw_full_published", "raw_2024only", "residualized_full"]:
    sub = n5[n5.variant == v].set_index("asset")["pct"]
    order = sub.sort_values(ascending=False)
    btc_rank = list(order.index).index("BTC") + 1
    print(f"  {v:20s}: " + "  ".join(f"{a} {sub[a]:+.1f}%" for a in ["BTC"] + CONTROLS)
          + f"   | BTC rank {btc_rank}/6")

# ---------------------------------------------------------------- N6 commonality permutation
print("\n=== N6 commonality block-permutation / circular-shift ===")
cm = pd.read_parquet(COMMON)
cm["date"] = pd.to_datetime(cm["date"]).dt.tz_localize(None).dt.normalize()
cm = cm.sort_values("date").reset_index(drop=True)
ccol = "C_pairwise"
s = cm[ccol].to_numpy(); dates = cm["date"].to_numpy()
ev = np.datetime64("2024-01-10")
def jump_at(series, center, w=30):
    idx = np.searchsorted(dates, center)
    pre = series[max(0, idx - w):idx]; post = series[idx:idx + w]
    if len(pre) < 5 or len(post) < 5: return np.nan
    return np.nanmean(post) - np.nanmean(pre)
obs = jump_at(s, ev)
# circular-shift null (preserves autocorrelation structure)
shifts = np.random.randint(1, len(s), size=5000)
null_cs = np.array([jump_at(np.roll(s, k), ev) for k in shifts])
p_cs = float(np.nanmean(np.abs(null_cs) >= abs(obs)))
# moving-block bootstrap of jump distribution at random centers
valid_centers = dates[30:len(dates) - 30]
rc = np.random.choice(len(valid_centers), size=5000)
null_mb = np.array([jump_at(s, valid_centers[i]) for i in rc])
p_mb = float(np.nanmean(np.abs(null_mb) >= abs(obs)))
n6 = pd.DataFrame([{"commonality_col": ccol, "obs_jump": obs,
                    "p_circular_shift": p_cs, "p_moving_block": p_mb,
                    "welch_t_published": 7.47, "perm_p_published": 0.41}])
n6.to_csv(OUT / "N6_commonality_permutation.csv", index=False)
print(f"  obs jump={obs:.4f}  circular-shift p={p_cs:.3f}  moving-block p={p_mb:.3f}  "
      f"(published Welch t=7.47, perm p=0.41)")

# ---------------------------------------------------------------- N7 event-study (venue pre-trend)
print("\n=== N7 venue-reallocation DDD-contrast event study (pre-trend) ===")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
es_rows = []
W7 = 21
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (name, s_) in zip(axes, [("gemini_only", {"gemini"}), ("coinbase_only", {"coinbase"})]):
    w = window(d, "2024-01-10", W7).copy()
    w["usv"] = w["venue"].isin(s_)
    w["isbtc"] = w["asset"] == "BTC"
    piv = (w.dropna(subset=[TURN]).groupby(["event_time", "isbtc", "usv"])[TURN].mean()
           .reset_index().pivot_table(index="event_time", columns=["isbtc", "usv"], values=TURN))
    try:
        d_btc = piv[(True, True)] - piv[(True, False)]      # BTC: USV - offshore
        d_ctrl = piv[(False, True)] - piv[(False, False)]   # controls: USV - offshore
        ddd = (d_btc - d_ctrl).dropna()
        for t, v in ddd.items():
            es_rows.append({"proxy": name, "event_time": int(t), "ddd_contrast": float(v)})
        pre = ddd[ddd.index < 0]; post = ddd[ddd.index >= 0]
        sl = float(np.polyfit(pre.index, pre.values, 1)[0]) if len(pre) > 2 else np.nan
        ax.axhline(0, color="grey", lw=.7); ax.axvline(-0.5, color="red", ls="--", lw=.8)
        ax.plot(ddd.index, ddd.values, "o-", ms=3)
        ax.set_title(f"{name}\npre={pre.mean():+.3f} post={post.mean():+.3f} pre-slope={sl:+.4f}")
        ax.set_xlabel("event time (days)")
    except Exception as e:
        ax.set_title(f"{name}: {type(e).__name__}")
axes[0].set_ylabel("DDD contrast (log turnover)")
fig.suptitle("Venue-reallocation DDD contrast around BTC approval (pre-trend check)")
fig.tight_layout(); fig.savefig(OUT / "fig_eventstudy_venue.png", dpi=130)
es = pd.DataFrame(es_rows); es.to_csv(OUT / "N7_eventstudy_venue.csv", index=False)
for name in ["gemini_only", "coinbase_only"]:
    sub = es[es.proxy == name]
    pre = sub[sub.event_time < 0]["ddd_contrast"]; post = sub[sub.event_time >= 0]["ddd_contrast"]
    print(f"  {name:16s} pre-mean={pre.mean():+.3f}  post-mean={post.mean():+.3f}  jump={post.mean()-pre.mean():+.3f}")

# ---------------------------------------------------------------- N2 two-way + RI (last: slowest)
print("\n=== N2 two-way clustering + randomization inference ===")
n2rows = []
assets = sorted(d["asset"].unique())
for name, s in USV.items():
    c1, se1, p1, n = fit_ddd(appro, TURN, s)                      # one-way (entity)
    c2, se2, p2, _ = fit_ddd(appro, TURN, s, twoway=True)         # two-way (entity x date)
    null = []
    if name in ("gemini_only", "coinbase_only"):
        for _ in range(150):
            appro["_treated"] = np.random.choice(assets)
            try:
                cc, _, _, _ = fit_ddd(appro, TURN, s)
                null.append(cc)
            except Exception:
                pass
        appro["_treated"] = "BTC"
    null = np.array(null)
    ri_p = float(np.mean(np.abs(null) >= abs(c1))) if len(null) else np.nan
    n2rows.append({"proxy": name, "coef": c1, "pct": pct(c1),
                   "se_oneway": se1, "p_oneway": p1,
                   "se_twoway": se2, "p_twoway": p2,
                   "ri_p_assetperm": ri_p, "ri_draws": len(null)})
    print(f"  {name:16s} coef {c1:+.3f}  1way p={p1:.3f}  2way p={p2:.3f}  RI(asset) p={ri_p:.3f}")
pd.DataFrame(n2rows).to_csv(OUT / "N2_twoway_randomization.csv", index=False)

print("\nAll outputs ->", OUT)
