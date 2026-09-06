#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4 (fig3_falsification.pdf) --- TWO-PANEL cross-asset falsification.

Reviewer R2 (minor) asked that the figure not display, on its own, the
full-sample "Solana above Bitcoin" ordering that the text disowns. We keep the
full-sample panel (a) and add the restricted-window panel (b):

  (a) Full sample (as published): controls bracket Bitcoin and Solana exceeds
      it --- the ordering the text attributes to the controls' own late-2025
      ETFs and to beta.
  (b) 2024 windows only (before Solana/Litecoin/Dogecoin obtained U.S. ETFs):
      Bitcoin loads highest, reversing the full-sample ordering.

Panel (a) numbers come straight from the published result CSVs; panel (b) is
recomputed here with the SAME single-asset entity-FE / cluster-by-entity
estimator used in results/revision_newcalcs (N5) --- validated to reproduce the
published clustered SEs to 4 decimals (see recompute_falsif.py).

Style mirrors scripts/.../make_exhibits.py so the figure matches the others.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _P
BASE = str(_P(os.environ.get("CRYPTO1_ROOT", _P(__file__).resolve().parents[1])))
OUTDIR = os.environ.get("EXHIBIT_OUT", str(_P(__file__).resolve().parent / "output"))
os.makedirs(OUTDIR, exist_ok=True)
RF = f"{BASE}/results/revision_fixed"
OUT = (f"{OUTDIR}/fig3_falsification.pdf")

TURN = "log_rolling_notional_depth_proxy_24h"
CONTROLS = ["SOL", "DOGE", "LTC", "BCH", "UNI"]
Z = 1.959963985

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.edgecolor": "#3a3a3a", "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.3,
})
C = {"treated": "#b3261e", "neutral": "#37474f", "ref": "#9e9e9e"}


def pct(c):
    return (np.exp(c) - 1.0) * 100.0


# ----------------------------------------------------------------- estimator
def load_dd():
    d = pd.read_parquet(f"{RF}/daily_revision.parquet")
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    cons = pd.read_csv(f"{RF}/etf_flow_daily_consolidated.csv", parse_dates=["date"])
    cons["date"] = cons["date"].dt.normalize()
    btcv = (cons[cons.asset == "BTC"][["date", "etf_total_dollar_volume"]]
            .rename(columns={"etf_total_dollar_volume": "btcv"}))
    dd = d.merge(btcv, on="date", how="left")
    dd["log_btcv"] = np.log1p(dd["btcv"].fillna(0.0))
    dd["is_usv"] = dd["venue"].isin({"coinbase", "gemini"}).astype(int)
    return dd


def falsif_2024(dd, asset):
    """Single-asset entity-FE regression of rolling 24h volume on is_usv*log_btcv,
    restricted to 2024, cluster-robust (raw sandwich) by entity == venue.
    Reproduces results/revision_newcalcs/N5 raw_2024only (coef) and the
    published clustered SE convention."""
    sub = dd[(dd.asset == asset) & (dd.date >= "2024-01-01") & (dd.date <= "2024-12-31")].copy()
    sub["term"] = sub["is_usv"] * sub["log_btcv"]
    s = sub.dropna(subset=[TURN, "term"])
    ent = (s["asset"] + "_" + s["venue"]).to_numpy()
    y = s[TURN].to_numpy(float); x = s["term"].to_numpy(float)
    g = pd.DataFrame({"ent": ent, "y": y, "x": x})
    g["y"] -= g.groupby("ent")["y"].transform("mean")
    g["x"] -= g.groupby("ent")["x"].transform("mean")
    xt = g["x"].to_numpy(); yt = g["y"].to_numpy()
    Sxx = float((xt * xt).sum())
    beta = float((xt * yt).sum() / Sxx)
    e = yt - beta * xt
    meat = float((pd.DataFrame({"ent": g["ent"], "xe": xt * e})
                  .groupby("ent")["xe"].sum() ** 2).sum())
    se = float(np.sqrt(meat / Sxx ** 2))
    return beta, se


def panel_a_rows():
    """Full sample (as published): BTC from mechanism_btc; controls from the
    cross-asset falsification file (dropping degenerate se<1e-5 cells)."""
    btc = pd.read_csv(f"{RF}/mechanism_btc.csv")
    bd = btc[btc.model == "depth_x_etfvol"].iloc[0]
    rows = [("BTC", float(bd.coef), float(bd.se), True)]
    xa = pd.read_csv(f"{RF}/mechanism_cross_asset_falsification.csv")
    xa = xa[(xa.test == "control_asset_x_btc_etfvol") & (xa.outcome == TURN)]
    xa = xa[xa.asset.isin(CONTROLS) & (xa.se >= 1e-5)]
    for _, r in xa.iterrows():
        rows.append((r.asset, float(r.coef), float(r.se), False))
    return rows


def panel_b_rows(dd):
    rows = []
    for a in ["BTC"] + CONTROLS:
        b, se = falsif_2024(dd, a)
        rows.append((a, b, se, a == "BTC"))
    return rows


def draw(ax, rows, title):
    df = pd.DataFrame(rows, columns=["asset", "coef", "se", "treated"])
    df["e"] = pct(df.coef)
    df["lo"] = pct(df.coef - Z * df.se)
    df["hi"] = pct(df.coef + Z * df.se)
    df = df.sort_values("e").reset_index(drop=True)
    for i, r in df.iterrows():
        col = C["treated"] if r.treated else C["neutral"]
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.5, zorder=2)
        ax.scatter(r.e, i, color=col, s=58 if r.treated else 32,
                   marker="D" if r.treated else "o", zorder=3)
    ax.axvline(0, color=C["ref"], lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.asset} (treated)" if r.treated else r.asset
                        for _, r in df.iterrows()])
    ax.set_xlabel("Effect on rolling 24-hour volume (%)")
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.08)
    return df


if __name__ == "__main__":
    dd = load_dd()
    ra = panel_a_rows()
    rb = panel_b_rows(dd)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 3.3), sharex=True)
    dfa = draw(axL, ra, "(a) Full sample (as published)")
    dfb = draw(axR, rb, "(b) 2024 windows only (pre controls' ETFs)")

    lo = min(dfa.lo.min(), dfb.lo.min()) - 0.6
    hi = max(dfa.hi.max(), dfb.hi.max()) + 0.6
    axL.set_xlim(lo, hi)

    # shared legend
    h = [plt.Line2D([], [], marker="D", ls="", mec=C["treated"], mfc=C["treated"],
                    ms=7, label="Bitcoin (treated)"),
         plt.Line2D([], [], marker="o", ls="", mec=C["neutral"], mfc=C["neutral"],
                    ms=6, label="Control asset (no ETF in 2024)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)

    print("panel (a) full sample:")
    for _, r in dfa.sort_values("e", ascending=False).iterrows():
        print(f"   {r.asset:5s} {r.e:+6.2f}%  [{r.lo:+.2f}, {r.hi:+.2f}]")
    print("panel (b) 2024 only:")
    for _, r in dfb.sort_values("e", ascending=False).iterrows():
        print(f"   {r.asset:5s} {r.e:+6.2f}%  [{r.lo:+.2f}, {r.hi:+.2f}]")
    print("\nwrote", OUT)
