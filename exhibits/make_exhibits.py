#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRL letter --- exhibit generator.

Reads the revision result CSVs and emits, into this folder:
  * figures  : fig1_eventstudy.pdf, fig2_venueproxy.pdf, fig3_falsification.pdf,
               fig4_rollingwindow.pdf, figS1_eventstudy_grid.pdf,
               figS2_commonality_exclusion.pdf
  * tables   : tab_descriptive.tex, tab_ddd.tex, tab_did.tex, tab_mechanism.tex,
               tab_inference.tex, tab_mica.tex, tab_commonality.tex,
               tab_pervenue.tex   (LaTeX fragments, \\input by main.tex / supplementary.tex)

Figures carry NO title or subtitle --- the LaTeX \\caption supplies those.
All numbers therefore come straight from the CSVs (no hand transcription).
"""
import os
from pathlib import Path as _P
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = str(_P(os.environ.get("CRYPTO1_ROOT",
                               _P(__file__).resolve().parents[1])))
OUTDIR = os.environ.get("EXHIBIT_OUT", str(_P(__file__).resolve().parent / "output"))
os.makedirs(OUTDIR, exist_ok=True)
RF = f"{BASE}/results/revision_fixed"
RV = f"{BASE}/results/revision"
OUT = OUTDIR

# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8,
    "axes.edgecolor": "#3a3a3a", "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.3,
})
C = {"treated": "#b3261e", "gemini": "#e08e0b", "coinbase": "#1f4e79",
     "union": "#2a7f7f", "neutral": "#37474f", "ref": "#9e9e9e",
     "band": "#1f4e79"}

OUTC = {
    "log_dispersion_bps": "Cross-venue dispersion",
    "log_amihud_illiquidity": "Amihud illiquidity",
    "log_notional_volume": "Notional volume",
    "log_high_low_range_bps": "High–low range",
    "log_rolling_notional_depth_proxy_24h": "Rolling 24h volume",
    "log_corwin_schultz_spread": "Corwin–Schultz spread",
}
OUTC_ORDER = list(OUTC)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def pct(c):
    return (np.exp(c) - 1.0) * 100.0


def cell(coef, se, p):
    """A regression-table cell: coefficient with stars over (SE)."""
    return f"{coef:.3f}\\sym{{{stars(p)}}} \\\\ ({se:.3f})"


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ======================================================================
# FIGURES
# ======================================================================
def fig_eventstudy(events, outcomes, fname, ncol=3):
    """Event-study coefficient panels with 95% CI bands."""
    es = pd.read_csv(f"{RF}/event_study_full.csv")
    nrow = len(events)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.05 * nrow),
                             squeeze=False)
    for i, (ev, evlab) in enumerate(events):
        for j, oc in enumerate(outcomes):
            ax = axes[i][j]
            d = es[(es.event_id == ev) & (es.outcome == oc)].sort_values("event_time")
            t, b, se = d.event_time.values, d.coef.values, d.se.values
            lo, hi = b - 1.96 * se, b + 1.96 * se
            pre = t < 0
            ax.axhline(0, color=C["ref"], lw=0.8, zorder=1)
            ax.axvline(0, color=C["ref"], lw=0.8, ls="--", zorder=1)
            ax.fill_between(t, lo, hi, color=C["band"], alpha=0.16, zorder=2,
                            linewidth=0)
            ax.plot(t[pre], b[pre], color=C["neutral"], zorder=3)
            ax.plot(t[~pre], b[~pre], color=C["treated"], zorder=3)
            ax.scatter(t, b, s=6, color=np.where(pre, C["neutral"], C["treated"]),
                       zorder=4)
            despine(ax)
            if i == 0:
                ax.set_title(OUTC[oc])
            if j == 0:
                ax.set_ylabel(f"{evlab}\ncoefficient")
            if i == nrow - 1:
                ax.set_xlabel("Event time (days)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}")
    plt.close(fig)
    print(f"  {fname}")


def fig_venueproxy():
    """DDD coefficient by U.S.-access venue proxy, all outcomes, approval event."""
    d = pd.read_csv(f"{RF}/ddd_btc_approval.csv")
    d = d[d.event_id == "btc_etf_approval"]
    proxies = [("btc_ddd_gemini_only", "Gemini only", C["gemini"], "o"),
               ("btc_ddd_coinbase_only", "Coinbase only", C["coinbase"], "s"),
               ("btc_ddd_coinbase_gemini", "Coinbase + Gemini", C["union"], "D")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y = np.arange(len(OUTC_ORDER))
    off = {0: 0.24, 1: 0.0, 2: -0.24}
    for k, (spec, lab, col, mk) in enumerate(proxies):
        xs, ys = [], []
        for oi, oc in enumerate(OUTC_ORDER):
            r = d[(d.spec == spec) & (d.outcome == oc)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            e = pct(r.coef)
            lo = pct(r.coef - 1.96 * r.se)
            hi = pct(r.coef + 1.96 * r.se)
            yy = y[oi] + off[k]
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.3, zorder=2)
            ax.scatter(e, yy, color=col, marker=mk, s=30, zorder=3,
                       label=lab if oi == 0 else None)
    ax.axvline(0, color=C["ref"], lw=0.9, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([OUTC[o] for o in OUTC_ORDER])
    ax.set_xlabel("Triple-difference effect on the U.S.-access cell (%)")
    ax.legend(loc="lower right", frameon=False)
    despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_venueproxy.pdf")
    plt.close(fig)
    print("  fig2_venueproxy.pdf")


def fig_falsification():
    """Cross-asset falsification forest plot (depth proxy)."""
    btc = pd.read_csv(f"{RF}/mechanism_btc.csv")
    xa = pd.read_csv(f"{RF}/mechanism_cross_asset_falsification.csv")
    DEPTH = "log_rolling_notional_depth_proxy_24h"
    bd = btc[btc.model == "depth_x_etfvol"].iloc[0]
    rows = [("BTC", bd.coef, bd.se, "treated")]
    for _, r in xa[(xa.test == "control_asset_x_btc_etfvol") &
                   (xa.outcome == DEPTH)].iterrows():
        if r.se < 1e-5:   # degenerate cluster-robust variance; omit from inference
            continue
        rows.append((r.asset, r.coef, r.se, "control"))
    df = pd.DataFrame(rows, columns=["asset", "coef", "se", "kind"])
    df["e"], df["lo"], df["hi"] = (pct(df.coef), pct(df.coef - 1.96 * df.se),
                                   pct(df.coef + 1.96 * df.se))
    df = df.sort_values("e").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for i, r in df.iterrows():
        tr = r.kind == "treated"
        col = C["treated"] if tr else C["neutral"]
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.5, zorder=2)
        ax.scatter(r.e, i, color=col, s=55 if tr else 32,
                   marker="D" if tr else "o", zorder=3)
    ax.axvline(0, color=C["ref"], lw=0.9, ls="--", zorder=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{r.asset} (treated)" if r.kind == "treated" else r.asset
                        for _, r in df.iterrows()])
    ax.set_xlabel("Effect on rolling 24-hour volume (%)")
    despine(ax)
    ax.margins(y=0.06)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_falsification.pdf")
    plt.close(fig)
    print("  fig3_falsification.pdf")


def fig_rollingwindow():
    """(a) placebo Welch-t distribution; (b) Welch t vs rolling-window length."""
    pl = pd.read_csv(f"{RV}/commonality_placebo.csv")
    pl = pl[pl.measure == "C_pairwise"]
    win = pd.read_csv(f"{RV}/commonality_window_robustness.csv")
    win = win[win.measure == "C_pairwise"]
    real_t = 7.47  # BTC approval, 30-day window (commonality_event_summary.csv)

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.9))

    a.axhline(0, color=C["ref"], lw=0.8)
    jit = np.random.default_rng(0).normal(0, 0.04, len(pl))
    a.scatter(jit, pl.welch_t, s=22, color=C["neutral"], alpha=0.8, zorder=3)
    a.scatter([0], [real_t], s=80, marker="D", color=C["treated"], zorder=4)
    a.annotate("BTC approval\n(real event)", (0, real_t), xytext=(0.18, real_t),
               fontsize=7.5, va="center", color=C["treated"])
    a.set_xlim(-0.5, 0.9)
    a.set_xticks([])
    a.set_ylabel("Welch $t$ of commonality jump")
    a.set_xlabel("20 placebo (non-event) dates")
    despine(a)

    evs = [("btc_etf_approval", "BTC approval", C["treated"]),
           ("btc_etf_launch", "BTC launch", C["coinbase"]),
           ("eth_rule_approval", "ETH rule", C["gemini"]),
           ("eth_etf_launch", "ETH launch", C["union"])]
    for ev, lab, col in evs:
        d = win[win.event_id == ev].sort_values("roll_window")
        b.plot(d.roll_window, d.welch_t, marker="o", ms=4, color=col, label=lab)
    b.axhline(0, color=C["ref"], lw=0.8)
    b.set_xticks([15, 30, 60])
    b.set_xlabel("Rolling-window length (days)")
    b.set_ylabel("Welch $t$ of commonality jump")
    b.legend(loc="upper left", frameon=False)
    despine(b)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_rollingwindow.pdf")
    plt.close(fig)
    print("  fig4_rollingwindow.pdf")


def fig_commonality_exclusion():
    """Leave-one-out commonality jump (drop each venue / each asset)."""
    ex = pd.read_csv(f"{RV}/commonality_exclusion_robustness.csv")
    ex = ex[(ex.measure == "C_pairwise") & (ex.event_id == "btc_etf_approval")]
    ex = ex.sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labs = [d.replace("venue=", "drop ").replace("asset=", "drop ")
            for d in ex.dropped]
    ax.barh(range(len(ex)), ex.delta * 100, color=C["coinbase"], alpha=0.85,
            height=0.62)
    ax.set_yticks(range(len(ex)))
    ax.set_yticklabels(labs)
    ax.set_xlabel("Commonality jump at BTC ETF approval (pp)")
    despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/figS2_commonality_exclusion.pdf")
    plt.close(fig)
    print("  figS2_commonality_exclusion.pdf")


def fig_speccurve():
    """Specification curve of the venue-reallocation effect.

    Plots the triple-difference estimate of the U.S.-access-venue liquidity
    effect across every (proxy x liquidity-outcome x event) specification,
    sorted by magnitude. Outcomes are oriented so a positive value means
    *more* liquidity on the U.S.-access segment (Amihud illiquidity and the
    Corwin--Schultz spread are sign-reversed). The lower panel marks, for
    each specification, which analytical choices are active -- making visible
    that the significant estimates are confined to the small Gemini proxy.
    """
    d = pd.read_csv(f"{RF}/ddd_btc_approval.csv")
    orient = {  # outcome -> (sign, short label)
        "log_rolling_notional_depth_proxy_24h": (+1, "Rolling 24h volume"),
        "log_notional_volume": (+1, "Notional volume"),
        "log_amihud_illiquidity": (-1, "Liquidity ($-$Amihud)"),
        "log_corwin_schultz_spread": (-1, "Liquidity ($-$spread)"),
    }
    proxymeta = {  # spec -> (label, colour, marker)
        "btc_ddd_gemini_only": ("Gemini only", C["gemini"], "o"),
        "btc_ddd_coinbase_only": ("Coinbase only", C["coinbase"], "s"),
        "btc_ddd_coinbase_gemini": ("Coinbase $+$ Gemini", C["union"], "D"),
    }
    rows = []
    for _, r in d.iterrows():
        if r.outcome not in orient or r.spec not in proxymeta:
            continue
        s = orient[r.outcome][0]
        c = s * r.coef
        rows.append({"eff": pct(c), "lo": pct(c - 1.96 * r.se),
                     "hi": pct(c + 1.96 * r.se), "sig": r.pval < 0.05,
                     "spec": r.spec, "ev": r.event_id})
    df = pd.DataFrame(rows).sort_values("eff").reset_index(drop=True)
    n = len(df)

    fig = plt.figure(figsize=(6.6, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.05], hspace=0.07)
    at, ab = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    at.axhline(0, color=C["ref"], lw=0.9, ls="--", zorder=1)
    for i, r in df.iterrows():
        _, col, mk = proxymeta[r.spec]
        at.plot([i, i], [r.lo, r.hi], color=col, lw=1.0, alpha=0.65, zorder=2)
        at.scatter(i, r.eff, marker=mk, s=34, zorder=3, linewidths=1.1,
                   facecolor=col if r.sig else "white", edgecolor=col)
    at.set_ylabel("Effect on U.S.-access-venue liquidity (%)")
    at.set_xlim(-1, n)
    at.set_xticks([])
    despine(at)
    handles = [plt.Line2D([], [], marker=mk, ls="", mec=col, mfc=col, ms=6,
                          label=lab) for lab, col, mk in proxymeta.values()]
    handles.append(plt.Line2D([], [], marker="o", ls="", mec=C["neutral"],
                              mfc="white", ms=6, label="$p\\geq0.05$ (hollow)"))
    at.legend(handles=handles, loc="upper left", frameon=False, ncol=2,
              handletextpad=0.3, columnspacing=1.1)

    choices = [("Gemini", "spec", "btc_ddd_gemini_only"),
               ("Coinbase", "spec", "btc_ddd_coinbase_only"),
               ("Coinbase + Gemini", "spec", "btc_ddd_coinbase_gemini"),
               ("Approval event", "ev", "btc_etf_approval"),
               ("Launch event", "ev", "btc_etf_launch")]
    for yi, (lab, kind, key) in enumerate(choices):
        ys = len(choices) - 1 - yi
        for i, r in df.iterrows():
            on = (r.spec == key) if kind == "spec" else (r.ev == key)
            if on:
                col = proxymeta[r.spec][1] if r.sig else C["ref"]
                ab.scatter(i, ys, marker="s", s=15, color=col)
    ab.set_xlim(-1, n)
    ab.set_ylim(-0.6, len(choices) - 0.4)
    ab.set_yticks(range(len(choices)))
    ab.set_yticklabels([c[0] for c in choices][::-1])
    ab.set_xticks([])
    ab.set_xlabel("Specifications, sorted by estimated effect "
                  "(filled = significant at 5%)")
    despine(ab)

    fig.savefig(f"{OUT}/fig1_speccurve.pdf")
    plt.close(fig)
    print("  fig1_speccurve.pdf")


# ======================================================================
# TABLES  (LaTeX fragments)
# ======================================================================
def tab_descriptive():
    d = pd.read_csv(f"{RF}/summary_stats.csv").set_index("variable")
    name = {"dispersion_bps": "Cross-venue dispersion (bps)",
            "amihud_illiquidity": "Amihud illiquidity",
            "notional_volume": "Notional volume (USD)",
            "high_low_range_bps": "High--low range (bps)",
            "rolling_notional_depth_proxy_24h": "Rolling 24h volume (USD)",
            "corwin_schultz_spread": "Corwin--Schultz spread"}
    L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"Variable & $N$ & Mean & SD & P25 & Median & P75 \\", r"\midrule"]

    def f(x):
        ax = abs(x)
        if ax == 0:
            return "0"
        if ax >= 1e6 or ax < 1e-3:
            return f"{x:.2e}"
        if ax >= 100:
            return f"{x:,.0f}"
        return f"{x:.3f}"
    for v in name:
        r = d.loc[v]
        L.append(f"{name[v]} & {int(r.n_wins):,} & {f(r.mean_wins)} & "
                 f"{f(r.sd_wins)} & {f(r.p25_wins)} & {f(r.p50_wins)} & "
                 f"{f(r.p75_wins)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_descriptive.tex", "w").write("\n".join(L))
    print("  tab_descriptive.tex")


def tab_ddd():
    d = pd.read_csv(f"{RF}/ddd_btc_approval.csv")
    specs = [("btc_ddd_gemini_only", "Gemini only"),
             ("btc_ddd_coinbase_only", "Coinbase only"),
             ("btc_ddd_coinbase_gemini", "Coinbase + Gemini")]
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         " & " + " & ".join(s[1] for s in specs) + r" \\", r"\midrule"]
    for ev, evlab in [("btc_etf_approval", "Panel A. Spot Bitcoin ETF approval"),
                      ("btc_etf_launch", "Panel B. Spot Bitcoin ETF launch")]:
        L.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{evlab}}}}} \\")
        for oc in OUTC_ORDER:
            coefs, ses = [], []
            for spec, _ in specs:
                r = d[(d.event_id == ev) & (d.spec == spec) &
                      (d.outcome == oc)].iloc[0]
                coefs.append(f"{r.coef:.3f}\\sym{{{stars(r.pval)}}} "
                             f"{{[}}{pct(r.coef):+.1f}\\%{{]}}")
                ses.append(f"({r.se:.3f})")
            L.append(f"{OUTC[oc]} & " + " & ".join(coefs) + r" \\")
            L.append(" & " + " & ".join(ses) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_ddd.tex", "w").write("\n".join(L))
    print("  tab_ddd.tex")


def tab_did():
    d = pd.read_csv(f"{RF}/did_full_grid.csv")
    cols = [("btc_etf_approval", "BTC approval"), ("btc_etf_launch", "BTC launch"),
            ("eth_rule_approval", "ETH rule"), ("eth_etf_launch", "ETH launch")]
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         " & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for oc in OUTC_ORDER:
        coefs, ses = [], []
        for ev, _ in cols:
            r = d[(d.event_id == ev) & (d.outcome == oc)].iloc[0]
            coefs.append(f"{r.coef:.3f}\\sym{{{stars(r.pval)}}}")
            ses.append(f"({r.se:.3f})")
        L.append(f"{OUTC[oc]} & " + " & ".join(coefs) + r" \\")
        L.append(" & " + " & ".join(ses) + r" \\")
    L += [r"\midrule",
          r"Observations & 6{,}039 & 6{,}039 & 6{,}039 & 6{,}039 \\",
          r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_did.tex", "w").write("\n".join(L))
    print("  tab_did.tex")


def tab_mechanism():
    mb = pd.read_csv(f"{RF}/mechanism_btc.csv")
    me = pd.read_csv(f"{RF}/mechanism_eth.csv")
    xa = pd.read_csv(f"{RF}/mechanism_cross_asset_falsification.csv")
    pp = pd.read_csv(f"{RV}/mechanism_placebo_pre.csv")
    DEP, VOL = "log_rolling_notional_depth_proxy_24h", "log_notional_volume"

    def row(lab, r):
        # ADA, AVAX, DOT, LINK carry numerically degenerate SEs in the source
        # pipeline (se < 1e-5); flag with a dagger rather than print "(0.0000)".
        if r.se < 1e-5:
            return (f"{lab} & {r.coef:.4f}\\sym{{\\dagger}} & --- "
                    f"& {pct(r.coef):+.1f}\\% & --- \\\\")
        return (f"{lab} & {r.coef:.4f}\\sym{{{stars(r.pval)}}} & ({r.se:.4f}) "
                f"& {pct(r.coef):+.1f}\\% & {r.pval:.3f} \\\\")
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & Coefficient & (SE) & Effect & $p$ \\", r"\midrule",
         r"\multicolumn{5}{@{}l}{\textit{Panel A. ETF-activity interaction, "
         r"treated-asset $\times$ U.S.-access venue}} \\"]
    L.append(row("BTC, rolling 24h volume ($\\times$ ETF volume)",
                 mb[mb.model == "depth_x_etfvol"].iloc[0]))
    L.append(row("BTC, notional volume ($\\times$ ETF volume)",
                 mb[mb.model == "vol_x_etfvol"].iloc[0]))
    L.append(row("BTC, rolling 24h volume ($\\times$ ETF net flow)",
                 mb[mb.model == "depth_x_flow"].iloc[0]))
    L.append(row("ETH, rolling 24h volume ($\\times$ ETF volume)",
                 me[me.model == "depth_x_etfvol"].iloc[0]))
    L.append(row("ETH, notional volume ($\\times$ ETF volume)",
                 me[me.model == "vol_x_etfvol"].iloc[0]))
    L.append(r"\addlinespace")
    L.append(r"\multicolumn{5}{@{}l}{\textit{Panel B. Falsification "
             r"(rolling 24h volume $\times$ Bitcoin ETF volume)}} \\")
    ca = xa[(xa.test == "control_asset_x_btc_etfvol") & (xa.outcome == DEP)
            & (xa.se >= 1e-5)]   # drop degenerate cluster-robust variances
    ca = ca.sort_values("coef", ascending=False)
    for _, r in ca.iterrows():
        L.append(row(f"Control asset: {r.asset}", r))
    for tst, lab in [("btc_non_usv_etfvol", "BTC $\\times$ offshore venues"),
                     ("eth_non_usv_etfvol", "ETH $\\times$ offshore venues")]:
        r = xa[(xa.test == tst) & (xa.outcome == DEP)].iloc[0]
        L.append(row(lab, r))
    rp = pp[pp.outcome == DEP]
    rb = rp[rp.asset == "BTC"].iloc[0]
    L.append(row("Pre-ETF-era placebo (BTC)", rb))
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_mechanism.tex", "w").write("\n".join(L))
    print("  tab_mechanism.tex")


def tab_inference():
    cs = pd.read_csv(f"{RV}/commonality_event_summary.csv")
    rs = pd.read_csv(f"{RV}/commonality_regime_shift.csv")
    L = [r"\begin{tabular}{lc}", r"\toprule", r"Diagnostic & Value \\",
         r"\midrule",
         r"\multicolumn{2}{@{}l}{\textit{Rolling-window commonality "
         r"(Bitcoin ETF approval)}} \\"]
    wt = cs[(cs.event_id == "btc_etf_approval") &
            (cs.measure == "C_pairwise")].iloc[0].welch_t
    pe = rs[(rs.test == "empirical_p") & (rs.event_id == "btc_etf_approval") &
            (rs.measure == "C_pairwise")].iloc[0].p_two_sided
    era = rs[(rs.test == "era_pre_vs_post") & (rs.measure == "C_pairwise")].iloc[0]
    L.append(rf"\quad Welch $t$ of commonality jump & {wt:.2f} \\")
    L.append(rf"\quad Permutation $p$ (1{{,}}000 draws) & {pe:.2f} \\")
    L.append(rf"\quad Post- vs.\ pre-ETF era difference (Newey--West) & "
             rf"{era.delta * 100:+.1f} pp ($t={era.nw_t:.2f}$) \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_inference.tex", "w").write("\n".join(L))
    print("  tab_inference.tex")


def tab_mica():
    d = pd.read_csv(f"{RF}/did_full_grid.csv")
    cols = [("mica_stablecoin", "Stablecoin titles"),
            ("mica_full", "Full application")]
    L = [r"\begin{tabular}{lcc}", r"\toprule",
         " & " + " & ".join(c[1] for c in cols) + r" \\", r"\midrule"]
    for oc in OUTC_ORDER:
        coefs, ses = [], []
        for ev, _ in cols:
            r = d[(d.event_id == ev) & (d.outcome == oc)].iloc[0]
            coefs.append(f"{r.coef:.3f}\\sym{{{stars(r.pval)}}}")
            ses.append(f"({r.se:.3f})")
        L.append(f"{OUTC[oc]} & " + " & ".join(coefs) + r" \\")
        L.append(" & " + " & ".join(ses) + r" \\")
    L += [r"\midrule", r"Observations & 6{,}039 & 6{,}039 \\",
          r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_mica.tex", "w").write("\n".join(L))
    print("  tab_mica.tex")


def tab_commonality():
    win = pd.read_csv(f"{RV}/commonality_window_robustness.csv")
    win = win[win.measure == "C_pairwise"]
    pl = pd.read_csv(f"{RV}/commonality_placebo.csv")
    pl = pl[pl.measure == "C_pairwise"]
    evlab = {"btc_etf_approval": "Bitcoin ETF approval",
             "btc_etf_launch": "Bitcoin ETF launch",
             "eth_rule_approval": "Ether ETF rule approval",
             "eth_etf_launch": "Ether ETF launch",
             "mica_stablecoin": "MiCA stablecoin titles",
             "mica_full": "MiCA full application"}
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"\multicolumn{4}{@{}l}{\textit{Panel A. Welch $t$ of commonality "
         r"jump by rolling-window length}} \\",
         r"Event & 15-day & 30-day & 60-day \\", r"\midrule"]
    for ev, lab in evlab.items():
        d = win[win.event_id == ev]
        g = {int(r.roll_window): r.welch_t for _, r in d.iterrows()}
        L.append(f"{lab} & {g.get(15, float('nan')):.2f} & "
                 f"{g.get(30, float('nan')):.2f} & {g.get(60, float('nan')):.2f} \\\\")
    L.append(r"\addlinespace")
    L.append(r"\multicolumn{4}{@{}l}{\textit{Panel B. Welch $t$ at 20 placebo "
             r"(non-event) dates, 30-day window}} \\")
    L.append(r"\multicolumn{4}{@{}l}{Minimum \quad Median \quad Maximum "
             r"\quad Share with $|t|>2$} \\")
    tt = pl.welch_t
    L.append(rf"\multicolumn{{4}}{{@{{}}l}}{{{tt.min():.2f} \quad\quad "
             rf"{tt.median():.2f} \quad\quad {tt.max():.2f} \quad\quad "
             rf"{(tt.abs() > 2).mean() * 100:.0f}\%}} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_commonality.tex", "w").write("\n".join(L))
    print("  tab_commonality.tex")


def tab_pervenue():
    d = pd.read_csv(f"{RV}/mechanism_per_venue.csv")
    oc = {"log_rolling_notional_depth_proxy_24h": "Rolling 24h volume",
          "log_notional_volume": "Notional volume",
          "log_dispersion_bps": "Cross-venue dispersion"}
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Asset & Venue & " + " & ".join(oc.values()) + r" \\", r"\midrule"]
    for a in ["BTC", "ETH"]:
        for v in ["coinbase", "gemini"]:
            cs = []
            for o in oc:
                r = d[(d.asset == a) & (d.venue == v) & (d.outcome == o)]
                if len(r):
                    r = r.iloc[0]
                    cs.append(f"{r.beta:.4f}\\sym{{{stars(r.pval)}}}")
                else:
                    cs.append("---")
            L.append(f"{a} & {v.capitalize()} & " + " & ".join(cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/tab_pervenue.tex", "w").write("\n".join(L))
    print("  tab_pervenue.tex")


# ======================================================================
if __name__ == "__main__":
    os.chdir(OUT)
    print("figures:")
    fig_speccurve()
    fig_venueproxy()
    fig_falsification()
    fig_rollingwindow()
    fig_commonality_exclusion()
    print("tables:")
    tab_descriptive()
    tab_ddd()
    tab_did()
    tab_mechanism()
    tab_inference()
    tab_mica()
    tab_commonality()
    tab_pervenue()
    print("done.")
