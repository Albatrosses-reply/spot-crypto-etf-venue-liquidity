# Did Spot Crypto ETFs Reallocate Venue Liquidity? Fragile Evidence

Replication package for the paper of the same name, forthcoming in
*Finance Research Letters*.

Hojun Kang (Dongduk Women's University) · Kyuyeon Hwang (Sejong University) ·
Yong-Ki Lee (Chungbuk National University) · Sang-Gun Lee (Sogang University,
corresponding author)

---

## What the paper does

The 2024 launches of the US spot Bitcoin and Ether ETFs are widely read as
institutional-access shocks. A common corollary is that they pulled spot
liquidity toward the venues US institutions can use. We test that
venue-reallocation hypothesis on an hourly panel of twelve cryptocurrencies
across nine exchanges, and find the evidence too fragile to support or refute
it.

- **The estimate is proxy-sensitive.** A triple-difference is large and
  significant with Gemini as the US-access proxy (+37.0%, *p* = 0.005) and
  insignificant with Coinbase, the custodian for most approved spot Bitcoin
  ETFs (+12.7%, *p* = 0.31). The Coinbase specification is underpowered — its
  minimum detectable effect at 80% power is about +39%, essentially the Gemini
  estimate itself. Under randomization inference neither is distinguishable
  from a random-asset placebo.
- **The channel is neither venue- nor asset-specific.** A pooled regression
  puts the offshore slope at +4.3% against +5.1% on the US-access segment — a
  0.8 pp difference, *p* = 0.49. Coins with no ETF load on ETF-wrapper activity
  too.
- **The commonality version is an inference artifact.** A 30-day rolling
  commonality series has first-order autocorrelation of about 0.97 by
  construction. The Bitcoin-approval jump gives a Welch *t* of 7.5, but an
  event-date permutation test gives *p* = 0.41 and autocorrelation-robust
  resampling gives *p* = 0.26 and 0.25. A small era-level rise (6.4 pp,
  *t* = 2.31) does survive.

## Scope of this repository

This package contains **only** what is needed to reproduce the paper: the
acquisition and estimation code, and the estimation output the exhibits are
built from. Exploratory analyses that did not enter the paper, and work
belonging to other projects in the same research directory, are deliberately
excluded.

There is no R code here. The project's R scripts implement a synthetic-control
line of analysis (`synthdid`, `augsynth`) that is not part of this paper and
produces none of its results.

## Layout

```
scripts/     15 files — acquisition, panel construction, estimation
exhibits/     2 files — renders the paper's tables and figures from results/
results/     22 CSVs  — the estimation output every exhibit is built from
```

45 files in total. Every script here is on the path that produces a number in
the paper; nothing else is included.

## Quick start

```bash
git clone https://github.com/<owner>/spot-crypto-etf-venue-liquidity.git
cd spot-crypto-etf-venue-liquidity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export CRYPTO1_ROOT="$(pwd)"

python exhibits/make_exhibits.py            # -> exhibits/output/
python exhibits/make_fig4_falsification.py  # two-panel Figure 4
```

`results/` ships populated, so the exhibits rebuild with no network access.
This reproduces the paper's tables exactly: the regenerated Tables 2, 3 and 6
match the typeset manuscript on all 149 numbers, row labels included.

To rerun the estimation from raw data you first need the market data, which is
**not** in this repository — see [DATA.md](DATA.md).

## Pipeline

| Stage | Scripts |
|---|---|
| Acquire | `fetch_ccxt_ohlcv.py`, `import_kraken_csv.py`, `fetch_etf_flows.py`, `fetch_etf_prices.py` |
| Build | `build_revision_panel.py`, `fix_volume_units.py` |
| Estimate | `run_revision_did.py` (DiD, triple-difference, event study), `run_mechanism_tests.py`, `mechanism_falsification.py`, `mechanism_robustness.py`, `eth_launch_safety.py` |
| Commonality | `compute_commonality.py`, `commonality_robustness.py`, `commonality_regime_shift.py` |
| Review-round additions | `reviewer_newcalcs.py` (produces `results/revision_newcalcs/N1`–`N7`: minimum detectable effect, two-way clustering, randomization inference, Ether triple-difference, US-vs-offshore pooled test, falsification robustness, commonality permutations, venue event study) |

## Data availability

No raw market data is redistributed here. The panel is built from nine
exchange APIs, Farside Investors and Yahoo Finance, whose terms do not permit
wholesale redistribution. [DATA.md](DATA.md) documents every source and the
exact commands to reacquire it, so all results remain reproducible.

## Licence

| | |
|---|---|
| `scripts/`, `exhibits/` | [MIT](LICENSE) |
| `results/` | [CC BY 4.0](LICENSE-DATA) |

Neither licence extends to third-party market data, which this repository does
not contain.

## Citation

```bibtex
@article{kang2026venue,
  author  = {Kang, Hojun and Hwang, Kyuyeon and Lee, Yong-Ki and Lee, Sang-Gun},
  title   = {Did Spot Crypto {ETFs} Reallocate Venue Liquidity? {Fragile} Evidence},
  journal = {Finance Research Letters},
  year    = {2026},
  note    = {Forthcoming}
}
```

Please update the entry with the volume, pages and DOI once assigned.
