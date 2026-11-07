# Data files

## `tsla_tsm_prices_2020_2025.csv` — not redistributable

TSLA/TSM daily closes exported from Bloomberg. Bloomberg terms do not permit
redistributing the export, so the file is **not in this repository** — it is
gitignored. Every result in `../results/` and every locked number in the tests
was produced from it; the SHA256 below fixes exactly which file that was.

```
SHA256  4101faa1544069d19e3e9e0b75df8365f328657aea7769a9b4c239750ae43ec4
Shape   date,TSLA,TSM  —  2020-01-02 → 2025-03-07 daily closes, one row per date
```

To reproduce the full run: place your own Bloomberg export of TSLA and TSM
daily closing prices (2020-01-02 through 2025-03-07 inclusive) at this path
and check its SHA256 against the value above (also recorded as `data_sha256`
in `../results/summary.json`). Then `python run_analysis.py`.

Without the panel, the test suite still runs in full: everything except the
two calibration-from-prices checks uses the frozen contract inputs recorded
in `../results/summary.json`, and those two tests skip with an explicit
message. No test substitutes synthetic prices.

## `assumption_inputs.json` — public sources

TSMC dividend history (investor.tsmc.com) and the U.S. Treasury daily par
yields, as of the 2026-10-28 retrieval date recorded inside the file. Public
data, kept in the repository.
