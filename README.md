# Intraday momentum: a replication that failed

Gao, Han, Li and Zhou (2018, *Journal of Financial Economics*) find that the first half-hour return of the day, including the overnight gap, predicts the last half-hour return. I replicated their exact specification on two years of recent TAQ trade data across about 3,000 U.S. stocks. The effect is not there.

## Results

Two years of TAQ data, 6 August 2024 to 5 August 2026: 1,412,368 stock-days across 3,056 symbols.

Signal is the 10:00 price over the previous close, minus 1. Target is the close over the 15:30 price, minus 1.

| | Days | Slope t-stat | Sign hit rate | Long-short, net |
|---|---|---|---|---|
| SPY | 501 | -0.71 | 49.7% | -0.48 bp/day |
| QQQ | 501 | -0.33 | 51.1% | -0.27 bp/day |

Across 2,903 stocks with at least 250 days: the mean per-stock t-stat is -0.08, 48% have a positive slope, and the mean hit rate is 49.6%.

Net of an assumed round-trip spread, by liquidity (average daily dollar volume, five equal buckets):

| Liquidity bucket | Names | Mean t | Gross bp/day | Assumed spread bp | Net bp/day |
|---|---|---|---|---|---|
| Top 20% | 578 | -0.38 | -1.15 | 8.8 | -9.99 |
| Large | 577 | -0.04 | -0.49 | 14.7 | -15.17 |
| Mid | 577 | -0.07 | -0.23 | 25.0 | -25.23 |
| Small | 577 | -0.09 | +0.16 | 44.7 | -44.56 |
| Bottom 20% | 577 | +0.17 | +0.08 | 89.4 | -89.35 |

The strategy loses money net of spreads in every bucket, so the signal was dropped. Spreads are assumed by liquidity bucket, not measured from quotes.

## Code

- `src/01_spy_qqq_yfinance.py`: a first check on SPY and QQQ with free Yahoo bars (hourly for two years, 30-minute for 60 days).
- `src/02_taq_pull.py`: pulls three trade snapshots a day (10:00, 15:30, close) per symbol from WRDS TAQ, with monthly checkpoints.
- `src/03_analysis.py`: the regressions, per-stock statistics, and the liquidity, universe, momentum and price breakdowns.

TAQ data is licensed through WRDS and is not included. `02_taq_pull.py` needs a WRDS account. The symbol universe comes from iShares Russell 1000, Midcap and 2000 holdings files saved in `data/etf_holdings/`, plus an optional `data/universe.csv`.
