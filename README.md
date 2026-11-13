# Polymarket NVIDIA-CEO price forecaster

Entry for make.fun competition `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every trading market of Polymarket's "Next CEO of NVIDIA?" event
at every minute of the 4 hours after each checkpoint, as five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95). Python 3.13, standard library only, no network.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory, writes `<id>.out` per case.
One run takes ~70–90 ms per case (cap: 8,000 ms). Output is deterministic.

## Method

- **Median = last price (persistence).** Prediction-market prices are close to
  a martingale at this horizon. Every drift, trend and mean-reversion variant
  of the median tested on the practice cases scored worse than persistence.
- **Per-market spread from recent activity.** Each market's 1-minute move size
  is estimated as an exponentially weighted mean of |price changes| with a
  60-minute half-life, ×2.0, clamped to [0.0003, 0.003] per √minute. A plain
  standard deviation is ruined by the rare huge jumps; the recency-weighted
  mean absolute move tracks the market's current regime instead.
- **Peaked, asymmetric quantile shape.** Quantile multipliers
  [-3.0, -0.35, 0, 0.35, 2.0] × spread × √h: prices sit still ~88% of minutes
  (narrow interquartile range) but occasionally jump (wide tails), with a wider
  downside tail than upside.
- **Young markets widened 1.5×** while they have under 2 hours of history.
- Spreads grow as √h, are clamped to [0, 1] and rounded to 6 decimals.

## Validation (practice cases, scored against realized prices)

Realized prices for p00–p04 taken from the next case's own input file (same
1-minute grid); p05 fetched from Polymarket's public price history
(`clob.polymarket.com/prices-history`, fidelity=1, gridded per `input/NOTICE.txt`).
The scorer reproduces the house baseline's published practice loss exactly
(0.363589), confirming the realized data.

| case | this entry | house baseline |
|------|-----------:|---------------:|
| p00  | 0.481744   | 0.536138 |
| p01  | 0.311922   | 0.380352 |
| p02  | 0.333970   | 0.408408 |
| p03  | 0.253349   | 0.368599 |
| p04  | 0.211876   | 0.298794 |
| p05  | 0.113495   | 0.238232 |
| **mean** | **0.273914** | **0.363589** |

Mean pinball loss 0.273914 vs baseline 0.363589 (−24.7%), better in every
practice case. 90% coverage 0.9193 (nominal 0.9), median error 0.7675 points.

Things tested and rejected on the practice data: per-market standard-deviation
volatility (far worse — jumps inflate it), empirical quantiles of forward moves,
trend drift, mean reversion toward the recent mean, smoothed medians,
cross-market renormalization of medians to sum 1, and jump-reversal terms.
