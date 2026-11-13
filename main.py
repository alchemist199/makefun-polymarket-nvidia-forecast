#!/usr/bin/env python3
"""Polymarket "Next CEO of NVIDIA?" minute-by-minute price forecaster (round 3).

Reads every <id>.in case file from $MAKEFUN_INPUT_DIR (default /input) and writes
one <id>.out forecast file per case to $MAKEFUN_OUTPUT_DIR (default /output).

Forecast format: {"forecasts": [{"id": <conditionId>, "q": [[q05, q25, q50, q75, q95]
x 240 rows]}]}, one row per minute after the case's checkpoint.

Method (round 2 kept, median rebuilt from the practice histories)
------------------------------------------------------------------
Spread and quantile shape are round 2's: the spread scale is an exponentially
weighted mean of |1-minute price changes| (60-minute half-life) x 1.75,
clamped to [0.0003, 0.003] per square-root minute, x1.5 for markets with
under 2 hours of history; the quantile multipliers blend from the quiet shape
[-3.0, -0.6, 0, 0.2, 2.0] below price 0.10 to the active shape
[-3.0, -0.35, 0, 0.35, 2.0] above 0.30. The spread grows as sqrt(h).

Round 3 rebuilds the MEDIAN (the last price, as before) with three
median-fitted terms, all fitted on the practice histories' minutes only
(never on the realized 4-hour windows) and validated on the six practice
cases against the realized prices:

1. Small level pull. The MEDIAN future move (not the mean -- the mean is
   dominated by rare catastrophic decays) of a market sitting at price x is
   about 0 below 0.15, -0.026 at x = 0.225 and -0.035 at x >= 0.45 over 240
   minutes, growing roughly linearly in h. We apply a small fraction of that
   fitted pull (scale 0.15 on the piecewise-linear drift, i.e. about -0.004
   at H240 for a market at 0.30), because the full fitted pull overfits: it
   helps when the leader falls and hurts when it rises, and the practice
   checkpoints are split on that. An OLS fit of the conditional MEAN move was
   tried first and was catastrophic (0.75): the mean move is ~4x the median
   move for high-price markets, because a few decays dominate it.

2. Asymmetric jump reversion. A big UP jump reverts strongly in median terms
   (a +0.23 jump comes back ~0.20 by H240, ~85%), while a DOWN jump does not
   revert at all (median future move ~0.000). So only up-moves are reverted,
   by a horizon-dependent fraction (0.05 at H1 rising to 0.85 at H240) times
   0.7, and only when the last minute moved at least 0.005. Round 2 reverted
   both directions symmetrically at 0.5, which over-reverts down-jumps ~2.5x.

3. Cross-market consistency pull (top-3 markets only). The 24 markets are
   mutually exclusive, so their prices should sum to ~1. When the sum S is
   above 1 the book is overpriced and the liquid top markets absorb the
   adjustment: on history minutes the correlation of (S-1) with a top-3
   market's future move is -0.23 at H60 (slope about -0.16). The median of
   each of the top 3 markets (by last price) is pulled by
   -0.12 * (S - 1) * sqrt(h/240), in either direction: down when the book is
   overpriced (p00 S=1.079, p04 S=1.086), up when it is underpriced
   (p02 S=0.808). Dead low-price markets do not move, so they get no pull.

On the six practice cases this scores a mean pinball loss of 0.259868,
against round 2's 0.267093 and the house baseline's 0.363589, and it beats
round 2 in 5 of 6 cases (p00 -0.0322, p01 -0.0002, p02 -0.0049, p03 -0.0003,
p04 -0.0088; p05 +0.0001, a tie). Leave-one-case-out (refit the three scales
on 5 cases, evaluate the held-out one): no held-out case is worse than round 2
by more than +0.0023, so the model does not collapse out of sample. The
margins are thin (the mean clears 0.260 by 0.0001), because the remaining loss
is dominated by unpredictable jumps: a per-market oracle sigma would only
reach 0.244, and the optimal sigma for a given market swings from 0.5x to
2.2x across cases depending on whether that market happened to jump.
Everything is deterministic: same input, same bytes out. Standard library
only, no network, no randomness.
"""

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]

# Quantile multipliers around the median, per regime (round 2, unchanged).
# Quiet low-price markets: mildly asymmetric IQR (downside wider than upside).
Z_QUIET = [-3.0, -0.6, 0.0, 0.2, 2.0]
# Active high-price markets: round-1 shape (narrow IQR, wide tails).
Z_ACTIVE = [-3.0, -0.35, 0.0, 0.35, 2.0]
# Price levels between which the shape blends from quiet to active.
BLEND_LOW = 0.10
BLEND_HIGH = 0.30

# Per-market scale (price units per square-root minute) from the exponentially
# weighted mean of |1-minute price changes|.
HALF_LIFE_MIN = 60.0      # EWMA half-life in minutes
SCALE_MULT = 1.75         # multiplier on the EWMA mean |change|
SIGMA_FLOOR = 0.0003      # quietest markets still get this much spread
SIGMA_CAP = 0.003         # jumpy markets are capped here
# Markets that have barely traded yet are more uncertain than their short
# history suggests: widen their spread while they have under 2 hours of data.
SHORT_HISTORY_MIN = 120
SHORT_HISTORY_BOOST = 1.5

# --- Round-3 median model (fitted on practice-history minutes) ---
# 1. Level pull: median 240-minute drift at price x (piecewise linear).
#    Fitted medians: 0 below 0.15, -0.0255 at 0.225, -0.0350 at >= 0.45.
DRIFT_PTS = [(0.0, 0.0), (0.15, 0.0), (0.225, -0.0255), (0.45, -0.0350),
             (1.0, -0.0350)]
DRIFT_SCALE = 0.15        # small fraction of the fitted pull (full pull overfits)
# 2. Asymmetric jump reversion: fraction of an UP jump reverted by horizon h
#    (interpolated linearly in sqrt(h)); DOWN jumps are not reverted.
REV_H_GRID = [1, 5, 15, 30, 60, 120, 240]
REV_UP = {1: 0.05, 5: 0.15, 15: 0.30, 30: 0.45, 60: 0.70, 120: 0.90,
          240: 0.85}
REV_UP_SCALE = 0.7         # shrink the fitted reversion fraction
REV_THRESHOLD = 0.005      # only revert when the last minute moved this much
# 3. Cross-market consistency pull for the top-N markets by last price.
XSUM_COEF = 0.12           # pull per unit of (sum of prices - 1)
XSUM_TOP_N = 3


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def market_sigma(prices):
    """Per-market spread scale from recent 1-minute move sizes.

    Exponentially weighted mean of |price changes| with a 60-minute half-life,
    walking backwards from the checkpoint so the most recent minutes weigh
    most. Falls back to the floor when there is no usable history.
    """
    if len(prices) < 2:
        return SIGMA_FLOOR
    decay = 0.5 ** (1.0 / HALF_LIFE_MIN)
    weighted_sum = 0.0
    weight_total = 0.0
    weight = 1.0
    for i in range(len(prices) - 1, 0, -1):
        weighted_sum += weight * abs(prices[i] - prices[i - 1])
        weight_total += weight
        weight *= decay
        if weight < 1e-4:
            break
    if weight_total <= 0.0:
        return SIGMA_FLOOR
    sigma = SCALE_MULT * weighted_sum / weight_total
    sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
    if len(prices) < SHORT_HISTORY_MIN:
        # Applied after the cap on purpose: a young market's spread may
        # legitimately exceed the cap for established markets.
        sigma *= SHORT_HISTORY_BOOST
    return sigma


def blend_z(x):
    """Quantile multipliers for a market at price x.

    Blends the quiet shape into the active shape linearly with the price
    level: fully quiet at or below BLEND_LOW, fully active at or above
    BLEND_HIGH.
    """
    w = (x - BLEND_LOW) / (BLEND_HIGH - BLEND_LOW)
    w = min(1.0, max(0.0, w))
    return [Z_QUIET[j] * (1.0 - w) + Z_ACTIVE[j] * w for j in range(5)]


def drift240(x):
    """Fitted median 240-minute drift of a market sitting at price x."""
    if x <= DRIFT_PTS[0][0]:
        return DRIFT_PTS[0][1]
    for i in range(len(DRIFT_PTS) - 1):
        x0, y0 = DRIFT_PTS[i]
        x1, y1 = DRIFT_PTS[i + 1]
        if x0 <= x <= x1:
            w = (x - x0) / max(x1 - x0, 1e-9)
            return y0 * (1.0 - w) + y1 * w
    return DRIFT_PTS[-1][1]


def rev_up_at(h):
    """Fitted fraction of an up-jump reverted by horizon h (sqrt-interpolated)."""
    if h <= REV_H_GRID[0]:
        return REV_UP[REV_H_GRID[0]]
    if h >= REV_H_GRID[-1]:
        return REV_UP[REV_H_GRID[-1]]
    for i in range(len(REV_H_GRID) - 1):
        if REV_H_GRID[i] <= h <= REV_H_GRID[i + 1]:
            s0 = math.sqrt(REV_H_GRID[i])
            s1 = math.sqrt(REV_H_GRID[i + 1])
            w = (math.sqrt(h) - s0) / (s1 - s0)
            return REV_UP[REV_H_GRID[i]] * (1.0 - w) + REV_UP[REV_H_GRID[i + 1]] * w
    return REV_UP[REV_H_GRID[-1]]


def forecast_market(market, horizon, xsum_pull):
    """240 rows of five quantiles for one market.

    xsum_pull is the cross-market consistency pull for this market
    (0 for markets outside the top 3): -XSUM_COEF * (S - 1) * sqrt(h/240),
    applied inside the loop via the precomputed per-horizon factor.
    """
    prices = market["prices"]
    if not prices:
        raise ValueError("market %s has no prices" % market["id"])
    x = prices[-1]
    sigma = market_sigma(prices)
    z = blend_z(x)
    # Last minute's move, for the asymmetric jump reversion.
    last_move = prices[-1] - prices[-2] if len(prices) >= 2 else 0.0
    up_jump = last_move if last_move >= REV_THRESHOLD else 0.0
    # Level pull at H240, scaled down (see module docstring).
    level_pull = drift240(x) * DRIFT_SCALE
    rows = []
    for h in range(1, horizon + 1):
        med = x + level_pull * (h / 240.0)
        # Only up-jumps revert (down-jumps do not, in median terms).
        med -= rev_up_at(h) * REV_UP_SCALE * up_jump
        # Cross-market consistency pull (top-3 markets only; 0 otherwise).
        med -= xsum_pull * math.sqrt(h / 240.0)
        med = min(1.0, max(0.0, med))
        spread = sigma * math.sqrt(h)
        rows.append([price6(med + z[j] * spread) for j in range(5)])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for filename in inputs:
        case_id = filename[: -len(".in")]
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            case = json.load(f)
        if list(case["quantiles"]) != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %r, not %r"
                % (filename, case["quantiles"], QUANTILES)
            )
        horizon = case["horizon_minutes"]
        markets = case["markets"]
        # Cross-market consistency: the 24 mutually exclusive prices should
        # sum to ~1. The top-3 markets (by last price) absorb the adjustment.
        total = sum(m["prices"][-1] for m in markets if m["prices"])
        excess = total - 1.0
        ranked = sorted(markets, key=lambda m: -m["prices"][-1])
        top_ids = set(m["id"] for m in ranked[:XSUM_TOP_N])
        forecasts = []
        for m in markets:
            pull = XSUM_COEF * excess if m["id"] in top_ids else 0.0
            forecasts.append({"id": m["id"], "q": forecast_market(m, horizon, pull)})
        with open(
            os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
