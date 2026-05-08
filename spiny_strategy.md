# spiny.py Strategy Explanation

`spiny.py` implements a simplified backtest of a "complacency fade" volatility strategy for AI semiconductor stocks. The model looks for periods when the sector appears unusually calm, then enters a long-volatility trade on the assumption that this calm is unstable and likely to break with a large move.

## Core idea

The strategy assumes AI semiconductor names such as `NVDA`, `AVGO`, and `TSM` tend to alternate between:

- quiet, low-volatility drift
- sudden high-volatility repricing

The goal is to identify the quiet regime early enough to own volatility before the repricing happens.

In code terms, "complacency" means recent market behavior is unusually calm relative to its own history. When that complacency becomes extreme, the strategy enters a trade. It exits after the regime normalizes, flips to fear, or the hold period gets too long.

## Universe and data

The primary universe is:

- Index proxy: `SMH`
- Constituents: `NVDA`, `AVGO`, `TSM`

There is also an extra universe:

- Index proxy: `SOXX`
- Constituents: `NVDA`, `AVGO`, `TSM`

The script downloads about 5 years of daily adjusted close data from Yahoo Finance for the index, constituents, and `^VXN` as an implied-volatility proxy.

## Features the signal uses

For each ticker, the script computes daily log returns and then builds four features that are meant to describe whether the sector is calm or stressed.

### 1. Short vs long realized volatility

For each asset:

- short realized vol = 10-day rolling standard deviation of returns, annualized
- long realized vol = 63-day rolling standard deviation of returns, annualized
- term structure ratio = short vol / long vol

If this ratio is low, recent movement is quieter than the longer-run norm. That is treated as complacent.

### 2. IV/RV proxy

The script uses `^VXN` as the base implied-volatility proxy. For each constituent, it scales that proxy by the constituent's short-vol ratio relative to the index. That produces an approximate constituent IV.

Then it computes:

- IV/RV = implied vol proxy / short realized vol

This is clipped to the range `0.3` to `5.0`.

This is not true option-surface data. It is a proxy used to keep the backtest self-contained and easy to run.

### 3. Average constituent correlation

The strategy calculates 10-day rolling pairwise correlations among the constituent returns and averages them.

Low average correlation is interpreted as a relaxed market where names are trading on separate stories rather than moving together in stress.

### 4. Return skew

The script computes 21-day rolling skewness of constituent returns and averages it across the basket.

This is used as another regime descriptor. In practice, it helps distinguish smooth upside drift from more unstable recent return distributions.

## Composite complacency signal

The basket-level inputs are:

- average constituent term-structure ratio
- average constituent IV/RV
- average constituent correlation
- average constituent skew

Each of those is converted into a rolling percentile over a 252-day lookback. The script then combines them into a weighted composite score:

```text
0.35 * term-structure percentile
+ 0.30 * IV/RV percentile
+ 0.20 * correlation percentile
+ 0.15 * skew percentile
```

That composite score is itself ranked over the same 252-day lookback to get `composite_pctile`.

Interpretation:

- low `composite_pctile` = unusual calm = complacency
- high `composite_pctile` = stress/fear

The regime rules are:

- `COMPLACENT` if composite percentile <= `15`
- `FEAR` if composite percentile >= `75`
- `NEUTRAL` otherwise

## Entry logic

The backtester uses yesterday's regime to decide today's trade action, which avoids lookahead bias.

If the strategy is flat and the prior day's regime is `COMPLACENT`, it opens a position named:

`SHORT_UNDERLYING_LONG_VOL`

In the current default configuration, only the long-vol leg is active:

- `enable_options_leg = True`
- `enable_underlying_leg = False`

So although the position name includes a short-underlying leg, the default live backtest behavior is effectively options-only.

## What the position is trying to represent

The position is a simplified abstraction of:

- optional short exposure to the semiconductor index and basket
- long gamma / long vega style exposure through options

This is not priced from real options chains. The script approximates option behavior with a few daily P&L components.

## Exit logic

Once in a trade, the script tracks holding days and exits when one of the following occurs:

1. max hold is reached: `42` days
2. after at least `7` days, the regime flips to `FEAR`
3. after at least `7` days, the composite percentile rises above `50`

Those rules are designed to:

- avoid exiting too early
- capture the volatility snapback
- stop holding once the original complacency thesis is no longer present

## P&L model

Each open trade accrues daily P&L from four components.

### 1. Underlying P&L

```text
- underlying_weight * (0.5 * index return + 0.5 * average constituent return)
```

This only contributes if `enable_underlying_leg` is turned on. By default it is disabled, so this term is zero.

### 2. Theta decay

```text
- options_weight * theta_daily_long
```

This is the daily carrying cost of being long options. Default theta cost is `0.00035` per day on the options notional.

### 3. Gamma-style payoff

```text
options_weight * 0.5 * max(0, realized_sq - implied_daily_var) * 100
```

Where:

- `realized_sq` is the square of the average constituent return for the day
- `implied_daily_var` is derived from the index short realized vol

This term rewards days where realized movement is larger than what the implied-vol proxy suggested.

### 4. Vol-move payoff

```text
options_weight * 0.8 * change in index short realized vol
```

This is a crude vega-style term. If short realized vol rises from one day to the next, the long-vol position benefits.

## Trading costs

The script charges slippage at entry and exit:

- stock slippage: `0.001`
- options slippage: `0.012`

These are added as trading costs to the trade record and deducted from daily P&L when the trade is opened or closed.

## Why the strategy can work

The thesis is straightforward:

- calm regimes often compress recent realized volatility, correlation, and perceived need for protection
- those calm regimes can reverse abruptly in semiconductor names
- long-vol exposure benefits when realized moves and volatility expand

The composite signal is just a systematic way to define "the market looks too calm relative to its own recent history."

## Important simplifications and limitations

`spiny.py` is a research backtest, not an execution-ready options model. Important approximations include:

- implied volatility is proxied using `^VXN`, not real option chains
- option P&L is represented by stylized theta, gamma, and vol-move formulas
- no strike, expiry, delta, or surface-level option modeling is used
- the underlying short leg exists in the framework but is disabled by default
- the same constituent basket is reused across the `SMH` and `SOXX` runs

That means the script is best understood as a regime-testing engine: it checks whether entering long-vol exposure during extreme complacency would have had a positive profile, without pretending to be a full options pricer.

## In one sentence

`spiny.py` buys simplified long-vol exposure in AI semiconductor regimes that look historically calm, then exits once volatility normalizes, fear arrives, or the trade has been held too long.
