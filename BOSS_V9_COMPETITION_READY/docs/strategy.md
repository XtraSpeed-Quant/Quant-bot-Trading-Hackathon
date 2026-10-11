# BOSS V9 Mathematical Strategy and Exit Rules

## 1. Objective

BOSS is a cross-sectional long/short portfolio. It ranks the full eligible USD market universe, forms directional candidates, and solves a constrained portfolio problem every rebalance. It does not optimize for trade count. It seeks positive expected net edge after execution friction while penalizing portfolio covariance risk and unnecessary changes in exposure.

For candidate asset weights \(w\), estimated horizon returns \(\alpha\), horizon covariance \(\Sigma_h\), prior signed weights \(w^-\), and one-way trading costs \(c_i\), the live numerical objective is:

\[
\max_w\quad \alpha^\top |w| - \frac{\lambda}{2} w^\top\Sigma_h w - \eta\sum_i c_i|w_i-w_i^-|.
\]

The sign of each candidate is fixed by the strategy ranker: positive weights are longs and negative weights are shorts. The optimizer applies long/short gross budgets, a total gross-exposure budget, and per-asset position bounds. There is no daily turnover quota. The covariance matrix is estimated from recent log returns with exponential time-decay (a 180-minute half-life by default), shrunk toward a diagonal target to reduce estimation error, and projected to positive semidefinite form. The expected-return horizon and covariance horizon both default to the hourly rebalance/forecast interval. A proximal-gradient solver handles the absolute reallocation-cost term.

Here, \(\alpha\) is a strategy estimate, not a known future return. It comes from a validated ridge ranker when its chronological validation gates pass; otherwise the strategy uses its deterministic momentum/relative-strength edge proxy. This is why the replay should be judged on later, unseen periods rather than the optimizer objective alone.

## 2. Signal construction

1. **Momentum across horizons:** 1-hour, 4-hour, 6-hour, 12-hour, and 24-hour relative return features capture persistence and acceleration where enough history exists. All windows are in elapsed minutes and are converted to bars using `POLL_SECONDS`.
2. **Trend, breakout and breakdown:** price relative to its EMA and prior highs/lows distinguishes continuation from failed impulse. Long and short sides have separate ranks and eligibility conditions.
3. **Cross-sectional relative strength:** performance is measured against market medians, not solely against an asset's own history.
4. **Execution quality:** quote volume and quoted spread are checked because gross predictive edge is not enough if it is smaller than expected execution cost.
5. **Predictive overlay:** a regularized ridge ranker uses lagged momentum/trend/volatility/market-relative features. A separate directional classifier remains gated unless its purged chronological validation passes. Predictor training samples span the stored history rather than concentrating on the most recent snapshots.

## 3. Volatility-adaptive exits

BOSS estimates each asset's realized hourly volatility \(\sigma_i\) as the sample standard deviation of recent non-overlapping 60-minute log returns. Using non-overlapping returns avoids pretending that many highly-overlapping 60-second observations are independent hourly samples. With round-trip friction \(C_i\), the exit thresholds are:

- Hard stop: \(d_{stop,i}=\max(k_{stop}\sigma_i,2C_i)\).
- Trailing activation: \(d_{activate,i}=\max(k_{activate}\sigma_i,2C_i)\).
- Trailing distance: \(d_{trail,i}=\max(k_{trail}\sigma_i,C_i)\).
- Take-profit distance: \(d_{profit,i}=\max(k_{profit}\sigma_i,3C_i)\).

Short-side stop and trailing multipliers are configured separately to allow for squeeze asymmetry. The cost floors keep thresholds outside modeled fees, spread and slippage. A static legacy exit mode is available with `ADAPTIVE_EXITS_ENABLED=false`; the live template enables adaptive exits.

At the sampled price path, stop, trailing, and take-profit conditions can force the relevant position to close. The research replay tests the same configured thresholds against intervening snapshots, charges estimated exit costs, and then uses the resulting signed weights for the next rebalance. A sampled-data stop may still fill worse than its trigger price during a fast market; real fills must be used for the final performance report.

## 4. Long and short handling

Long positions are spot holdings. Shorts use the configured competition short API and are tracked using collateral, quantity, entry price, and unrealized P&L. A spot sell does not create a short. Both sides enter the joint signed optimizer and share the total gross budget. The current portfolio is supplied to the optimizer so the objective prices the cost of rebalancing rather than treating each cycle as if it started from cash.

## 5. Research and parameter selection

`tools/walkforward_optimize.py` (requires at least 48 hourly decision windows after warm-up):

1. Reads a SQLite price tape without submitting exchange requests.
2. Splits on large timestamp gaps and resamples the longest continuous segment to a fixed cadence rather than pretending a gap is a normal bar.
3. Uses the earlier chronological segment for parameter selection and reserves later windows as an untouched out-of-sample report.
4. Searches portfolio risk aversion, edge entry thresholds, cost penalty, covariance shrinkage, and three predeclared volatility-exit parameter sets. The search spans risk aversion 20–120 in the full grid.
5. Ranks training candidates with the predeclared composite `annualized Sharpe - 2 × maximum drawdown + 2 × net total return` and reports the chosen candidate and the default baseline on the later segment.
6. Deducts estimated fees, spread, slippage and stop/exit costs. It does not model order-book impact, queue position, partial fills, API rejects or any borrow/funding charges absent from the price tape.

Do not repeatedly tune to the reported out-of-sample segment and then continue calling it untouched. Use a fresh later period for the next decision. The optimizer is a parameter-selection method, not a proof of a profitable edge.

## 6. Runtime configuration

| Parameter | V9 default | Role |
|---|---:|---|
| `TOP_N` / `SHORT_TOP_N` | 20 / 20 | Number of highest-ranked candidates per side, not the number of markets scanned |
| `MAX_GROSS_TARGET` | 0.70 | Strategy's long-book budget before regime scaling |
| `MAX_SHORT_GROSS_TARGET` | 0.42 | Strategy's short collateral budget before regime scaling |
| `MAX_TOTAL_GROSS_TARGET` | 0.98 | Joint long-plus-short gross-exposure budget |
| `PORTFOLIO_RISK_AVERSION` | 60.0 | Weight on covariance risk in optimizer utility; search explores a broad scale range |
| `PORTFOLIO_COST_PENALTY` | 1.0 | Weight on modeled marginal rebalancing cost |
| `PORTFOLIO_COVARIANCE_SHRINKAGE` | 0.65 | Shrinkage toward a diagonal covariance target |
| `PORTFOLIO_COVARIANCE_HALF_LIFE_MINUTES` | 180 | EW covariance half-life; recent returns receive more weight |
| `ADAPTIVE_EXITS_ENABLED` | true | Scales exits with realized volatility and estimated execution friction |
| `REBALANCE_BAND` | 0 | No extra relative-weight deadband; normal exchange/minimum-order checks still apply |
| `MAX_DAILY_TURNOVER` / `MAX_ORDER_USD` | 0 / 0 | Internal turnover and per-order caps disabled |
| `ROTATION_MAX_PER_REBALANCE` | 0 | Internal rotation-count cap disabled |
| `DAILY_LOSS_LIMIT_PCT` / `MAX_DRAWDOWN_LIMIT_PCT` | 0 / 0 | Optional portfolio-level trade halts disabled |

The portfolio gross/asset weights and per-position exit rules are strategy design, not hidden turnover quotas. Cash/collateral availability, exchange precision/minimums and the exchange's own rate/permission limits remain unavoidable.
