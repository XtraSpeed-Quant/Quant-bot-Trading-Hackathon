# BOSS V9 changelog

## Mathematical strategy and execution refinements

- Replaced score/volatility-only allocation with a joint signed mean-variance optimizer. It maximizes expected edge minus covariance risk and the modeled marginal cost of reallocating current signed exposures.
- Added covariance shrinkage, PSD projection, long/short/total gross budgets, and per-asset constraints to the numerical portfolio solver.
- Added volatility-adaptive hard stops, trailing activation/distances and take-profit thresholds, using non-overlapping hourly log returns. Stop distances are compared against fee/spread/slippage friction floors; static exits remain available with `ADAPTIVE_EXITS_ENABLED=false`.
- Expanded ranked candidates to 20 longs and 20 shorts while preserving the full eligible trading-universe scan.
- Set `REBALANCE_BAND=0`, `MAX_ORDER_USD=0`, `MAX_DAILY_TURNOVER=0`, and `ROTATION_MAX_PER_REBALANCE=0` in the deployment template so no extra internal threshold/cap suppresses normal strategy orders (exchange minimums/precision still apply).
- Preserved chronological, gap-aware walk-forward optimization. It refuses to merge data across missing intervals or poll-cadence changes.
- Added adaptive-exit unit tests and kept the optimizer's risk/cost, direction, and exposure tests.

## Research limitation

The previous archived database contains mixed cadences and large gaps. Its longest uninterrupted 300-second segment is only about 5.9 hours, shorter than the model's warm-up. Therefore there is no valid out-of-sample profitability estimate from that archive. V9 does not label its default parameters optimal; the first clean week of collection must be evaluated with the included walk-forward tool before any empirical claim can be made.

- Aligned portfolio risk covariance horizon with the hourly forecast/rebalance horizon, moved covariance estimation to exponentially weighted returns with a 180-minute half-life, and broadened risk-aversion search to 20/60/120; the deployment starting point is 60 and the one-hour covariance horizon is aligned with the forecast/rebalance cycle.
- Increased the walk-forward minimum to 48 hourly decision windows and preserves at least 12 later windows as a final untouched holdout.


## V9 competition-specific corrections

- Prefilled the API host documented by the official Roostoo REST repository (`https://mock-api.roostoo.com`) because the linked event describes the real-time mock-exchange venue. API keys remain unfilled.
- Aligned the default portfolio to the event's $100,000 starting capital, 0.1% taker commission, and 0.05% maker commission.
- Changed data sampling to 60 seconds and portfolio evaluation to 120 seconds to avoid a high-frequency request pattern, while retaining unlimited internal turnover/order/rotation settings.
- Fixed a critical live short-close call so it uses only documented `/v6/short_close` parameters; the simulator-only close price remains local. Added a test to enforce the endpoint contract.
- Made the account refresh schedule configurable and aligned docs, DB names and walk-forward command defaults for V9.
- The Notion Data Sources Pack was not accessible in this environment; no unavailable content is assumed.
