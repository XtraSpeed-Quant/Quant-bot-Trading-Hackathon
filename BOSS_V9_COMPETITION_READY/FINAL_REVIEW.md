# BOSS V9 — Competition-Specific Final Review

## Result

Prepared a release for a continuous seven-day AWS trading/data-collection run, tailored to the HK vs AU vs IN Quant Trading Hackathon's published rules: $100,000 initial portfolio, 1× spot longs and shorts only, 0.1% taker commission, 0.05% maker commission, and no HFT, market-making, arbitrage, leverage or excessive API calls.

## Operating profile

- API base URL: `https://mock-api.roostoo.com`, the public REST host in the linked Roostoo API repository. The event page describes the venue as Roostoo's real-time mock exchange; confirm that event organizers have not issued a dedicated alternate host for your account.
- Credentials: participant-specific API key and secret remain blank in `.env.example`.
- Polling: 60 seconds; rebalance evaluation: 120 seconds; exchange metadata refresh: 900 seconds; account refresh: every 120 seconds.
- Recent signal history: 2,000 snapshots; predictor history: 25,000 snapshots; predictor refit schedule: every 1,800 seconds when valid training history is available.
- `$100,000` starting cash; `SHORTS_ENABLED=true`; spot/short fee assumptions reflect the event's published commissions. Short endpoints use collateral-based `/v6/short_open`, parameter-limited `/v6/short_close`, and `/v6/short_positions`.
- `MAX_ORDER_USD=0`, `MAX_DAILY_TURNOVER=0`, `ROTATION_MAX_PER_REBALANCE=0`: internal order-size, turnover and rotation-count caps disabled.
- `DAILY_LOSS_LIMIT_PCT=0`, `MAX_DRAWDOWN_LIMIT_PCT=0`: optional portfolio-wide trade halts disabled. Position-level exits remain active.
- All API-listed tradable USD markets remain eligible. `DIAGNOSTIC_TOP_N` affects dashboard diagnostics only, not trade universe size.
- Docker Compose restarts the process after failure and persistently mounts SQLite/log directories.
- The older snapshot database is excluded so its historical cadence is not mixed with new 60-second observations.

## Verification completed

- **46/46 unit tests passed**, including live short-close contract validation and regression tests for the no-history `--inspect` dashboard path.
- Python compilation completed successfully for bot code, tests, and tools.
- `python -m roostoo_bot --help` succeeded.
- Environment template uses `https://mock-api.roostoo.com`, `$100,000` initial capital, 60-second polling, 120-second rebalancing, 0.1% taker fee, 0.05% maker fee, shorts enabled, no internal turnover/order/rotation caps, and disabled global-loss halts. Team API keys remain blank.
- Docker build/runtime was not tested because Docker is not installed in this environment. Live trading with your credentials was not tested.

## Important API correction

The documented live `/v6/short_close` endpoint accepts `pair`, `timestamp`, and optional `close_qty` or `close_pct`; it does not take a price argument and fills at the current best ask. The previous caller sent an unsupported `price` keyword, which would have caused a live short-close failure. V9 fixes this and includes a regression test. `/v6/short_open` sizes by USD collateral and applies the documented 0.1% short-opening fee.

## Data Sources Pack status

The GitHub API document was inspected directly. The Luma event page was reviewed and the competition's rules/portfolio objective were reflected in deployment docs. The Notion Data Sources Pack could not be fetched from the current environment; its contents have not been inferred. If that pack provides required market/news/on-chain data sources, those still need to be incorporated explicitly.

## Competition objective and research limits

The published competition objective first ranks portfolio return; its risk-adjusted composite is `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar`. BOSS uses a covariance-aware long/short allocator, cross-sectional momentum/relative strength, trend/volatility, breakouts/breakdowns, market-quality filters, regularized prediction, and volatility-adaptive exits. These are defensible starting methods, not a guarantee of profit. The archived sample has mixed cadence and large gaps, with a longest clean segment around 5.9 hours; it cannot support a trustworthy winning parameter selection. Use the new clean run for chronological walk-forward evaluation with actual fills, commissions, spread/slippage, and short collateral flows.

## Ready-to-run checklist

1. `cp .env.example .env` and set `ROOSTOO_API_KEY` and `ROOSTOO_API_SECRET`.
2. Verify `ROOSTOO_API_URL=https://mock-api.roostoo.com` unless organizers have given you a separate host.
3. Run `python -m roostoo_bot --check-api` and `python -m roostoo_bot --inspect`.
4. Build with `docker compose build`; launch with `docker compose up -d`; monitor `docker compose logs -f boss`.
5. Keep `data/` and `logs/` on persistent storage for the full collection period. Do not commit `.env`.

## V9.1 follow-up fix

The release now includes `V9.1_FIXLOG.md`. A post-release run of `python -m roostoo_bot --inspect` exposed a warm-up dashboard crash when the new database had no usable market history. The empty `StrategyResult` had positional fields, which placed `"warming up"` in the numeric Brier metric after a field was added. This is fixed using keyword construction and covered by two regression tests. The missing `.env.example` referenced by the deployment docs is also now included.
