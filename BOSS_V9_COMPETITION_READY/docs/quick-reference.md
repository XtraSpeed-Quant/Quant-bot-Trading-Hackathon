# BOSS V9 Quick Reference

## One-week AWS defaults

```dotenv
PAPER_MODE=false
ORDERS_ENABLED=true
POLL_SECONDS=60
REBALANCE_SECONDS=120
EXCHANGE_INFO_REFRESH_SECONDS=900
HISTORY_LIMIT=2000
PREDICTOR_HISTORY_LIMIT=25000
PREDICTION_RETRAIN_SECONDS=1800
SHORTS_ENABLED=true
MAX_DAILY_TURNOVER=0
MAX_ORDER_USD=0
ROTATION_MAX_PER_REBALANCE=0
DAILY_LOSS_LIMIT_PCT=0
MAX_DRAWDOWN_LIMIT_PCT=0
BOT_DB_PATH=data/boss_v9_live.sqlite3
```

The API host is prefilled as `https://mock-api.roostoo.com`; fill both credentials first. The V9 example config is live mode and should not be started with dummy credentials.

## Quick commands

```bash
# Check public server-time and market-universe endpoints (no orders)
python -m roostoo_bot --check-api

# View signals and query account/short-position state (no orders)
python -m roostoo_bot --inspect

# Tests
python -m unittest discover -s tests -v

# Continuous deployment
docker compose up -d --build
docker compose logs -f boss
```

## Unlimited controls

- `MAX_DAILY_TURNOVER=0`: no internal daily turnover cap.
- `MAX_ORDER_USD=0`: no internal per-order notional cap.
- `ROTATION_MAX_PER_REBALANCE=0`: no internal rotation-count cap.
- `DAILY_LOSS_LIMIT_PCT=0` / `MAX_DRAWDOWN_LIMIT_PCT=0`: optional account-level trade halts disabled.

Portfolio weights, position-specific exits, API precision/minimums, cash balance, short collateral and real exchange permissions still apply.

## What 7 days means

At 60-second sampling, one uninterrupted week is approximately 10,080 timestamps per market. `HISTORY_LIMIT` only bounds the history read for current signals; it does not delete old records. The predictor reads up to 25,000 snapshots per pair. Keep the V9 database persistent and do not mix the older 5-minute snapshot DB into it.

## Database checks

```sql
SELECT pair, COUNT(*) AS samples,
       MIN(timestamp_ms) AS first_ts,
       MAX(timestamp_ms) AS last_ts
FROM prices
GROUP BY pair
ORDER BY samples DESC;

SELECT side, COUNT(*) AS count, SUM(notional) AS turnover
FROM trades
GROUP BY side
ORDER BY count DESC;
```
