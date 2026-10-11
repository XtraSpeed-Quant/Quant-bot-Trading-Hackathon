# BOSS V9 Deployment Guide — Roostoo hackathon AWS run

## Competition constraints that affect deployment

The event uses Roostoo's real-time mock exchange. It permits 1× spot longs and shorts, lists a $100,000 initial portfolio, and charges 0.1% taker / 0.05% maker commission. Rules forbid high-frequency trading, market-making, arbitrage, excessive API calls and leverage. The 60-second ticker collection and 120-second rebalance schedule target a directional strategy rather than HFT. API request spacing only avoids bursts; there is no bot-level turnover/order-count quota.

The event's risk-adjusted composite is `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar`, after leaderboard qualification by net portfolio return. The system records daily equity and trade data so those metrics can be evaluated after the run.

## 1. Verify API access first

The event page describes the venue as Roostoo’s real-time mock exchange backend. The official Roostoo API repository lists `https://mock-api.roostoo.com` as its REST base URL, which is prefilled in `.env.example`. If organizers provided a dedicated URL for your competition account, use it instead. Confirm the key has spot-trading and short-open/close permissions before deploying.

Do not test production order placement with a dummy URL or dummy keys. The read-only checks below do not place orders:

```bash
python -m roostoo_bot --check-api
python -m roostoo_bot --inspect
```

`--inspect` reads current tickers/balances and attempts to read short positions in live mode; that may fail if the API credential does not have the expected permission. Fix that with the competition/API owner rather than silently treating spot sells as shorts.

## 2. AWS host and persistent storage

A reasonable baseline is Ubuntu LTS, 2 vCPU, 4 GB RAM, and at least 10 GB persistent EBS for seven days of snapshots/logging. Make sure the filesystem containing the project is persistent; a container layer alone is not a durable database.

Install Docker Engine and the Docker Compose plugin using the official Docker instructions for your Ubuntu version. Place the project directory on the instance and prepare the directories:

```bash
mkdir -p data logs
# The image runs as UID/GID 10001 so mounted SQLite/log directories must be writable.
sudo chown -R 10001:10001 data logs
cp .env.example .env
chmod 600 .env
```

Edit `.env` before startup:

- Confirm `ROOSTOO_API_URL=https://mock-api.roostoo.com` (unless event organizers issued a different host), and set the API key and secret.
- Confirm `EXPECTED_STARTING_CASH` matches the actual initial USD balance reported by the exchange.
- Verify `PAPER_MODE=false` and `ORDERS_ENABLED=true` only after read-only API checks and account permissions are confirmed.
- Ensure `BOT_DB_PATH=data/boss_v9_live.sqlite3` so the new dataset starts clean at the 60-second cadence.

`.env.example` ships in live mode because that is the requested deployment target. It deliberately contains blank API keys. The service should fail validation until real keys are set.

## 3. Build and launch

```bash
docker compose build
docker compose up -d

docker compose ps
docker compose logs --tail=200 -f boss
```

The container uses `restart: unless-stopped`. Data and logs are bind-mounted from the host. Stop the bot with `docker compose stop`; restart it with `docker compose start`. If updating the source, first inspect existing positions and unresolved orders, then rebuild/recreate deliberately.

## 4. Sampling and model warm-up

- `POLL_SECONDS=60`: saves a snapshot approximately every 60 seconds for every returned tradable USD market.
- `REBALANCE_SECONDS=120`: evaluates portfolio targets every two minutes.
- `HISTORY_LIMIT=2000`: limits recent signal histories loaded in each cycle (~33.3h at 60 seconds).
- `PREDICTOR_HISTORY_LIMIT=25000`: trains on the most recent ~17.4 days at the configured cadence, including a full 7-day run and warm-up history.
- `PREDICTION_RETRAIN_SECONDS=1800`: allows retraining every 30 minutes when enough valid rows are available.

The bot begins signal analysis after enough data exists for the fast features. The full predictor uses a longer feature window and prediction horizon and will not be ready immediately after a fresh start. This is expected; observations accumulate from the start. Do not seed V9 with the old 5-minute database because that would mix sampling cadences and invalidate fixed-time windows.

## 5. Intentionally unlimited bot-level constraints

The provided configuration does not impose a bot-level turnover limit, order-notional cap, rotation-count cap, daily-loss halt, or account drawdown halt:

```dotenv
MAX_DAILY_TURNOVER=0
MAX_ORDER_USD=0
ROTATION_MAX_PER_REBALANCE=0
DAILY_LOSS_LIMIT_PCT=0
MAX_DRAWDOWN_LIMIT_PCT=0
```

Zero means disabled/unlimited for these controls. Portfolio allocation, per-asset weights, strategy quality filters, per-position exits, cash/collateral requirements, exchange minimum notional/precision, and the exchange's own API limits still apply. These are distinct from the removed turnover/order/rotation/global halt controls.

## 6. Monitor for a full week

```bash
docker compose ps
docker compose logs --since=1h boss
du -h data logs
```

Ensure the instance clock is synced. Monitor container restarts, API errors, stale ticker warnings, order failures, short endpoint errors, disk usage, available USD/collateral, and exchange positions. At 60-second polling, the run produces approximately 10,080 snapshot timestamps over seven uninterrupted days; the actual number depends on outages and API availability.

Back up the SQLite database and logs without stopping the process if using a SQLite-safe backup method. Do not copy only the database file mid-write with an unsafe file-copy procedure. Keep persistent EBS snapshots as an additional recovery option.

## 7. Emergency operational commands

```bash
# Recent application output
docker compose logs --tail=300 boss

# Stop new activity but keep the container and data
docker compose stop

# Start after inspecting the account and logs
docker compose start
```

If an order request times out or the exchange response is ambiguous, the bot records an execution-uncertain state to avoid submitting a duplicate blindly. Check exchange order history, open orders, balances, and short positions first. Only then use:

```bash
python -m roostoo_bot --clear-execution-halt --confirm-reconciled
```

The confirmation flag is intentionally required because clearing a reconciliation state without checking the exchange can duplicate exposure. This is an execution-integrity check, not a strategy or turnover cap.

## 8. After the week

Stop new trades before analysis if that is what the competition schedule requires, and retain the SQLite DB, logs, configuration (with secrets removed), and exchange fills. Run a first pass with `python tools/walkforward_optimize.py --db data/boss_v9_live.sqlite3 --poll-seconds 60 --quick`; run the full grid on a sufficiently capable machine with the same arguments but without `--quick`. The tool requires at least 48 clean hourly decision windows after warm-up, reserves at least 12 later decisions as an untouched holdout, and reports whether the selected parameter set actually beats the default baseline out of sample. It prints proposed environment assignments, but review them rather than automatically overwriting `.env`. Include actual fill prices, taker fees, spread/slippage, short-open/close collateral flows, and funding/borrow charges if applicable. A one-week window is useful initial evidence, not proof that a parameter set is universally optimal.
