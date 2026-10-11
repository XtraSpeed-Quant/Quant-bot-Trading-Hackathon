# BOSS V9 Architecture

This file supersedes the earlier V6 architecture description.

## Runtime flow

1. Load `.env` into the validated `Config` dataclass.
2. Connect to the configured Roostoo host; load exchange metadata and market tickers.
3. Select all markets marked tradeable whose symbols end in `/USD` and appear in the ticker response.
4. Persist ticker snapshots to the clean `data/boss_v9_live.sqlite3` database at the configured 60-second polling interval.
5. Load a short rolling signal history and, when due/warm, a separate larger predictor training history.
6. Calculate market regime, cross-sectional momentum/relative-strength/trend/quality features, long targets and collateral-sized short targets.
7. Validate the ridge ranker on chronological, purged timestamps and gate directional classification separately.
8. Calculate spot sells and short closes first, then short opens and spot buys; account for cash, collateral, fees, exchange minimums/precision and API fills.
9. Reconcile exchange state, persist fills/equity/prediction labels and print the dashboard.
10. Docker Compose restarts a process that exits and bind-mounts SQLite/log data for durability.

## Sampling semantics

All indicator time windows and the predictor horizon are expressed in elapsed minutes, converted to sample bars via `POLL_SECONDS`. With a 60-second cadence, 60 minutes corresponds to 120 samples. The initial live DB is kept separate from the included source archive because older snapshots use a different sampling cadence.

## Portfolio and execution

The strategy targets long market value and short USD collateral. Both books consume aggregate portfolio budget; short exposure is not represented by a spot `SELL`. The exchange's V6 short endpoints are used. The runtime template disables bot-level turnover and order-notional caps, rotation-count limits, and optional global daily-loss/drawdown halts via zero values. Standard portfolio allocation, position-level exits, actual account collateral and exchange rules remain active.

An ambiguous live order response is reconciled before any subsequent orders can be submitted. This state prevents duplicate execution, and is cleared only after the operator confirms exchange order history and positions have been reconciled.

## Storage model

SQLite persists per-market price snapshots, model predictions and settled outcomes, trade/fill records, equity snapshots, API health events and engine state. `HISTORY_LIMIT` caps query result size for strategy calculations but does not prune the database. `PREDICTOR_HISTORY_LIMIT` determines how much history is passed into predictor training. For a one-week run, keep the database on persistent EBS and monitor disk usage.

## Validation and limitations

Unit tests exercise local model/portfolio/accounting paths. They do not certify a particular competition API instance, private-account permissions, execution slippage or profitability. Use chronological walk-forward evaluation with actual order fills and costs before claiming positive expected returns.
