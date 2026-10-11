# BOSS V9 Competition AWS Release Manifest

## Included

- `roostoo_bot/`: command-line entry point and version metadata.
- `src/`: API client, configuration, DB, order planner, paper trader, predictor, risk manager, strategy, and logging modules.
- `tests/`: regression, API-host validation, cost-aware optimizer, EW-covariance, adaptive-exit, strategy, and empty-history dashboard tests.
- `.env.example`: live-mode configuration template with official documented Roostoo REST host and blank team-specific credentials.
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: container deployment.
- `README.md`, `DEPLOYMENT.md`, `BOSS_V8_CHANGELOG.md`, `V9.1_FIXLOG.md`, `FINAL_REVIEW.md`, `tools/walkforward_optimize.py` and `docs/`: architecture, strategy, research basis and release validation notes.
- `data/.gitkeep`, `logs/.gitkeep`: mountpoint placeholders.

## Deliberately excluded

- `.env`, API credentials, private keys, virtual environments, Python cache files, runtime logs, and the legacy `data/bot.sqlite3` snapshot.
- The active DB path is `data/boss_v9_live.sqlite3`; it is created on first run so the new 60-second cadence is not mixed with the old snapshot’s cadence.

## Key runtime defaults

- `PAPER_MODE=false`, `ORDERS_ENABLED=true`; credentials must be filled in before startup.
- `POLL_SECONDS=60`, `REBALANCE_SECONDS=120`, `EXCHANGE_INFO_REFRESH_SECONDS=900`, `ACCOUNT_REFRESH_SECONDS=120`.
- `HISTORY_LIMIT=2000`, `PREDICTOR_HISTORY_LIMIT=25000`, predictor retraining every 1800 seconds when enough valid samples exist.
- `MAX_ORDER_USD=0`, `MAX_DAILY_TURNOVER=0`, `ROTATION_MAX_PER_REBALANCE=0` (internal caps disabled).
- `DAILY_LOSS_LIMIT_PCT=0`, `MAX_DRAWDOWN_LIMIT_PCT=0` (optional global trade halts disabled).
- `PORTFOLIO_RISK_AVERSION=60`, `PORTFOLIO_RISK_HORIZON_MINUTES=60`, and covariance half-life 180 minutes; optimize on accumulated history before treating these defaults as tuned.
- `SHORTS_ENABLED=true`; requires the competition's short API permission and collateral support.

## Verification scope

Local unit tests, bytecode compilation, template-config validation, archive integrity, and clean-release inspection are covered. Live API trading and profitability have not been verified in this environment because private competition credentials and live fills were not provided.
