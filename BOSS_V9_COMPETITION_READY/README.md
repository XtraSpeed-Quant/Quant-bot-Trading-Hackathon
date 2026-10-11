# BOSS V9 — HK vs AU vs IN Quant Trading Hackathon bot

BOSS V9 is a competition-specific market-wide cross-sectional momentum/ranking bot for the HK vs AU vs IN Quant Trading Hackathon, running on Roostoo’s real-time mock exchange. It combines multi-horizon relative momentum, trend and breakout/breakdown features, market-quality checks, volatility-aware portfolio sizing, a regularized predictive ranker, collateral-backed shorts, and per-position exits. The deployment profile is set up for a **continuous seven-day AWS run** that trades while saving the market tape to SQLite for model training and later evaluation. The event rules favor net portfolio return and also calculate a risk-adjusted composite of 0.4 Sortino + 0.3 Sharpe + 0.3 Calmar; they prohibit HFT, market-making, arbitrage, leverage and excessive API calls.

> **Live-mode configuration:** `.env.example` is a live-mode template (`PAPER_MODE=false`, `ORDERS_ENABLED=true`). Do not start trading until the API host and account permissions are confirmed and the two API keys have been added to a private `.env` file. No credentials are included in this package.

## V9.1 maintenance fix

The empty-history `--inspect` dashboard fix and regression tests are documented in [V9.1_FIXLOG.md](V9.1_FIXLOG.md).

## Competition-specific changes in V9

- **60-second sampling / 120-second rebalancing.** Strategy feature windows are configured in elapsed minutes and converted to bars from `POLL_SECONDS`; a 1-hour window is 60 observations at 60-second sampling.
- **Full tradable USD universe.** Each scan uses all exchange-information pairs marked tradable that are present in the ticker response. `ALLOW_PAIRS` can optionally narrow that universe; leave it blank to scan all pairs. `DIAGNOSTIC_TOP_N` only limits how many candidate diagnostics are printed, not how many assets are evaluated. `TOP_N=20` and `SHORT_TOP_N=20` select the candidate list per side.
- **Long and short portfolios.** Short positions use Roostoo's short endpoint and collateral accounting. Short-open/close/position endpoints are configured in the API client. The official REST base URL listed in Roostoo’s API repository is `https://mock-api.roostoo.com`; the hackathon event explicitly uses Roostoo’s real-time mock exchange backend. API keys remain participant-specific.
- **Mathematical portfolio sizing.** Targets solve an explicit mean-variance utility with signed long/short weights, covariance shrinkage, marginal trading costs and current exposure in the objective. The tool also runs gap-aware chronological walk-forward parameter selection, requires at least 48 hourly decisions after warm-up, and keeps at least 12 later decisions untouched for OOS reporting.
- **Volatility-adaptive exits.** Hard stops, trailing activation/distances and take-profit thresholds scale to realized hourly volatility, with fee/spread/slippage floors; the research replay evaluates exits on the intervening sampled price path.
- **No internal turnover cap.** `MAX_DAILY_TURNOVER=0` is unlimited. `MAX_ORDER_USD=0` disables the bot's per-order notional cap. `ROTATION_MAX_PER_REBALANCE=0` disables the rotation-count cap. Exchange-level precision, minimum order rules, API permissions, and available collateral remain applicable.
- **No forced account-level stop in the sample configuration.** `DAILY_LOSS_LIMIT_PCT=0` and `MAX_DRAWDOWN_LIMIT_PCT=0` disable those optional portfolio-wide trade halts. Per-position stop, trailing-stop, and take-profit logic still operates.
- **Full-week model data.** Main signal windows use the recent `HISTORY_LIMIT`; the predictor trains from `PREDICTOR_HISTORY_LIMIT`. SQLite history is append-only and is not deleted when the in-memory query window is smaller.
- **Separate live database.** V9 defaults to `data/boss_v9_live.sqlite3`. The older bundled snapshot database is deliberately excluded so older historical rows do not get mixed into the new 60-second live time series.
- **Predictor validation is chronological and cross-sectional.** The ranker is checked by timestamp and estimated fees/slippage are deducted from the top-decile validation return. Weak classifiers remain inactive instead of being assumed predictive.
- **AWS process recovery.** Docker Compose restarts the container after process exit and persists `data/` and `logs/` through bind mounts.

## Install and configure

Python 3.12 is recommended. On a local machine:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and fill `ROOSTOO_API_KEY` and `ROOSTOO_API_SECRET`. `ROOSTOO_API_URL` is prefilled as `https://mock-api.roostoo.com`, the REST host listed in the official Roostoo API docs and used by the event’s real-time mock exchange. If the organizers gave your team a different dedicated host, use that exact host instead. Keep `.env` private and out of version control.

Before live mode, verify connectivity and permission:

```bash
python -m roostoo_bot --check-api
python -m roostoo_bot --inspect
python -m unittest discover -s tests -v
```

`--check-api` checks public server time and exchange information; it does not submit orders. `--inspect` displays signals and attempts a short-position endpoint read in live mode; it does not submit orders. To test execution without exchange orders, set `PAPER_MODE=true` in `.env`; paper positions are local to the process and reset on restart.

## Seven-day AWS profile

The event says the initial portfolio is $100,000, spot longs/shorts are 1× only, each taker order incurs 0.1% commission and each maker order 0.05%. The current defaults are configured for continuous directional trading and data collection while avoiding a high-frequency request pattern:

| Setting | Default | Meaning |
|---|---:|---|
| Poll | 60 s | Samples all currently tradable USD pairs returned by the API |
| Rebalance | 120 s | Recomputes portfolio targets every two minutes |
| Signal history | 2,000 samples | About 33.3 hours at 60-second polling for active signal features |
| Predictor history | 25,000 samples | About 17.4 days at 60-second polling; enough room for one week plus warm-up |
| Predictor retrain | 1,800 s | Retrains at most every 30 minutes when enough valid observations are present |
| `MAX_DAILY_TURNOVER` | `0` | No internal daily turnover cap |
| `MAX_ORDER_USD` | `0` | No internal per-order notional cap |
| `ROTATION_MAX_PER_REBALANCE` | `0` | No internal rotation count cap |
| Account-level loss / drawdown halts | `0` / `0` | Disabled; per-position exits are still enabled |

At a 60-second interval, one week gives roughly 10,080 market timestamps before missed requests and uptime gaps. Exact usable ML sample counts will differ because the model needs a feature lookback and a future-label horizon, and it only trains on timestamps where features and targets are valid. New observations are collected even before the model is warm enough to train.

A reasonable AWS starting size is 2 vCPU and 4 GB RAM, with at least 10 GB of persistent EBS storage. Use a persistent volume for the project `data/` and `logs/` directories. Store credentials in the instance's secret-management mechanism or a restricted `.env` file; do not bake them into a Docker image or commit them.

For Docker deployment, follow [DEPLOYMENT.md](DEPLOYMENT.md).

## Portfolio and model design

The portfolio is cross-sectional: it compares eligible assets rather than only applying one pair's moving-average crossover. Long selection uses relative strength, multi-horizon returns, trend/breakout and quality filters. The short side ranks weak assets, applies separate short-side filters, and sizes exposure as USD collateral. Target weights are solved by the signed mean-variance objective described in [docs/strategy.md](docs/strategy.md), including covariance risk and the estimated cost of changing from current weights.

The baseline uses regularized linear/ridge models rather than a high-capacity model that can overfit a single week's noisy cryptocurrency tape. Research on crypto cross-sectional returns highlights momentum, past returns, liquidity, and price-related characteristics; newer minute-level research also stresses spread/realized-volatility features, realistic costs, and strict time-ordered validation. References are linked in [docs/research_basis.md](docs/research_basis.md).

**Parameter honesty:** these are researched starting parameters, not a mathematical proof of optimality. A single live week is useful for collecting fresh observations and evaluating drift; use chronological walk-forward holdouts and include fees, slippage, short collateral rules, and actual fill prices before changing thresholds based on P&L.

## Data and operations

- `data/boss_v9_live.sqlite3` stores the new run's price history, states, trade log, and model observations. Do not delete it during the seven-day run.
- `logs/bot.log` records console output and execution diagnostics.
- `HISTORY_LIMIT` limits how much history is loaded for ordinary signal calculations; it does **not** delete old SQLite rows.
- `PREDICTOR_HISTORY_LIMIT` is the training input size. Keep it large enough for a full-week sampling window.
- If the process stops, Compose's restart policy restarts it. Inspect logs and account positions after any unexpected restart.
- The order reconciliation halt is not a strategy limiter: it prevents duplicate orders after an ambiguous exchange response. Resolve the order against exchange order history, balances, and short positions before clearing it.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q roostoo_bot src tools tests
```

Tests and compilation verify the local code paths only. The release has not been tested with your private competition credentials and no profitability claim is made before actual fills and walk-forward evaluation.
