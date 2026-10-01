# Roostoo Quant Trading Bot V4

A real-time quantitative trading engine for the Roostoo mock exchange. The implementation is rule-based with a risk-gated predictive overlay and optional short support.

## Strategy architecture

- multi-horizon momentum: 1h, 4h, 12h and 24h
- trend confirmation and breakout context
- cross-sectional relative strength
- trend-preserving pullback detection
- short-term overextension penalty
- liquidity and spread filters
- correlation-aware portfolio selection
- volatility-aware sizing
- continuous market-regime exposure scaling
- predictive ensemble: ridge forward-return model + three-state ridge classifier
- purged time-ordered validation
- online evaluation of predictions against later market outcomes
- optional short-position sleeve through Roostoo's documented `/v6/short_open`, `/v6/short_close`, and `/v6/short_positions` endpoints

No trading model is risk-free. The predictive layer is gated and is not permitted to influence decisions unless validation criteria are met.

## Competition alignment

The supplied competition brief requires autonomous Roostoo GET/POST trading, prohibits high-frequency trading, market making and arbitrage, permits spot trading including shorting, uses a $100,000 mock portfolio, and charges 0.1% taker and 0.05% maker fees. Final evaluation uses portfolio return, Sortino, Sharpe and Calmar, followed by code/strategy review.

This bot uses a 60-second observation cycle and 15-minute rebalancing cycle, avoiding high-frequency behavior.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your API credentials only in `.env`.

### Public API check

```powershell
python -m roostoo_bot --check-api
```

### Authenticated account check

```powershell
python -m roostoo_bot --account-check
```

This does not place orders.

### Live inspection without orders

Keep `ORDERS_ENABLED=false` and run:

```powershell
python -m roostoo_bot --inspect
```

### Real-time engine

```powershell
python -m roostoo_bot
```

### Controlled test-account execution

This step can submit real mock-exchange orders to the account in `.env`. Use a separate namespace and conservative limits for the first test:

```env
BOT_NAMESPACE=test-live
ORDERS_ENABLED=true
ENABLE_SHORTS=false
TOP_N=1
MAX_GROSS_TARGET=0.10
MAX_ASSET_WEIGHT=0.10
MAX_ORDER_USD=250
MAX_DAILY_TURNOVER=0.02
```

Then run one cycle:

```powershell
python -m roostoo_bot --once
```

Inspect the resulting order/fill logs and account balance before allowing a continuous process.

### Competition execution

Set the organizer-provided API URL and competition credentials in `.env`, use a new `BOT_NAMESPACE`, verify account allocation and permissions, and only then set `ORDERS_ENABLED=true` for the AWS deployment.

For a separate competition API, change `ROOSTOO_API_URL` and use a new `BOT_NAMESPACE` so account risk state is not mixed between environments. Keep the competition credentials only in the local `.env`.

## Data and model monitoring

The bot stores live market observations in SQLite at `BOT_DB_PATH`. It also stores predictive calls and settles them against later prices, allowing live measurement of prediction accuracy and correlation over time.

The terminal dashboard shows the model's current reasoning, target weights, proposed buys/sells, risk state, and predictive validation metrics.
