# Source-link review for BOSS V9

## 1. Roostoo API documents

Link: https://github.com/roostoo/Roostoo-API-Documents

Inspected the current `README.md` and `PartnerDocument.md` through GitHub's repository connector.

Relevant details incorporated:
- Public REST API host: `https://mock-api.roostoo.com`.
- HMAC-SHA256 signature using sorted parameters, `RST-API-KEY`, `MSG-SIGNATURE`, and millisecond timestamp; POST uses `application/x-www-form-urlencoded`.
- `GET /v3/exchangeInfo`, `GET /v3/ticker` (one call without `pair` returns the ticker universe), and signed wallet/order endpoints.
- Short opening/closing/inventory are the `/v6/short_open`, `/v6/short_close`, and `/v6/short_positions` endpoints.
- Short opening is sized by USD collateral (not quantity), the opening fee is 0.1% of position value, short close executes at best ask, and short close does not accept a `price` parameter.
- API error responses may be HTTP 200, so callers must check `Success`.

Code consequence: fixed live short closing to use only supported endpoint arguments and added a regression test.

## 2. Luma event page

Link: https://luma.com/coghwiyt

Reviewed the event details. The event describes the venue as a real-time Roostoo mock exchange; the competition is not a generic public crypto-exchange deployment. Competition details recorded from the page:
- initial mock portfolio of $100,000;
- only 1× spot long/short trading, no leverage;
- taker commission 0.1%; maker commission 0.05%;
- prohibited: HFT, market-making and arbitrage; excessive API calls may trigger failure;
- net portfolio return is the primary ranking objective; risk-adjusted composite: `0.4 × Sortino + 0.3 × Sharpe + 0.3 × Calmar`;
- open-source repository and autonomous continuous runtime requirements.

Code consequence: poll all tickers once per minute, consider rebalances every two minutes, keep trading turnover unrestricted, set initial cash to $100,000, and align cost assumptions. Request pacing is to prevent bursts and is not a turnover/order-count cap.

## 3. Notion Data Sources Pack

Link: https://roostoo.notion.site/Data-Sources-Pack-318ba22fed7980118a69c7a614995930?source=copy_link

I retried the Notion URL and search via available tools, but the page remains unreachable from this environment. Its actual contents have not been guessed or represented as reviewed. If the pack lists mandatory data providers, API keys, fields, usage limitations or attribution requirements, those items still need to be checked against the bot.
