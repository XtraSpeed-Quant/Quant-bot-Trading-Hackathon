from __future__ import annotations

import logging
import math
import time
from typing import Any, Mapping

import requests

from .client import RoostooClient, RoostooError
from .config import Config
from .risk import RiskDecision, RiskManager
from .store import Store
from .strategy import Market, MultiFactorStrategy, StrategyResult


class TradingEngine:
    def __init__(self, config: Config, client: RoostooClient, store: Store) -> None:
        self.config, self.client, self.store = config, client, store
        self.log = logging.getLogger(self.__class__.__name__)
        self.ns = config.bot_namespace
        self.strategy = MultiFactorStrategy(
            fast_window=config.fast_window, medium_window=config.medium_window, long_window=config.long_window,
            trend_window=config.trend_window, vol_window=config.vol_window, top_n=config.top_n,
            max_gross_target=config.max_gross_target, max_asset_weight=config.max_asset_weight,
            max_short_gross=config.max_short_gross, max_short_weight=config.max_short_weight,
            min_score=config.min_score, min_relative_edge=config.min_relative_edge,
            min_1h_return=config.min_1h_return, min_4h_return=config.min_4h_return,
            min_12h_return=config.min_12h_return, min_quote_volume=config.min_quote_volume,
            max_spread_bps=config.max_spread_bps, volatility_floor=config.volatility_floor,
            predictor_enabled=config.predictor_enabled, prediction_horizon=config.prediction_horizon,
            predictor_feature_window=config.predictor_feature_window, predictor_min_samples=config.predictor_min_samples,
            predictor_max_samples=config.predictor_max_samples, predictor_ridge_alpha=config.predictor_ridge_alpha,
            predictor_classifier_alpha=config.predictor_classifier_alpha,
            predictor_validation_fraction=config.predictor_validation_fraction, predictor_purge_gap=config.predictor_purge_gap,
            predictor_min_directional_accuracy=config.predictor_min_directional_accuracy,
            predictor_min_correlation=config.predictor_min_correlation,
            predictor_min_balanced_accuracy=config.predictor_min_balanced_accuracy,
            predictor_min_top_decile_edge=config.predictor_min_top_decile_edge,
            predictor_class_threshold=config.predictor_class_threshold, predictor_max_brier=config.predictor_max_brier, min_prediction_edge=config.min_prediction_edge,
            prediction_retrain_seconds=config.prediction_retrain_seconds,
            correlation_window=config.correlation_window, max_pair_correlation=config.max_pair_correlation,
            correlation_penalty=config.correlation_penalty, overextension_penalty=config.overextension_penalty,
            breakout_window=config.breakout_window, taker_fee_assumption=config.taker_fee_assumption,
            enable_shorts=config.enable_shorts,
        )
        self.risk = RiskManager(config, store)
        self.exchange_info: dict[str, Any] | None = None
        self.last_rebalance_ms = int(store.get_state(store.key(self.ns, "last_rebalance_ms"), 0) or 0)
        self.last_time_sync_ms = 0
        self.last_prediction_log_ms = int(store.get_state(store.key(self.ns, "last_prediction_log_ms"), 0) or 0)
        self.order_in_flight = False

    def _state_key(self, name: str) -> str:
        return self.store.key(self.ns, name)

    def _ensure_exchange_info(self) -> dict[str, Any]:
        now = int(time.time() * 1000)
        if self.exchange_info is None:
            self.client.sync_time()
            self.exchange_info = self.client.exchange_info()
            self.last_time_sync_ms = now
        elif now - self.last_time_sync_ms > 300_000:
            self.client.sync_time()
            self.last_time_sync_ms = now
        return self.exchange_info

    def _markets(self, tickers: Mapping[str, Any]) -> dict[str, Market]:
        info = self._ensure_exchange_info()
        pairs = info.get("TradePairs", {})
        result: dict[str, Market] = {}
        for pair, ticker in tickers.items():
            meta = pairs.get(pair)
            if not isinstance(meta, Mapping):
                continue
            try:
                market = Market(
                    pair=pair,
                    coin=str(meta.get("Coin", pair.split("/")[0])),
                    unit=str(meta.get("Unit", pair.split("/")[1])).upper(),
                    price=float(ticker["LastPrice"]), bid=float(ticker["MaxBid"]), ask=float(ticker["MinAsk"]),
                    change_24h=float(ticker.get("Change", 0.0)), quote_volume_24h=float(ticker.get("UnitTradeValue", 0.0)),
                    amount_precision=int(meta.get("AmountPrecision", 8)), price_precision=int(meta.get("PricePrecision", 8)),
                    min_order=float(meta.get("MiniOrder", 1.0)), can_trade=bool(meta.get("CanTrade", False)),
                )
            except (TypeError, ValueError, KeyError, IndexError):
                continue
            if market.spread_bps <= 0 or not math.isfinite(market.spread_bps):
                continue
            result[pair] = market
        return result

    def _strategy_markets(self, markets: Mapping[str, Market]) -> dict[str, Market]:
        allow = set(self.config.allow_pairs)
        return {p: m for p, m in markets.items() if m.can_trade and m.unit == "USD" and (not allow or p in allow)}

    @staticmethod
    def _wallet_amount(wallet: Mapping[str, Mapping[str, float]], asset: str, free_only: bool = False) -> float:
        item = wallet.get(asset, {})
        free, locked = float(item.get("Free", 0.0)), float(item.get("Lock", 0.0))
        return free if free_only else free + locked

    def _short_map(self) -> dict[str, dict[str, float]]:
        try:
            positions = self.client.short_positions()
        except (RoostooError, requests.RequestException):
            return {}
        return {str(p.get("Pair")): {"qty": float(p.get("ShortQty", 0.0)), "collateral": float(p.get("Collateral", 0.0)), "pnl": float(p.get("UnrealizedPNL", 0.0))} for p in positions}

    def equity_and_holdings(self, wallet: Mapping[str, Mapping[str, float]], markets: Mapping[str, Market], shorts: Mapping[str, Mapping[str, float]] | None = None) -> tuple[float, dict[str, float]]:
        equity = self._wallet_amount(wallet, "USD")
        holdings: dict[str, float] = {}
        seen: set[str] = set()
        for pair, market in markets.items():
            if market.coin in seen:
                continue
            seen.add(market.coin)
            qty = self._wallet_amount(wallet, market.coin)
            if qty > 0:
                value = qty * market.price
                if value > 0:
                    holdings[pair] = value
                    equity += value
        for pair, pos in (shorts or {}).items():
            equity += float(pos.get("pnl", 0.0))
        return equity, holdings

    @staticmethod
    def _floor_quantity(value: float, precision: int) -> float:
        scale = 10 ** max(precision, 0)
        return math.floor(value * scale + 1e-12) / scale

    def _current_values(self, wallet: Mapping[str, Mapping[str, float]], markets: Mapping[str, Market]) -> dict[str, float]:
        return {pair: self._wallet_amount(wallet, market.coin, free_only=True) * market.price for pair, market in markets.items()}

    def _order_plan(self, long_targets: Mapping[str, float], short_targets: Mapping[str, float], markets: Mapping[str, Market], wallet: Mapping[str, Mapping[str, float]], shorts: Mapping[str, Mapping[str, float]], equity: float, liquidate: bool = False, timestamp_ms: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        current = self._current_values(wallet, markets)
        target_values = {} if liquidate else {pair: equity * float(w) for pair, w in long_targets.items()}
        threshold = max(self.config.min_order_usd, equity * self.config.rebalance_band)
        sells = []
        buys = []
        hold_ms = self.config.min_hold_minutes * 60_000
        for pair, cur in current.items():
            tgt = target_values.get(pair, 0.0)
            gap = cur - tgt
            last_trade = self.store.recent_trade_timestamp(pair) if not liquidate else None
            recently_traded = (
                timestamp_ms is not None
                and last_trade is not None
                and timestamp_ms - last_trade < hold_ms
            )
            if gap >= threshold and not recently_traded:
                sells.append({"pair": pair, "current": cur, "target": tgt, "notional": gap})
        for pair, tgt in target_values.items():
            cur = current.get(pair, 0.0)
            gap = tgt - cur
            last_trade = self.store.recent_trade_timestamp(pair)
            recently_traded = (
                timestamp_ms is not None
                and last_trade is not None
                and timestamp_ms - last_trade < hold_ms
            )
            if gap >= threshold and not recently_traded:
                buys.append({"pair": pair, "current": cur, "target": tgt, "notional": gap})
        short_opens, short_closes = [], []
        for pair, pos in shorts.items():
            cur_collateral = float(pos.get("collateral", 0.0))
            target = 0.0 if liquidate else equity * float(short_targets.get(pair, 0.0))
            if cur_collateral - target >= threshold:
                short_closes.append({"pair": pair, "collateral": cur_collateral, "target": target})
        for pair, target_weight in short_targets.items():
            target = equity * float(target_weight)
            cur = float(shorts.get(pair, {}).get("collateral", 0.0))
            if target - cur >= threshold:
                short_opens.append({"pair": pair, "collateral": target - cur, "current": cur, "target": target})
        sells.sort(key=lambda x: x["notional"], reverse=True)
        buys.sort(key=lambda x: x["notional"], reverse=True)
        short_opens.sort(key=lambda x: x["collateral"], reverse=True)
        return sells, buys, short_opens, short_closes

    def _settle_order(self, detail: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(detail)
        order_id = out.get("OrderID")
        if order_id is None:
            return out
        for _ in range(self.config.order_settle_attempts):
            status = str(out.get("Status", "UNKNOWN")).upper()
            if status in {"FILLED", "CANCELED", "CANCELLED", "REJECTED"}:
                break
            time.sleep(self.config.order_settle_delay_seconds)
            try:
                rows = self.client.query_order(order_id=order_id)
            except (RoostooError, requests.RequestException):
                continue
            if rows:
                out = rows[0]
        return out

    def _query_recent_order(self, pair: str, side: str, submitted_at_ms: int) -> dict[str, Any] | None:
        try:
            rows = self.client.query_order(pair=pair, pending_only=False, limit=20)
        except (RoostooError, requests.RequestException):
            return None
        cutoff = submitted_at_ms - self.config.order_reconcile_window_seconds * 1000
        candidates = [r for r in rows if str(r.get("Side", "")).upper() == side.upper() and int(r.get("CreateTimestamp", 0) or 0) >= cutoff]
        return max(candidates, key=lambda x: int(x.get("CreateTimestamp", 0) or 0)) if candidates else None

    def _execute_market_order(self, market: Market, side: str, target_notional: float, wallet: Mapping[str, Mapping[str, float]], timestamp_ms: int) -> tuple[float, bool, str]:
        price = market.ask if side == "BUY" else market.bid
        if target_notional < self.config.min_order_usd or price <= 0:
            return 0.0, False, "below_minimum_or_invalid_price"
        notional = min(target_notional, self.config.max_order_usd)
        if side == "BUY":
            cash = self._wallet_amount(wallet, "USD", free_only=True)
            notional = min(notional, cash / (1.0 + self.config.taker_fee_assumption))
        else:
            qty = self._wallet_amount(wallet, market.coin, free_only=True)
            notional = min(notional, qty * price)
        qty = self._floor_quantity(notional / price, market.amount_precision)
        notional = qty * price
        if qty <= 0 or notional < max(self.config.min_order_usd, market.min_order):
            return 0.0, False, "below_exchange_minimum"
        self.order_in_flight = True
        try:
            submitted = int(time.time() * 1000)
            try:
                response = self.client.place_market_order(market.pair, side, f"{qty:.{market.amount_precision}f}")
                detail = response.get("OrderDetail", {})
            except (RoostooError, requests.RequestException) as exc:
                detail = self._query_recent_order(market.pair, side, submitted)
                if detail is None:
                    self.store.record_api_event(timestamp_ms, "/v3/place_order", False, f"unknown execution {exc}")
                    return 0.0, False, "unknown_execution"
            detail = self._settle_order(detail)
            status = str(detail.get("Status", "UNKNOWN")).upper()
            fill_qty = float(detail.get("FilledQuantity", 0.0) or 0.0)
            fill_price = float(detail.get("FilledAverPrice", detail.get("Price", price)) or price)
            fee = float(detail.get("CommissionChargeValue", fill_qty * fill_price * self.config.taker_fee_assumption) or 0.0)
            if fill_qty > 0 and fill_price > 0:
                self.store.record_trade(timestamp_ms, market.pair, side, fill_qty, fill_price, fee, "live", str(detail.get("OrderID", "")) or None, status)
            self.store.record_api_event(timestamp_ms, "/v3/place_order", status not in {"REJECTED", "CANCELED", "CANCELLED", "UNKNOWN"}, f"pair={market.pair} side={side} status={status}")
            return fill_qty * fill_price, status == "FILLED" and fill_qty > 0, status.lower()
        finally:
            self.order_in_flight = False

    def _execute_short_open(self, pair: str, collateral: float, timestamp_ms: int) -> tuple[float, bool, str]:
        if collateral < self.config.min_order_usd:
            return 0.0, False, "below_minimum"
        collateral = min(collateral, self.config.max_order_usd)
        self.order_in_flight = True
        try:
            response = self.client.short_open(pair, f"{collateral:.2f}")
            status = str(response.get("Status", "UNKNOWN")).upper()
            if status not in {"OPEN", "PENDING"}:
                return 0.0, False, status.lower()
            order_id = response.get("ID")
            entry = float(response.get("EntryPrice", 0.0) or 0.0)
            qty = float(response.get("ShortQty", 0.0) or 0.0)
            fee = float(response.get("OpenFee", 0.0) or 0.0)
            self.store.record_trade(timestamp_ms, pair, "SHORT_OPEN", qty, entry, fee, "live", str(order_id) if order_id is not None else None, status)
            return collateral, status == "OPEN", status.lower()
        except (RoostooError, requests.RequestException) as exc:
            self.store.record_api_event(timestamp_ms, "/v6/short_open", False, str(exc))
            return 0.0, False, "short_open_failed"
        finally:
            self.order_in_flight = False

    def _execute_short_close(self, pair: str, collateral: float, qty: float, timestamp_ms: int) -> tuple[float, bool, str]:
        if qty <= 0:
            return 0.0, False, "no_position"
        self.order_in_flight = True
        try:
            response = self.client.short_close(pair, close_qty=f"{qty:.8f}")
            closed = float(response.get("ClosedQty", qty) or qty)
            price = float(response.get("ClosePrice", 0.0) or 0.0)
            fee = float(response.get("CloseFee", 0.0) or 0.0)
            self.store.record_trade(timestamp_ms, pair, "SHORT_CLOSE", closed, price, fee, "live", None, "FILLED")
            return collateral, True, "filled"
        except (RoostooError, requests.RequestException) as exc:
            self.store.record_api_event(timestamp_ms, "/v6/short_close", False, str(exc))
            return 0.0, False, "short_close_failed"
        finally:
            self.order_in_flight = False

    def _rebalance(self, strategy_result: StrategyResult, markets: Mapping[str, Market], wallet: Mapping[str, Mapping[str, float]], shorts: Mapping[str, Mapping[str, float]], equity: float, decision: RiskDecision, timestamp_ms: int) -> dict[str, Any]:
        sells, buys, short_opens, short_closes = self._order_plan(
            strategy_result.long_targets, strategy_result.short_targets, markets, wallet, shorts, equity,
            decision.liquidate, timestamp_ms,
        )
        result = {"orders": 0, "turnover": 0.0, "blocked": "", "planned_sells": sells, "planned_buys": buys, "planned_short_opens": short_opens, "planned_short_closes": short_closes}
        if self.order_in_flight:
            result["blocked"] = "order_in_flight"
            return result
        try:
            pending = self.client.pending_count()
        except (RoostooError, requests.RequestException) as exc:
            result["blocked"] = "pending_check_failed"
            self.store.record_api_event(timestamp_ms, "/v3/pending_count", False, str(exc))
            return result
        if pending > 0:
            result["blocked"] = "pending_orders"
            return result
        if not self.config.orders_enabled:
            result["blocked"] = "orders_disabled"
            return result

        remaining = max(self.config.max_daily_turnover * equity - decision.turnover_used, 0.0)
        if decision.liquidate:
            remaining = max(remaining, sum(x["notional"] for x in sells))

        updated_wallet = wallet
        for plan in sells:
            market = markets.get(plan["pair"])
            if market is None or remaining < self.config.min_order_usd:
                continue
            traded, filled, status = self._execute_market_order(market, "SELL", min(plan["notional"], remaining), updated_wallet, timestamp_ms)
            if traded > 0:
                result["orders"] += int(filled)
                result["turnover"] += traded
                remaining = max(0.0, remaining - traded)
                if filled:
                    updated_wallet = self.client.balance()
            if status == "unknown_execution":
                result["blocked"] = "unknown_execution"
                return result

        if decision.liquidate:
            for pair, pos in shorts.items():
                market = markets.get(pair)
                if not market:
                    continue
                _, ok, status = self._execute_short_close(pair, float(pos.get("collateral", 0.0)), float(pos.get("qty", 0.0)), timestamp_ms)
                result["orders"] += int(ok)
                if not ok and status not in {"no_position"}:
                    result["blocked"] = status
                    return result
            result["blocked"] = decision.reason
            return result

        updated_wallet = self.client.balance()
        for plan in buys:
            if remaining < self.config.min_order_usd:
                break
            market = markets.get(plan["pair"])
            if market is None:
                continue
            traded, filled, status = self._execute_market_order(market, "BUY", min(plan["notional"], remaining), updated_wallet, timestamp_ms)
            if traded > 0:
                result["orders"] += int(filled)
                result["turnover"] += traded
                remaining = max(0.0, remaining - traded)
                if filled:
                    updated_wallet = self.client.balance()
            if status == "unknown_execution":
                result["blocked"] = "unknown_execution"
                return result

        if self.config.enable_shorts:
            for plan in short_closes:
                pos = shorts.get(plan["pair"], {})
                _, ok, status = self._execute_short_close(plan["pair"], float(pos.get("collateral", 0.0)), float(pos.get("qty", 0.0)), timestamp_ms)
                result["orders"] += int(ok)
                if ok:
                    result["turnover"] += float(plan["collateral"])
            for plan in short_opens:
                if remaining < self.config.min_order_usd:
                    break
                collateral = min(float(plan["collateral"]), remaining, self.config.max_order_usd)
                traded, ok, status = self._execute_short_open(plan["pair"], collateral, timestamp_ms)
                if traded > 0:
                    result["orders"] += int(ok)
                    result["turnover"] += traded
                    remaining = max(0.0, remaining - traded)
                if not ok and status == "unknown_execution":
                    result["blocked"] = status
                    return result
        return result

    def cycle(self, execute: bool = True) -> dict[str, Any]:
        self._ensure_exchange_info()
        raw = self.client.tickers()
        tickers = raw.get("Data", {})
        if not isinstance(tickers, Mapping) or not tickers:
            raise RuntimeError("Roostoo returned no ticker data")
        server_timestamp = int(raw.get("ServerTime", self.client._timestamp()))
        local_timestamp = int(time.time() * 1000)
        if abs(local_timestamp - server_timestamp) > self.config.stale_data_seconds * 1000:
            raise RuntimeError(
                f"stale exchange timestamp: local={local_timestamp} server={server_timestamp}"
            )
        markets = self._markets(tickers)
        strategy_markets = self._strategy_markets(markets)
        if not strategy_markets:
            raise RuntimeError("No tradable USD markets found")
        newest = server_timestamp
        self.store.add_market_snapshots(server_timestamp, [{
            "pair": p, "price": m.price, "bid": m.bid, "ask": m.ask,
            "change_24h": m.change_24h, "quote_volume_24h": m.quote_volume_24h, "spread_bps": m.spread_bps,
        } for p, m in strategy_markets.items()])
        self.store.set_state(self._state_key("last_market_timestamp_ms"), newest)
        wallet = self.client.balance()
        shorts = self._short_map() if self.config.enable_shorts else {}
        equity, holdings = self.equity_and_holdings(wallet, markets, shorts)
        self.store.record_equity(server_timestamp, equity)
        self.store.set_state(self._state_key("last_equity"), equity)
        decision = self.risk.assess(server_timestamp, equity)

        histories = self.store.histories(list(strategy_markets), self.config.history_limit)
        history_points = min((len(v) for v in histories.values()), default=0)
        required = self.strategy.required_history
        due = server_timestamp - self.last_rebalance_ms >= self.config.rebalance_seconds * 1000

        strategy_result = StrategyResult(
            long_targets={}, short_targets={}, signals=[], regime="warming_up",
            breadth=0.0, median_4h=0.0, median_12h=0.0, median_24h=0.0,
            market_strength=0.0, gross_target=0.0, short_gross_target=0.0,
            median_volatility=0.0, volatility_scale=0.0, predictor_enabled=False,
            predictor_samples=0, predictor_directional_accuracy=0.0,
            predictor_correlation=0.0, predictor_balanced_accuracy=0.0,
            predictor_top_decile_edge=0.0, predictor_brier=1.0,
        )
        result: dict[str, Any] = {"orders": 0, "turnover": 0.0, "blocked": "", "planned_sells": [], "planned_buys": [], "planned_short_opens": [], "planned_short_closes": []}
        status = "warming_up"

        current_prices = {p: m.price for p, m in strategy_markets.items()}
        self.store.settle_predictions(server_timestamp, current_prices, self.config.predictor_class_threshold)
        live_metrics = self.store.prediction_metrics()

        if history_points >= required:
            strategy_result = self.strategy.analyze(histories, strategy_markets, now_ms=server_timestamp)
            status = "running"
            predictor_result = self.strategy.last_predictor_result
            should_log_predictions = (
                predictor_result is not None
                and (
                    self.last_prediction_log_ms == 0
                    or server_timestamp - self.last_prediction_log_ms
                    >= self.config.prediction_log_interval_seconds * 1000
                )
            )
            if should_log_predictions:
                ranked_predictions = sorted(
                    predictor_result.predictions.items(),
                    key=lambda item: (abs(item[1].predicted_return), item[1].confidence),
                    reverse=True,
                )[: self.config.prediction_log_top_n]
                for pair, prediction in ranked_predictions:
                    market = strategy_markets.get(pair)
                    if market is None:
                        continue
                    self.store.record_prediction(
                        server_timestamp, pair, market.price,
                        server_timestamp + self.config.prediction_horizon * 60_000,
                        prediction.predicted_return, prediction.probability_up,
                        prediction.probability_flat, prediction.probability_down,
                        prediction.confidence, strategy_result.predictor_enabled,
                    )
                self.last_prediction_log_ms = server_timestamp
                self.store.set_state(
                    self._state_key("last_prediction_log_ms"),
                    self.last_prediction_log_ms,
                )

            if not strategy_result.signals:
                pass
            elif not execute:
                sells, buys, short_opens, short_closes = self._order_plan(
                    strategy_result.long_targets, strategy_result.short_targets, strategy_markets, wallet, shorts,
                    equity, decision.liquidate, server_timestamp,
                )
                result = {"orders": 0, "turnover": 0.0, "blocked": "inspection_only", "planned_sells": sells, "planned_buys": buys, "planned_short_opens": short_opens, "planned_short_closes": short_closes}
            elif due or decision.liquidate:
                result = self._rebalance(strategy_result, strategy_markets, wallet, shorts, equity, decision, server_timestamp)
                self.last_rebalance_ms = server_timestamp
                self.store.set_state(self._state_key("last_rebalance_ms"), self.last_rebalance_ms)

        today_trades = self.store.daily_trade_counts().get(self.risk.day_key(server_timestamp), 0)
        exposure = sum(holdings.values()) + sum(float(p.get("collateral", 0.0)) for p in shorts.values())
        return {
            "status": status, "timestamp_ms": server_timestamp, "equity": equity,
            "orders": result["orders"], "turnover": result["turnover"], "blocked": result.get("blocked", ""),
            "regime": strategy_result.regime, "breadth": strategy_result.breadth, "median_4h": strategy_result.median_4h,
            "median_12h": strategy_result.median_12h, "median_24h": strategy_result.median_24h,
            "market_strength": strategy_result.market_strength, "gross_target": strategy_result.gross_target,
            "short_gross_target": strategy_result.short_gross_target, "signals": len(strategy_result.signals),
            "signal_rows": [{
                "pair": s.pair, "direction": s.direction, "score": s.score, "fast_return": s.fast_return,
                "medium_return": s.medium_return, "long_return": s.long_return, "return_24h": s.return_24h,
                "trend": s.trend, "predicted_return": s.predicted_return, "probability_up": s.probability_up,
                "probability_down": s.probability_down, "prediction_confidence": s.prediction_confidence,
                "target_weight": s.target_weight, "reason": s.reason,
            } for s in strategy_result.signals],
            "holdings": holdings, "current_exposure": exposure, "risk_reason": decision.reason,
            "daily_pnl_pct": decision.daily_pnl_pct, "drawdown_pct": decision.drawdown_pct,
            "prediction_enabled": strategy_result.predictor_enabled, "prediction_samples": strategy_result.predictor_samples,
            "prediction_accuracy": strategy_result.predictor_directional_accuracy, "prediction_correlation": strategy_result.predictor_correlation,
            "prediction_balanced_accuracy": strategy_result.predictor_balanced_accuracy, "prediction_top_decile_edge": strategy_result.predictor_top_decile_edge,
            "prediction_brier": strategy_result.predictor_brier, "live_prediction_samples": live_metrics["samples"],
            "live_prediction_accuracy": live_metrics["directional_accuracy"], "live_prediction_correlation": live_metrics["correlation"],
            "median_volatility": strategy_result.median_volatility, "volatility_scale": strategy_result.volatility_scale,
            "history_points": history_points, "history_required": required, "rebalance_due": due,
            "orders_enabled": self.config.orders_enabled, "planned_sells": result.get("planned_sells", []),
            "planned_buys": result.get("planned_buys", []), "planned_short_opens": result.get("planned_short_opens", []),
            "planned_short_closes": result.get("planned_short_closes", []), "today_trades": today_trades,
        }
