from __future__ import annotations

import logging
import math
import os
import numpy as np
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv(override=False)

from .api.client import RoostooClient
from .config import Config
from .db import BotDB
from .logger.trade_logger import TradeLogger
from .paper_trader import PaperTrader
from .risk.manager import RiskManager
from .strategy.boss import BossStrategy, Market, Signal, StrategyResult

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("boss")


class TeeStream:
    def __init__(self, path: str):
        self._stdout = sys.stdout
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, value: str):
        self._stdout.write(value)
        self._file.write(value)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return self._stdout.isatty()


@dataclass
class Position:
    pair: str
    coin: str
    qty: float
    free_qty: float
    current_price: float
    value: float
    entry_price: float
    entry_ts: int
    pnl_pct: float
    age_minutes: float


@dataclass
class ShortPosition:
    pair: str
    qty: float
    collateral: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    position_value: float
    entry_ts: int
    pnl_pct: float
    age_minutes: float


def setup_logging(path: str) -> None:
    if not isinstance(sys.stdout, TeeStream):
        sys.stdout = TeeStream(path)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def now_ms() -> int:
    return int(time.time() * 1000)


def now_label() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def wallet_from_balance(data: dict) -> dict:
    if not isinstance(data, Mapping) or data.get("Success") is False:
        raise RuntimeError(f"balance endpoint failed: {data.get('ErrMsg', 'invalid response') if isinstance(data, Mapping) else 'invalid response'}")
    wallet = data.get("SpotWallet") or data.get("Wallet")
    if not isinstance(wallet, Mapping):
        raise RuntimeError("balance endpoint returned no SpotWallet/Wallet")
    return dict(wallet)


def portfolio_value(wallet: dict, tickers: dict, short_positions: list[ShortPosition] | list[dict] | tuple = ()) -> float:
    """Mark spot holdings plus the equity value (collateral + P&L) of open shorts."""
    usd = wallet.get("USD", {}) or {}
    total = float(usd.get("Free", 0) or 0) + float(usd.get("Lock", 0) or 0)
    for coin, amounts in wallet.items():
        if coin == "USD":
            continue
        qty = float(amounts.get("Free", 0) or 0) + float(amounts.get("Lock", 0) or 0)
        ticker = tickers.get(f"{coin}/USD", {}) or {}
        px = float(ticker.get("LastPrice", 0) or 0)
        if px > 0:
            total += qty * px
    for pos in short_positions:
        if isinstance(pos, ShortPosition):
            total += max(0.0, pos.position_value)
        elif isinstance(pos, Mapping):
            collateral = float(pos.get("Collateral", pos.get("collateral", 0)) or 0)
            pnl = float(pos.get("UnrealizedPNL", pos.get("unrealized_pnl", 0)) or 0)
            value = pos.get("PositionValue", pos.get("position_value"))
            total += max(0.0, float(value) if value is not None else collateral + pnl)
    return total


def free_usd(wallet: dict) -> float:
    return float((wallet.get("USD", {}) or {}).get("Free", 0) or 0)


def _order_cap(notional: float, cfg: Config) -> float:
    """Apply an optional order-size ceiling; MAX_ORDER_USD=0 means unlimited."""
    value = max(0.0, float(notional))
    return value if cfg.max_order_usd <= 0 else min(value, cfg.max_order_usd)


def build_markets(exchange_info: dict, tickers: dict, cfg: Config) -> dict[str, Market]:
    """Build every currently tradable USD market; the strategy applies entry whitelists."""
    result: dict[str, Market] = {}
    metas = exchange_info.get("TradePairs", {}) if isinstance(exchange_info, Mapping) else {}
    for pair, meta in metas.items():
        if not meta.get("CanTrade", False) or not str(pair).endswith("/USD") or pair not in tickers:
            continue
        t = tickers[pair]
        price = float(t.get("LastPrice", 0) or 0)
        bid = float(t.get("MaxBid", 0) or 0)
        ask = float(t.get("MinAsk", 0) or 0)
        if price <= 0 or bid <= 0 or ask <= 0 or bid > ask:
            continue
        volume = float(t.get("UnitTradeValue", 0) or t.get("CoinTradeValue", 0) or 0)
        result[pair] = Market(
            pair=pair,
            coin=str(pair).split("/")[0],
            price=price,
            bid=bid,
            ask=ask,
            change_24h=float(t.get("Change", 0) or 0),
            quote_volume_24h=volume,
            amount_precision=int(meta.get("AmountPrecision", 6) or 6),
            min_order=float(meta.get("MiniOrder", 1.0) or 1.0),
            can_trade=True,
        )
    return result


def positions_from_wallet(wallet: dict, markets: dict[str, Market], db: BotDB,
                          paper_trader: PaperTrader | None = None) -> dict[str, Position]:
    costs = {} if paper_trader is not None else db.position_cost_basis()
    out: dict[str, Position] = {}
    ts = now_ms()
    for coin, amounts in wallet.items():
        if coin == "USD":
            continue
        total_qty = float(amounts.get("Free", 0) or 0) + float(amounts.get("Lock", 0) or 0)
        free_qty = float(amounts.get("Free", 0) or 0)
        pair = f"{coin}/USD"
        market = markets.get(pair)
        if total_qty <= 0 or market is None:
            continue
        if paper_trader is not None:
            paper_pos = paper_trader._positions.get(coin)
            entry_price = float(paper_pos.avg_entry) if paper_pos else market.price
            entry_ts = int(paper_pos.create_timestamp) if paper_pos else ts
        else:
            cost = costs.get(pair, {})
            entry_price = float(cost.get("cost", 0) or 0) / float(cost.get("qty", 1) or 1) if cost.get("qty", 0) else market.price
            entry_ts = int(cost.get("entry_ts", ts) or ts)
        pnl = (market.price - entry_price) / entry_price if entry_price > 0 else 0.0
        out[pair] = Position(pair, coin, total_qty, free_qty, market.price, total_qty * market.price,
                             entry_price, entry_ts, pnl * 100.0, max(0.0, (ts - entry_ts) / 60_000.0))
    return out


def shorts_from_response(data: dict, markets: dict[str, Market], current_ms: int | None = None) -> dict[str, ShortPosition]:
    if not isinstance(data, Mapping) or data.get("Success") is not True:
        message = data.get("ErrMsg", "invalid response") if isinstance(data, Mapping) else "invalid response"
        raise RuntimeError(f"short_positions endpoint failed: {message}")
    raw_positions = data.get("Positions", []) or []
    if not isinstance(raw_positions, list):
        raise RuntimeError("short_positions endpoint returned malformed Positions")
    ts = current_ms or now_ms()
    out: dict[str, ShortPosition] = {}
    for item in raw_positions:
        if not isinstance(item, Mapping):
            continue
        pair = str(item.get("Pair", ""))
        qty = float(item.get("ShortQty", 0) or 0)
        collateral = float(item.get("Collateral", 0) or 0)
        entry = float(item.get("EntryPrice", 0) or 0)
        if not pair or qty <= 0 or collateral < 0 or entry <= 0:
            continue
        market = markets.get(pair)
        mark = float(item.get("CurrentPrice", 0) or (market.ask if market else 0) or entry)
        pnl = float(item.get("UnrealizedPNL", qty * (entry - mark)) or 0)
        value_field = item.get("PositionValue")
        value = max(0.0, float(value_field) if value_field is not None else collateral + pnl)
        create_ts = int(item.get("CreateTimestamp", item.get("Timestamp", ts)) or ts)
        out[pair] = ShortPosition(pair, qty, collateral, entry, mark, pnl, value, create_ts,
                                  pnl / collateral * 100.0 if collateral > 0 else 0.0,
                                  max(0.0, (ts - create_ts) / 60_000.0))
    return out


def _round_qty(qty: float, precision: int) -> float:
    factor = 10 ** max(0, int(precision))
    return math.floor(max(0.0, qty) * factor + 1e-9) / factor


def _save_order(result: dict, db: BotDB, trade_logger: TradeLogger, cfg: Config, mode: str,
                started: int, note: str) -> None:
    trade_logger.log_order(result, note)
    db.record_trade_from_order(result, mode, started)


def _query_orders(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, Mapping):
        return []
    for key in ("OrderMatched", "OrderDetails", "Orders", "Data", "OrderDetail"):
        value = raw.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
        if isinstance(value, dict) and ("Pair" in value or "OrderID" in value or "Side" in value):
            return [dict(value)]
    return []


def _latch_uncertain(db: BotDB, cfg: Config, detail: dict) -> None:
    key = cfg.bot_namespace + ":execution_uncertain"
    db.set_state(key, {"timestamp_ms": now_ms(), **detail})
    logger.critical("execution state uncertain; all new orders halted until manually reconciled: %s", detail)


def execute_order(client: Any, db: BotDB, logger_trade: TradeLogger, pair: str, side: str,
                  qty: float, cfg: Config, reason: str, price: float | None = None) -> tuple[bool, dict]:
    if qty <= 0:
        return False, {"Success": False, "ErrMsg": "quantity_nonpositive"}
    if cfg.paper_mode and not isinstance(client, PaperTrader):
        return False, {"Success": False, "ErrMsg": "paper_mode_requires_local_simulator"}
    # ORDERS_ENABLED is a live-exchange kill switch. Paper mode must still
    # execute against the local simulator with the safe default false setting.
    if not cfg.paper_mode and not cfg.orders_enabled:
        return False, {"Success": False, "ErrMsg": "orders_disabled"}
    started = now_ms()
    try:
        if cfg.paper_mode:
            result = client.place_order(pair, side, qty, order_type="MARKET", price=price)
        else:
            result = client.place_order(pair, side, qty, order_type="MARKET")
    except Exception as exc:
        logger.error("order exception %s %s: %s", side, pair, exc)
        result = {}
    if isinstance(result, Mapping) and result.get("Success") is True:
        wrapped = dict(result)
        _save_order(wrapped, db, logger_trade, cfg, "paper" if cfg.paper_mode else "live", started, reason)
        return True, wrapped
    if isinstance(result, Mapping) and result.get("Success") is False:
        logger.warning("order rejected %s %s: %s", side, pair, result.get("ErrMsg", "unspecified rejection"))
        return False, dict(result)

    # A network error may have happened after the matching engine accepted the order.
    # Reconcile without posting another order. Unknown results latch the executor safely.
    if not cfg.paper_mode:
        try:
            recent = _query_orders(client.query_order(pair=pair, pending_only=False, limit=10))
            for order in recent:
                dts = int(order.get("CreateTimestamp", 0) or 0)
                if dts >= started - 15_000 and str(order.get("Side", "")).upper() == side.upper():
                    wrapped = {"Success": True, "OrderDetail": order}
                    logger.warning("reconciled ambiguous order %s %s via query_order", side, pair)
                    _save_order(wrapped, db, logger_trade, cfg, "live", started, reason + ":reconciled")
                    return True, wrapped
        except Exception as exc:
            logger.error("order reconciliation failed for %s %s: %s", side, pair, exc)
    _latch_uncertain(db, cfg, {"type": "spot", "pair": pair, "side": side, "qty": qty, "reason": reason})
    return False, dict(result) if isinstance(result, Mapping) else {}


def _normalize_short_order(pair: str, side: str, qty: float, price: float, fee: float,
                           order_id: Any = None, status: str = "FILLED", timestamp_ms: int | None = None) -> dict:
    detail = {
        "Pair": pair, "Side": side, "ShortQty" if side == "SHORT_OPEN" else "ClosedQty": max(0.0, float(qty)),
        "EntryPrice" if side == "SHORT_OPEN" else "ClosePrice": float(price),
        "OpenFee" if side == "SHORT_OPEN" else "CloseFee": max(0.0, float(fee)),
        "ID": order_id or "", "Status": status, "CreateTimestamp": timestamp_ms or now_ms(),
    }
    return {"Success": True, "OrderDetail": detail}


def fetch_short_positions(trader: Any, paper_mode: bool, tickers: dict, markets: dict[str, Market]) -> dict[str, ShortPosition]:
    data = trader.short_positions(tickers) if paper_mode else trader.short_positions()
    return shorts_from_response(data, markets)


def execute_short_open(trader: Any, db: BotDB, trade_logger: TradeLogger, cfg: Config,
                       pair: str, collateral: float, market: Market, before: ShortPosition | None,
                       tickers: dict) -> tuple[bool, dict]:
    if collateral <= 0:
        return False, {"Success": False, "ErrMsg": "collateral_nonpositive"}
    if cfg.paper_mode and not isinstance(trader, PaperTrader):
        return False, {"Success": False, "ErrMsg": "paper_mode_requires_local_simulator"}
    # Local simulated shorts are safe to run with ORDERS_ENABLED=false.
    if not cfg.paper_mode and not cfg.orders_enabled:
        return False, {"Success": False, "ErrMsg": "orders_disabled"}
    started = now_ms()
    before_qty = before.qty if before else 0.0
    before_collateral = before.collateral if before else 0.0
    try:
        if cfg.paper_mode:
            result = trader.short_open(pair, collateral, price=market.bid)
        else:
            # No price field means marketable best-bid execution on the Roostoo short endpoint.
            result = trader.short_open(pair, collateral)
    except Exception as exc:
        logger.error("short-open exception %s: %s", pair, exc)
        result = {}
    if isinstance(result, Mapping) and result.get("Success") is True:
        d = result.get("OrderDetail", result)
        qty = float(d.get("ShortQty", d.get("Qty", collateral / market.bid)) or collateral / market.bid)
        price = float(d.get("EntryPrice", d.get("Price", market.bid)) or market.bid)
        fee = float(d.get("OpenFee", d.get("CommissionChargeValue", qty * price * cfg.short_fee_assumption)) or 0)
        normalized = _normalize_short_order(pair, "SHORT_OPEN", qty, price, fee,
                                            d.get("ID", d.get("OrderID")), "FILLED", int(d.get("CreateTimestamp", started) or started))
        _save_order(normalized, db, trade_logger, cfg, "paper" if cfg.paper_mode else "live", started, "short_enter")
        return True, {**dict(result), "normalized": normalized}
    if isinstance(result, Mapping) and result.get("Success") is False:
        return False, dict(result)

    # Reconcile an ambiguous open using the authoritative short-position inventory.
    if not cfg.paper_mode:
        try:
            after = fetch_short_positions(trader, False, tickers, {})
            post = after.get(pair)
            if post and (post.collateral > before_collateral + max(0.5, collateral * 0.75)):
                qty = max(0.0, post.qty - before_qty)
                if qty > 0:
                    fee = collateral * cfg.short_fee_assumption
                    normalized = _normalize_short_order(pair, "SHORT_OPEN", qty, post.entry_price, fee,
                                                        None, "FILLED", started)
                    _save_order(normalized, db, trade_logger, cfg, "live", started, "short_enter:reconciled")
                    return True, {"Success": True, "normalized": normalized}
        except Exception as exc:
            logger.error("short-open reconciliation failed %s: %s", pair, exc)
    _latch_uncertain(db, cfg, {"type": "short_open", "pair": pair, "collateral": collateral})
    return False, dict(result) if isinstance(result, Mapping) else {}


def execute_short_close(trader: Any, db: BotDB, trade_logger: TradeLogger, cfg: Config,
                        position: ShortPosition, market: Market, close_pct: float, tickers: dict) -> tuple[bool, dict]:
    if cfg.paper_mode and not isinstance(trader, PaperTrader):
        return False, {"Success": False, "ErrMsg": "paper_mode_requires_local_simulator"}
    # Local simulated closes are safe to run with ORDERS_ENABLED=false.
    if not cfg.paper_mode and not cfg.orders_enabled:
        return False, {"Success": False, "ErrMsg": "orders_disabled"}
    started = now_ms()
    try:
        if cfg.paper_mode:
            # The local simulator needs an explicit ask-side fill price. The live
            # Roostoo /v6/short_close endpoint takes no price and fills at MinAsk.
            result = trader.short_close(position.pair, close_pct=close_pct, price=market.ask)
        elif close_pct >= 99.999:
            result = trader.short_close(position.pair)
        else:
            result = trader.short_close(position.pair, close_pct=close_pct)
    except Exception as exc:
        logger.error("short-close exception %s: %s", position.pair, exc)
        result = {}
    if isinstance(result, Mapping) and result.get("Success") is True:
        d = result.get("OrderDetail", result)
        qty = float(d.get("ClosedQty", d.get("ShortQty", position.qty * close_pct / 100.0)) or position.qty * close_pct / 100.0)
        price = float(d.get("ClosePrice", d.get("CurrentPrice", market.ask)) or market.ask)
        fee = float(d.get("CloseFee", d.get("CommissionChargeValue", qty * price * cfg.short_fee_assumption)) or 0)
        normalized = _normalize_short_order(position.pair, "SHORT_CLOSE", qty, price, fee,
                                            d.get("ID", d.get("OrderID")), "FILLED", int(d.get("CreateTimestamp", started) or started))
        _save_order(normalized, db, trade_logger, cfg, "paper" if cfg.paper_mode else "live", started, "short_exit")
        return True, {**dict(result), "normalized": normalized}
    if isinstance(result, Mapping) and result.get("Success") is False:
        return False, dict(result)

    if not cfg.paper_mode:
        try:
            after = fetch_short_positions(trader, False, tickers, {})
            post = after.get(position.pair)
            post_qty = post.qty if post else 0.0
            closed_qty = max(0.0, position.qty - post_qty)
            if closed_qty > max(1e-8, position.qty * 0.01):
                fee = closed_qty * market.ask * cfg.short_fee_assumption
                normalized = _normalize_short_order(position.pair, "SHORT_CLOSE", closed_qty, market.ask, fee,
                                                    None, "FILLED", started)
                _save_order(normalized, db, trade_logger, cfg, "live", started, "short_exit:reconciled")
                return True, {"Success": True, "normalized": normalized}
        except Exception as exc:
            logger.error("short-close reconciliation failed %s: %s", position.pair, exc)
    _latch_uncertain(db, cfg, {"type": "short_close", "pair": position.pair, "qty": position.qty, "close_pct": close_pct})
    return False, dict(result) if isinstance(result, Mapping) else {}



def _realized_hourly_volatility(history: list[float] | tuple[float, ...] | None, cfg: Config) -> float:
    """Estimate hourly log-return volatility from non-overlapping 60-minute returns.

    Non-overlapping horizon returns avoid treating dozens of highly overlapping
    60-second observations as independent hourly samples.
    """
    fallback = max(float(cfg.target_hourly_vol), 1e-4)
    prices = np.asarray(history or (), dtype=float)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    horizon_bars = max(1, cfg.bars(60))
    if prices.size < horizon_bars * 4:
        return fallback
    values: list[float] = []
    end_idx = int(prices.size - 1)
    # Use up to 48 recent non-overlapping hourly returns.
    for offset in range(0, min(48, (prices.size - 1) // horizon_bars) * horizon_bars, horizon_bars):
        end = end_idx - offset
        start = end - horizon_bars
        if start < 0:
            break
        value = math.log(float(prices[end]) / float(prices[start]))
        if math.isfinite(value):
            values.append(value)
    if len(values) < 3:
        return fallback
    estimate = float(np.std(np.asarray(values, dtype=float), ddof=1))
    return estimate if math.isfinite(estimate) and estimate > 1e-6 else fallback


def _exit_thresholds(pair: str, cfg: Config, market: Market | None,
                     history: list[float] | tuple[float, ...] | None,
                     is_short: bool = False) -> dict[str, float]:
    """Return stop/activation/trail/target-profit distances as decimal fractions.

    Thresholds adapt to realized hourly volatility; the transaction-cost floor
    prevents a volatility stop from being placed inside the modeled round-trip
    cost. The fixed values remain available as a legacy mode via the env flag.
    """
    if not cfg.adaptive_exits_enabled:
        return {
            "stop": cfg.short_stop_loss_pct if is_short else cfg.stop_loss_pct,
            "activation": cfg.short_trailing_activation_pct if is_short else cfg.trailing_activation_pct,
            "trail": cfg.short_trailing_stop_pct if is_short else cfg.trailing_stop_pct,
            "take_profit": cfg.take_profit_pct,
        }
    hourly_vol = _realized_hourly_volatility(history, cfg)
    spread_bps = float(getattr(market, "spread_bps", 0.0)) if market is not None else 0.0
    if not math.isfinite(spread_bps):
        spread_bps = 0.0
    one_way_cost = max(0.0, cfg.commission_rate + spread_bps / 20_000.0 + cfg.slippage_bps / 10_000.0)
    round_trip_cost = 2.0 * one_way_cost
    if is_short:
        stop_multiple = cfg.short_stop_vol_multiple
        activation_multiple = cfg.short_trailing_activation_vol_multiple
        trail_multiple = cfg.short_trailing_stop_vol_multiple
    else:
        stop_multiple = cfg.stop_vol_multiple
        activation_multiple = cfg.trailing_activation_vol_multiple
        trail_multiple = cfg.trailing_stop_vol_multiple
    return {
        "stop": max(stop_multiple * hourly_vol, 2.0 * round_trip_cost),
        "activation": max(activation_multiple * hourly_vol, 2.0 * round_trip_cost),
        "trail": max(trail_multiple * hourly_vol, round_trip_cost),
        "take_profit": max(cfg.take_profit_vol_multiple * hourly_vol, 3.0 * round_trip_cost),
        "hourly_volatility": hourly_vol,
        "round_trip_cost": round_trip_cost,
    }

def plan_targets(strategy_result: StrategyResult, positions: dict[str, Position], equity: float,
                 risk: RiskManager, db: BotDB, cfg: Config, markets: dict[str, Market],
                 histories: Mapping[str, list[float]] | None = None) -> tuple[dict[str, float], set[str], list[str]]:
    targets = {pair: equity * weight for pair, weight in strategy_result.long_targets.items()}
    forced: set[str] = set(strategy_result.bearish_veto_pairs)
    reasons: list[str] = []
    if strategy_result.gross_target <= 0.0:
        forced.update(positions.keys())
        reasons.append("cash_protection")
    if risk.snapshot.halted:
        forced.update(positions.keys())
        reasons.append("risk_lock")

    state_prefix = cfg.bot_namespace + ":peak:"
    for pair, pos in positions.items():
        entry_key = state_prefix + "entry_ts:" + pair
        saved_entry_ts = db.get_state(entry_key, None)
        if saved_entry_ts is None or int(saved_entry_ts) != int(pos.entry_ts):
            # Do not carry a previous trade's peak into a newly opened position.
            peak = pos.current_price
            db.set_state(entry_key, int(pos.entry_ts))
        else:
            old_peak = db.get_state(state_prefix + pair, pos.entry_price)
            peak = max(float(old_peak or pos.entry_price), pos.current_price)
        db.set_state(state_prefix + pair, peak)
        thresholds = _exit_thresholds(pair, cfg, markets.get(pair), (histories or {}).get(pair), is_short=False)
        trailing_hit = (peak >= pos.entry_price * (1.0 + thresholds["activation"])
                        and pos.current_price <= peak * (1.0 - thresholds["trail"]))
        hard_stop = pos.current_price <= pos.entry_price * (1.0 - thresholds["stop"])
        if hard_stop:
            forced.add(pair); reasons.append(f"stop:{pair}")
        elif trailing_hit:
            forced.add(pair); reasons.append(f"trailing:{pair}")
        elif pos.pnl_pct <= -cfg.proactive_loss_pct * 100.0 and pair not in targets:
            forced.add(pair); reasons.append(f"proactive:{pair}")
        elif pos.current_price >= pos.entry_price * (1.0 + thresholds["take_profit"]):
            forced.add(pair); reasons.append(f"take_profit:{pair}")

    # Hysteresis: for a selected rotation, swap a stale incumbent for one incoming name
    # rather than repeatedly selling and rebuying around small score changes.
    rotations = 0
    max_rotations = max(0, int(cfg.rotation_max_per_rebalance)) if cfg.rotation_enabled else -1
    signal_scores = {s.pair: s.score for s in getattr(strategy_result, "signals", [])}
    selected_new = [p for p in targets if p not in positions and p not in forced]
    selected_new.sort(key=lambda p: signal_scores.get(p, 0.0))
    if strategy_result.gross_target > 0 and not risk.snapshot.halted and cfg.rotation_enabled:
        for pair, pos in sorted(positions.items(), key=lambda kv: kv[1].age_minutes):
            if (max_rotations > 0 and rotations >= max_rotations) or pair in forced or targets.get(pair, 0.0) > 0 or not selected_new:
                continue
            held_score = float(getattr(strategy_result, "all_scores", {}).get(pair, 0.0))
            incoming = selected_new[0]
            incoming_score = signal_scores.get(incoming, float(getattr(strategy_result, "all_scores", {}).get(incoming, 0.0)))
            recent = pos.age_minutes < cfg.min_hold_minutes
            close_score = held_score >= incoming_score - cfg.rotation_score_gap
            if not (recent or close_score):
                continue
            targets.pop(incoming, None)
            selected_new.pop(0)
            targets[pair] = min(pos.value, equity * cfg.max_asset_weight)
            rotations += 1
            reasons.append(f"rotation_hysteresis:{pair} retains slot vs {incoming}")

    for pair in forced:
        targets[pair] = 0.0
    return targets, forced, reasons


def build_plan(target_values: dict[str, float], forced: set[str], positions: dict[str, Position], markets: dict[str, Market],
               equity: float, cfg: Config, turnover_remaining: float, wallet: dict) -> tuple[list[dict], list[dict]]:
    threshold = max(cfg.min_order_usd, equity * cfg.rebalance_band)
    sells: list[dict] = []
    buys: list[dict] = []
    for pair, pos in positions.items():
        market = markets.get(pair)
        if not market:
            continue
        target = float(target_values.get(pair, 0.0))
        if pair in forced:
            if pos.free_qty > 0:
                sells.append({"pair": pair, "notional": pos.free_qty * pos.current_price, "forced": True, "reason": "forced_exit"})
            continue
        if target <= 0:
            if pos.free_qty > 0:
                sells.append({"pair": pair, "notional": _order_cap(pos.free_qty * pos.current_price, cfg), "forced": False, "reason": "target_zero"})
            continue
        gap = pos.value - target
        if gap > threshold and pos.free_qty > 0:
            sells.append({"pair": pair, "notional": _order_cap(min(gap, pos.free_qty * pos.current_price), cfg), "forced": False, "reason": "target_reduce"})

    for pair, target in target_values.items():
        market = markets.get(pair)
        if market is None or target <= 0:
            continue
        current = positions[pair].value if pair in positions else 0.0
        gap = target - current
        if gap > threshold:
            buys.append({"pair": pair, "notional": _order_cap(gap, cfg), "forced": False, "reason": "target_add"})
    sells.sort(key=lambda x: (not x["forced"], -x["notional"]))
    return sells, buys


def plan_short_targets(strategy_result: StrategyResult, shorts: dict[str, ShortPosition], equity: float,
                       risk: RiskManager, db: BotDB, cfg: Config,
                       markets: dict[str, Market] | None = None,
                       histories: Mapping[str, list[float]] | None = None) -> tuple[dict[str, float], set[str], list[str]]:
    targets = {pair: equity * weight for pair, weight in getattr(strategy_result, "short_targets", {}).items()}
    forced: set[str] = set()
    reasons: list[str] = []
    prefix = cfg.bot_namespace + ":short_trough:"
    for pair, pos in shorts.items():
        entry_key = prefix + "entry_ts:" + pair
        saved_entry_ts = db.get_state(entry_key, None)
        if saved_entry_ts is None or int(saved_entry_ts) != int(pos.entry_ts):
            # Reset the favorable-trough reference when a short position is new.
            trough = pos.current_price
            db.set_state(entry_key, int(pos.entry_ts))
        else:
            prior_trough = float(db.get_state(prefix + pair, pos.entry_price) or pos.entry_price)
            trough = min(prior_trough, pos.current_price)
        db.set_state(prefix + pair, trough)
        thresholds = _exit_thresholds(pair, cfg, (markets or {}).get(pair), (histories or {}).get(pair), is_short=True)
        trailing_hit = (trough <= pos.entry_price * (1.0 - thresholds["activation"])
                        and pos.current_price >= trough * (1.0 + thresholds["trail"]))
        hard_stop = pos.current_price >= pos.entry_price * (1.0 + thresholds["stop"])
        unavailable_regime = strategy_result.regime == "risk_on" and not cfg.allow_shorts_risk_on
        if risk.snapshot.halted:
            forced.add(pair); reasons.append("risk_lock")
        elif hard_stop:
            forced.add(pair); reasons.append(f"short_stop:{pair}")
        elif trailing_hit:
            forced.add(pair); reasons.append(f"short_trailing:{pair}")
        elif pos.current_price <= pos.entry_price * (1.0 - thresholds["take_profit"]):
            forced.add(pair); reasons.append(f"short_take_profit:{pair}")
        elif unavailable_regime:
            forced.add(pair); reasons.append(f"short_regime_exit:{pair}")
        elif pair not in targets and cfg.shorts_enabled and pos.age_minutes < cfg.min_hold_minutes:
            # Avoid a churn-close immediately after entry; this never overrides stops or risk locks.
            targets[pair] = min(pos.collateral, equity * cfg.max_short_asset_weight)
    for pair in forced:
        targets[pair] = 0.0
    return targets, forced, reasons


def build_short_plan(target_values: dict[str, float], forced: set[str], shorts: dict[str, ShortPosition],
                     markets: dict[str, Market], equity: float, cfg: Config) -> tuple[list[dict], list[dict]]:
    threshold = max(cfg.min_order_usd, equity * cfg.rebalance_band)
    closes: list[dict] = []
    opens: list[dict] = []
    for pair, pos in shorts.items():
        market = markets.get(pair)
        if market is None:
            continue
        target = max(0.0, float(target_values.get(pair, 0.0)))
        gap = pos.collateral - target
        if pair in forced:
            closes.append({"pair": pair, "close_pct": 100.0, "notional": pos.qty * market.ask,
                           "forced": True, "reason": "short_forced_exit"})
        elif target <= 0 and pos.collateral > threshold:
            fraction = 1.0 if cfg.max_order_usd <= 0 else min(1.0, cfg.max_order_usd / max(pos.collateral, 1e-12))
            closes.append({"pair": pair, "close_pct": 100.0 * fraction, "notional": pos.qty * market.ask * fraction,
                           "forced": False, "reason": "short_target_zero"})
        elif gap > threshold:
            reduce_collateral = _order_cap(gap, cfg)
            closes.append({"pair": pair, "close_pct": min(100.0, 100.0 * reduce_collateral / max(pos.collateral, 1e-12)),
                           "notional": pos.qty * market.ask * reduce_collateral / max(pos.collateral, 1e-12),
                           "forced": False, "reason": "short_target_reduce"})

    for pair, target in target_values.items():
        market = markets.get(pair)
        if market is None or target <= 0:
            continue
        current = shorts[pair].collateral if pair in shorts else 0.0
        gap = float(target) - current
        if gap > threshold:
            opens.append({"pair": pair, "collateral": _order_cap(gap, cfg), "notional": _order_cap(gap, cfg),
                          "forced": False, "reason": "short_target_add" if pair in shorts else "short_enter"})
    closes.sort(key=lambda x: (not x["forced"], -x["notional"]))
    return closes, opens


def print_dashboard(cfg: Config, equity: float, start_equity: float, wallet: dict,
                    markets: dict[str, Market], result: StrategyResult, positions: dict[str, Position],
                    turnover_used: float, turnover_cap: float, next_rebalance_ms: int,
                    executed: list[tuple[str, str, float, float]], blocked: list[str], risk: RiskManager,
                    short_positions: dict[str, ShortPosition] | None = None,
                    execution_override: str | None = None) -> None:
    short_positions = short_positions or {}
    # Competition P&L includes all account performance since the original starting
    # portfolio, including losses realized by an earlier model. The current exchange
    # equity remains the sole basis for live allocation and order sizing.
    performance_baseline = float(cfg.expected_starting_cash or start_equity or 0.0)
    performance_pnl = equity - performance_baseline if performance_baseline else 0.0
    performance_pnl_pct = (performance_pnl / performance_baseline * 100.0
                            if performance_baseline else 0.0)
    exposure = (sum(p.value for p in positions.values()) + sum(p.position_value for p in short_positions.values())) / equity if equity else 0.0
    execution_line = execution_override or ("LOCAL SIMULATOR" if cfg.paper_mode else ("LIVE ENABLED" if cfg.orders_enabled else "LIVE DISABLED"))
    next_secs = max(0, int((next_rebalance_ms - now_ms()) / 1000))
    next_label = "N/A (inspection only)" if execution_override and "INSPECT" in execution_override.upper() else f"{next_secs}s"
    turnover_line = (f"${turnover_used:,.2f} / UNLIMITED" if cfg.max_daily_turnover == 0 else
                     f"${turnover_used:,.2f} / ${turnover_cap:,.2f} (remaining ${max(0.0, turnover_cap-turnover_used):,.2f})")
    print("\n" + "=" * 118)
    print("ROOSTOO QUANT BOSS V9.4 — LONG / SHORT")
    print("=" * 118)
    print(f"TIME {now_label()} | EXECUTION {execution_line} | POLL {cfg.poll_seconds}s | REBALANCE {cfg.rebalance_seconds}s | NEXT {next_label}")
    print(f"EQUITY ${equity:,.2f} | BOSS P&L ${performance_pnl:+,.2f} ({performance_pnl_pct:+.2f}%) from ${performance_baseline:,.2f} competition baseline | GROSS EXPOSURE {exposure*100:.2f}% | CASH ${free_usd(wallet):,.2f}")
    print(f"RISK daily={risk.snapshot.daily_pnl_pct*100:+.2f}% drawdown={risk.snapshot.drawdown_pct*100:+.2f}% halted={risk.snapshot.halted}")
    print(f"REGIME {result.regime.upper()} | breadth={result.breadth*100:.1f}% | strength={result.market_strength:.3f} | 4h={result.median_4h*100:+.2f}% | 12h={result.median_12h*100:+.2f}% | 24h={result.median_24h*100:+.2f}% | long gross={result.gross_target*100:.2f}% | short collateral target={getattr(result, 'short_gross_target', 0.0)*100:.2f}%")
    print(f"MARKETS {len(markets)} | MODEL rank={'ACTIVE' if result.predictor_enabled else 'GATED'} direction={'ACTIVE' if result.predictor_directional_enabled else 'GATED'} | samples={result.predictor_samples} | dir={result.predictor_directional_accuracy*100:.1f}% | corr={result.predictor_correlation:+.3f} | rankIC={result.predictor_rank_ic:+.3f} | bal={result.predictor_balanced_accuracy*100:.1f}% | top10net={result.predictor_top_decile_net_return*100:+.2f}% | brier={result.predictor_brier:.3f}")
    print(f"MODEL GATE: {result.predictor_gate_reason}")
    print(f"TURNOVER {turnover_line} | ORDER CAP={'UNLIMITED' if cfg.max_order_usd == 0 else f'${cfg.max_order_usd:,.2f}'} | GLOBAL HALTS={'DISABLED' if cfg.daily_loss_limit_pct == 0 and cfg.max_drawdown_limit_pct == 0 else 'CONFIGURED'}")
    if result.bearish_veto_pairs:
        print("BEARISH VETO:", ", ".join(result.bearish_veto_pairs))
    print("\nLONG SIGNALS")
    for s in result.signals:
        print(f"  {s.pair:<14} score={s.score:.3f} target={s.target_weight*100:5.2f}% edge={s.expected_edge*100:+.2f}% pred={s.predicted_return*100:+.2f}% conf={s.prediction_confidence*100:4.0f}% {s.reason}")
    print("\nSHORT SIGNALS")
    for s in result.short_signals:
        print(f"  {s.pair:<14} score={s.score:.3f} collateral={s.target_weight*100:5.2f}% edge={s.expected_edge*100:+.2f}% pred={s.predicted_return*100:+.2f}% conf={s.prediction_confidence*100:4.0f}% {s.reason}")
    print("\nSPOT POSITIONS")
    if positions:
        for p in sorted(positions.values(), key=lambda x: x.value, reverse=True):
            print(f"  {p.pair:<14} ${p.value:9,.2f} pnl={p.pnl_pct:+6.2f}% age={p.age_minutes:6.1f}m entry={p.entry_price:.6g}")
    else:
        print("  NONE")
    print("\nSHORT POSITIONS")
    if short_positions:
        for p in sorted(short_positions.values(), key=lambda x: x.collateral, reverse=True):
            print(f"  {p.pair:<14} coll=${p.collateral:8,.2f} value=${p.position_value:8,.2f} pnl={p.pnl_pct:+6.2f}% age={p.age_minutes:6.1f}m entry={p.entry_price:.6g}")
    else:
        print("  NONE")
    print("\nEXECUTION")
    for side, pair, notional, _ in executed:
        print(f"  {side:<11} {pair:<14} ~${notional:,.2f}")
    for item in blocked:
        print(f"  BLOCK {item}")
    print("=" * 118)


def log_predictions(db: BotDB, result: StrategyResult, markets: dict[str, Market], cfg: Config, ts: int) -> int:
    if not result.predictor_predictions:
        return 0
    unsettled = db.unsettled_prediction_pairs(ts)
    candidates = [(p, pred) for p, pred in result.predictor_predictions.items() if p in markets and p not in unsettled]
    candidates.sort(key=lambda x: abs(x[1].predicted_return), reverse=True)
    wrote = 0
    target_time = ts + cfg.prediction_horizon * 60_000
    for pair, pred in candidates[:cfg.prediction_log_top_n]:
        db.record_prediction(ts, pair, markets[pair].price, target_time, pred.predicted_return,
                             pred.probability_up, pred.probability_flat, pred.probability_down,
                             pred.confidence, result.predictor_enabled)
        wrote += 1
    return wrote


def _daily_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _mode(cfg: Config) -> str:
    return "paper" if cfg.paper_mode else "live"


def _refresh_wallet(trader: Any) -> dict:
    return wallet_from_balance(trader.balance())


def run() -> None:
    cfg = Config()
    cfg.validate(require_credentials=not cfg.paper_mode, require_api_url=True)
    setup_logging(cfg.log_file)
    execution_mode = "LOCAL_SIMULATOR" if cfg.paper_mode else ("LIVE_ORDERS_ENABLED" if cfg.orders_enabled else "LIVE_ORDERS_DISABLED")
    print(f"[SYSTEM] BOSS V9 AWS | namespace={cfg.bot_namespace} | poll={cfg.poll_seconds}s | rebalance={cfg.rebalance_seconds}s | execution={execution_mode} | turnover={'UNLIMITED' if cfg.max_daily_turnover == 0 else 'CAPPED'}")

    # The exchange client supplies public prices in paper mode, but is never used for orders there.
    market_client = RoostooClient()
    if not market_client.server_time().get("ServerTime"):
        raise RuntimeError("Roostoo API connection failed: serverTime unavailable")
    exchange_info = market_client.exchange_info()
    if not exchange_info.get("TradePairs"):
        raise RuntimeError("exchangeInfo did not return TradePairs")
    last_exchange_refresh_ms = now_ms()
    last_account_refresh_ms = 0
    trader: Any = PaperTrader(cfg.expected_starting_cash) if cfg.paper_mode else market_client

    db = BotDB(cfg.db_path, cfg.history_limit)
    logger_trade = TradeLogger(os.path.dirname(cfg.log_file) or "logs")
    all_tickers = market_client.get_all_tickers()
    if not all_tickers:
        raise RuntimeError("ticker endpoint returned no markets")
    markets = build_markets(exchange_info, all_tickers, cfg)
    if not markets:
        raise RuntimeError("no tradable USD markets found")

    # Probe short inventory even when new shorts are disabled, so pre-existing positions
    # are not silently ignored. An explicit competition permission denial disables only shorts.
    shorts_available = True
    if cfg.paper_mode:
        shorts_available = True
    else:
        probe = market_client.short_positions()
        if probe.get("Success") is True:
            shorts_available = True
        else:
            message = str(probe.get("ErrMsg", "short endpoint probe failed"))
            if "does not allow short positions" in message.lower() or "not allow short" in message.lower():
                shorts_available = False
                logger.warning("shorting is disabled by this competition; continuing long-only")
            elif cfg.orders_enabled:
                raise RuntimeError(f"Could not verify short inventory safely: {message}")
            else:
                shorts_available = False
                logger.warning("short API probe failed; short side disabled in inspection mode: %s", message)

    strategy = BossStrategy(cfg, shorts_available=shorts_available)
    wallet = _refresh_wallet(trader)
    shorts = fetch_short_positions(trader, cfg.paper_mode, all_tickers, markets) if shorts_available else {}
    starting_equity = portfolio_value(wallet, all_tickers, list(shorts.values()))
    if cfg.paper_mode:
        starting_equity = cfg.expected_starting_cash
    db.set_state(cfg.bot_namespace + ":session_start_equity", starting_equity)
    db.set_state(cfg.bot_namespace + ":session_started_at_ms", now_ms())

    day_key = _daily_key()
    day_start = db.get_state(cfg.bot_namespace + ":day_start_equity")
    db_day = db.get_state(cfg.bot_namespace + ":risk_day")
    if db_day != day_key or day_start is None:
        day_start = starting_equity
        db.set_state(cfg.bot_namespace + ":day_start_equity", day_start)
        db.set_state(cfg.bot_namespace + ":risk_day", day_key)
    peak_reference = cfg.expected_starting_cash if not cfg.paper_mode else starting_equity
    peak = float(db.get_state(cfg.bot_namespace + ":peak_equity", peak_reference) or peak_reference)

    risk = RiskManager(starting_equity, daily_loss_limit=cfg.daily_loss_limit_pct, max_drawdown=cfg.max_drawdown_limit_pct)
    risk.peak_equity = max(peak, peak_reference, starting_equity)
    logger.info("starting equity=%.2f tradable_markets=%d shorts_api=%s db=%s", starting_equity, len(markets), shorts_available, cfg.db_path)
    logger.info("turnover cap=%s; order cap=%s; portfolio halt thresholds daily=%s drawdown=%s; short collateral is cash-funded",
                "disabled" if cfg.max_daily_turnover == 0 else f"{starting_equity * cfg.max_daily_turnover:.2f} USD/day",
                "disabled" if cfg.max_order_usd == 0 else f"{cfg.max_order_usd:.2f} USD/order",
                "disabled" if cfg.daily_loss_limit_pct == 0 else f"{cfg.daily_loss_limit_pct:.2%}",
                "disabled" if cfg.max_drawdown_limit_pct == 0 else f"{cfg.max_drawdown_limit_pct:.2%}")

    last_rebalance = int(db.get_state(cfg.bot_namespace + ":last_rebalance_ms", 0) or 0)
    last_prediction_log = int(db.get_state(cfg.bot_namespace + ":last_prediction_log_ms", 0) or 0)
    uncertainty_key = cfg.bot_namespace + ":execution_uncertain"

    while True:
        cycle_start = time.perf_counter()
        ts = now_ms()
        try:
            tickers = market_client.get_all_tickers()
            if not tickers:
                logger.warning("empty ticker response; no trading this cycle")
                time.sleep(cfg.poll_seconds)
                continue
            db.add_market_snapshots(ts, tickers.items())
            if ts - last_exchange_refresh_ms >= cfg.exchange_info_refresh_seconds * 1000:
                refreshed_info = market_client.exchange_info()
                if isinstance(refreshed_info, Mapping) and refreshed_info.get("TradePairs"):
                    exchange_info = refreshed_info
                    last_exchange_refresh_ms = ts
                    logger.info("refreshed exchangeInfo; returned %d pair definitions", len(exchange_info.get("TradePairs", {})))
                else:
                    logger.warning("exchangeInfo refresh failed; retaining last valid market metadata")
            markets = build_markets(exchange_info, tickers, cfg)
            account_refresh_due = (last_account_refresh_ms == 0 or
                                   ts - last_account_refresh_ms >= cfg.account_refresh_seconds * 1000)
            if account_refresh_due:
                wallet = _refresh_wallet(trader)
                if shorts_available:
                    shorts = fetch_short_positions(trader, cfg.paper_mode, tickers, markets)
                else:
                    shorts = {}
                last_account_refresh_ms = ts
            equity = portfolio_value(wallet, tickers, list(shorts.values()))

            current_day = _daily_key()
            if current_day != day_key:
                day_key = current_day
                day_start = equity
                risk.new_day(equity)
                db.set_state(cfg.bot_namespace + ":day_start_equity", equity)
                db.set_state(cfg.bot_namespace + ":risk_day", day_key)
            risk.update(equity, day_start)
            risk.peak_equity = max(risk.peak_equity, equity)
            db.set_state(cfg.bot_namespace + ":peak_equity", risk.peak_equity)
            db.record_equity(ts, equity)
            db.record_api_event(ts, "/v3/ticker", True, f"{len(markets)} tradable USD markets")

            histories = db.histories(list(markets), cfg.history_limit)
            # Feed current signed exposures into the turnover-cost term of the portfolio
            # utility. Shorts are represented by negative collateral weights.
            positions = positions_from_wallet(wallet, markets, db, trader if cfg.paper_mode else None)
            current_weights = {pair: (pos.value / equity if equity > 0 else 0.0)
                               for pair, pos in positions.items()}
            current_weights.update({pair: -(pos.collateral / equity if equity > 0 else 0.0)
                                    for pair, pos in shorts.items()})
            training_histories = None
            if strategy.predictor_training_due(ts, histories):
                training_histories = db.histories(list(markets), cfg.predictor_history_limit)
            result = strategy.analyze(histories, markets, ts,
                                      training_histories=training_histories,
                                      current_weights=current_weights)

            if last_prediction_log == 0 or ts - last_prediction_log >= cfg.prediction_log_interval_seconds * 1000:
                wrote = log_predictions(db, result, markets, cfg, ts)
                db.set_state(cfg.bot_namespace + ":last_prediction_log_ms", ts)
                last_prediction_log = ts
                if wrote:
                    logger.info("predictions logged=%d", wrote)
            db.settle_predictions(ts, {pair: market.price for pair, market in markets.items()}, cfg.predictor_class_threshold)

            due = last_rebalance == 0 or ts - last_rebalance >= cfg.rebalance_seconds * 1000
            executed: list[tuple[str, str, float, float]] = []
            blocked: list[str] = []
            if due:
                latch = db.get_state(uncertainty_key, None)
                if latch:
                    blocked.append(f"EXECUTION HALTED: unresolved ambiguous order {latch}; reconcile manually, then run --clear-execution-halt")
                else:
                    targets, forced, force_reasons = plan_targets(result, positions, equity, risk, db, cfg, markets, histories)
                    short_targets, forced_shorts, short_reasons = plan_short_targets(result, shorts, equity, risk, db, cfg, markets, histories)
                    # Competition does not set a turnover quota. Keep turnover accounting
                    # for reporting, but do not let it suppress otherwise valid target trades.
                    turnover_cap = math.inf
                    turnover_used = db.day_turnover(day_key, cfg.bot_namespace)
                    turnover_remaining = math.inf
                    sells, buys = build_plan(targets, forced, positions, markets, equity, cfg, turnover_remaining, wallet)
                    short_closes, short_opens = build_short_plan(short_targets, forced_shorts, shorts, markets, equity, cfg)
                    normal_remaining = math.inf
                    unsafe = False

                    # 1) Spot sells, with safety exits ahead of regular reductions.
                    for order in sells:
                        pair = order["pair"]
                        market = markets.get(pair)
                        pos = positions.get(pair)
                        if not market or not pos or pos.free_qty <= 0:
                            blocked.append(f"SELL {pair}: no free quantity/market")
                            continue
                        notional = min(float(order["notional"]), pos.free_qty * market.price)
                        if not order["forced"] and math.isfinite(normal_remaining):
                            notional = min(notional, normal_remaining)
                        if notional <= 0 or (not order["forced"] and notional < max(cfg.min_order_usd, market.min_order)):
                            blocked.append(f"SELL {pair}: available turnover budget below minimum")
                            continue
                        qty = min(_round_qty(notional / market.price, market.amount_precision), pos.free_qty)
                        if qty <= 0:
                            blocked.append(f"SELL {pair}: quantity rounds to zero")
                            continue
                        ok, response = execute_order(trader, db, logger_trade, pair, "SELL", qty, cfg, order["reason"], price=market.bid)
                        if ok:
                            d = response.get("OrderDetail", response)
                            fill_qty = float(d.get("FilledQuantity") or qty)
                            fill_price = float(d.get("FilledAverPrice") or d.get("Price") or market.bid)
                            traded = fill_qty * fill_price
                            executed.append(("SELL", pair, traded, fill_price))
                            if math.isfinite(normal_remaining): normal_remaining = max(0.0, normal_remaining - traded)
                        else:
                            blocked.append(f"SELL {pair}: {response.get('ErrMsg', 'API/reconciliation failure')}")
                            if db.get_state(uncertainty_key, None):
                                unsafe = True
                                break

                    # 2) Close short risk before adding any exposure. Forced close may exceed turnover cap.
                    if not unsafe:
                        for order in short_closes:
                            pair = order["pair"]
                            market = markets.get(pair)
                            pos = shorts.get(pair)
                            if not market or not pos:
                                blocked.append(f"SHORT_CLOSE {pair}: missing market/position")
                                continue
                            close_pct = float(order["close_pct"])
                            notional = float(order["notional"])
                            if not order["forced"] and math.isfinite(normal_remaining):
                                notional = min(notional, normal_remaining)
                                close_pct = min(close_pct, 100.0 * notional / max(pos.qty * market.ask, 1e-12))
                            if notional <= 0 or close_pct <= 0 or (not order["forced"] and notional < cfg.min_order_usd):
                                blocked.append(f"SHORT_CLOSE {pair}: available turnover budget below minimum")
                                continue
                            ok, response = execute_short_close(trader, db, logger_trade, cfg, pos, market, close_pct, tickers)
                            if ok:
                                d = response.get("normalized", {}).get("OrderDetail", {})
                                traded = float(d.get("ClosedQty", 0) or pos.qty * close_pct / 100.0) * float(d.get("ClosePrice", market.ask) or market.ask)
                                executed.append(("SHORT_CLOSE", pair, traded, float(d.get("ClosePrice", market.ask) or market.ask)))
                                if math.isfinite(normal_remaining): normal_remaining = max(0.0, normal_remaining - traded)
                            else:
                                blocked.append(f"SHORT_CLOSE {pair}: {response.get('ErrMsg', 'API/reconciliation failure')}")
                                if "does not allow short positions" in str(response.get("ErrMsg", "")).lower():
                                    strategy.shorts_available = False
                                    shorts_available = False
                                if db.get_state(uncertainty_key, None):
                                    unsafe = True
                                    break

                    # Refresh spendable cash and open-short inventory after all exits.
                    if not unsafe:
                        wallet = _refresh_wallet(trader)
                        if shorts_available:
                            shorts = fetch_short_positions(trader, cfg.paper_mode, tickers, markets)
                        else:
                            shorts = {}
                        current_usd = free_usd(wallet)
                        reserve = equity * cfg.cash_reserve

                        # Reserve projected long-entry cash first; short collateral cannot be borrowed
                        # from the cash reserve or spent twice across the two books.
                        projected_long_cost = sum(_order_cap(float(o["notional"]), cfg) * (1 + cfg.commission_rate)
                                                  for o in buys)
                        short_budget = max(0.0, current_usd - reserve - projected_long_cost)
                        if math.isfinite(normal_remaining):
                            short_budget = min(short_budget, normal_remaining)

                        # 3) Add short collateral only from cash not reserved for intended spot buys.
                        if shorts_available and cfg.shorts_enabled:
                            for order in short_opens:
                                pair = order["pair"]
                                market = markets.get(pair)
                                if not market:
                                    continue
                                collateral = min(_order_cap(float(order["collateral"]), cfg), short_budget / (1 + cfg.commission_rate))
                                if collateral < max(1.0, cfg.min_order_usd):
                                    blocked.append(f"SHORT_OPEN {pair}: cash reserve/long orders leave insufficient collateral")
                                    continue
                                if math.isfinite(normal_remaining):
                                    collateral = min(collateral, normal_remaining)
                                if collateral < max(1.0, cfg.min_order_usd):
                                    blocked.append(f"SHORT_OPEN {pair}: turnover budget below minimum")
                                    continue
                                ok, response = execute_short_open(trader, db, logger_trade, cfg, pair, collateral, market, shorts.get(pair), tickers)
                                if ok:
                                    d = response.get("normalized", {}).get("OrderDetail", {})
                                    qty = float(d.get("ShortQty", 0) or collateral / market.bid)
                                    price = float(d.get("EntryPrice", market.bid) or market.bid)
                                    executed.append(("SHORT_OPEN", pair, qty * price, price))
                                    short_budget = max(0.0, short_budget - collateral * (1 + cfg.commission_rate))
                                    if math.isfinite(normal_remaining): normal_remaining = max(0.0, normal_remaining - qty * price)
                                else:
                                    message = str(response.get("ErrMsg", "API/reconciliation failure"))
                                    blocked.append(f"SHORT_OPEN {pair}: {message}")
                                    if "does not allow short positions" in message.lower():
                                        strategy.shorts_available = False
                                        shorts_available = False
                                        break
                                    if db.get_state(uncertainty_key, None):
                                        unsafe = True
                                        break

                        # 4) Long buys last; each order is cash- and (optional) turnover-bounded.
                        if not unsafe:
                            wallet = _refresh_wallet(trader)
                            current_usd = free_usd(wallet)
                            for order in buys:
                                pair = order["pair"]
                                market = markets.get(pair)
                                if market is None:
                                    continue
                                if math.isfinite(normal_remaining) and normal_remaining <= 0:
                                    blocked.append(f"BUY {pair}: turnover exhausted")
                                    continue
                                max_cash = max(0.0, current_usd - reserve)
                                notional = min(_order_cap(float(order["notional"]), cfg),
                                               normal_remaining if math.isfinite(normal_remaining) else math.inf,
                                               max_cash / max(1 + cfg.commission_rate, 1e-9))
                                minimum = max(cfg.min_order_usd, market.min_order)
                                if notional < minimum:
                                    blocked.append(f"BUY {pair}: cash/budget below minimum")
                                    continue
                                qty = _round_qty(notional / market.ask, market.amount_precision)
                                actual = qty * market.ask
                                if qty <= 0 or actual < minimum:
                                    blocked.append(f"BUY {pair}: rounded order below minimum")
                                    continue
                                ok, response = execute_order(trader, db, logger_trade, pair, "BUY", qty, cfg, order["reason"], price=market.ask)
                                if ok:
                                    d = response.get("OrderDetail", response)
                                    fill_qty = float(d.get("FilledQuantity") or qty)
                                    fill_price = float(d.get("FilledAverPrice") or d.get("Price") or market.ask)
                                    traded = fill_qty * fill_price
                                    executed.append(("BUY", pair, traded, fill_price))
                                    if math.isfinite(normal_remaining): normal_remaining = max(0.0, normal_remaining - traded)
                                    current_usd = max(0.0, current_usd - traded * (1 + cfg.commission_rate))
                                else:
                                    blocked.append(f"BUY {pair}: {response.get('ErrMsg', 'API/reconciliation failure')}")
                                    if db.get_state(uncertainty_key, None):
                                        unsafe = True
                                        break

                    if not unsafe:
                        last_rebalance = ts
                        db.set_state(cfg.bot_namespace + ":last_rebalance_ms", ts)
                    else:
                        blocked.append("all subsequent orders halted because execution state is uncertain")

            # Re-read the actual state after execution for accurate equity/dashboard output.
            wallet = _refresh_wallet(trader)
            if shorts_available:
                shorts = fetch_short_positions(trader, cfg.paper_mode, tickers, markets)
            else:
                shorts = {}
            last_account_refresh_ms = ts
            positions = positions_from_wallet(wallet, markets, db, trader if cfg.paper_mode else None)
            equity = portfolio_value(wallet, tickers, list(shorts.values()))
            turnover_used = db.day_turnover(day_key, cfg.bot_namespace)
            turnover_cap = math.inf
            next_rebalance = last_rebalance + cfg.rebalance_seconds * 1000 if last_rebalance else ts
            print_dashboard(cfg, equity, starting_equity, wallet, markets, result, positions,
                            turnover_used, turnover_cap, next_rebalance, executed, blocked, risk, shorts)
            # Keep durable equity/performance history for later review. Snapshots are
            # written after reconciliation so the record reflects the latest known state.
            logger_trade.snapshot(equity, extra={"short_positions": len(shorts), "markets": len(markets)})

            if risk.snapshot.halted:
                logger.warning("RISK LOCK active; no new risk is added; risk exits remain prioritized")
            elapsed = time.perf_counter() - cycle_start
            time.sleep(max(0.5, cfg.poll_seconds - elapsed))

        except KeyboardInterrupt:
            print("\n[STOP] user interrupted; no automatic liquidation is performed")
            db.set_state(cfg.bot_namespace + ":stopped_at_ms", now_ms())
            db.close()
            return
        except Exception as exc:
            logger.exception("cycle failure: %s", exc)
            db.record_api_event(ts, "engine", False, str(exc)[:500])
            time.sleep(cfg.poll_seconds)


if __name__ == "__main__":
    run()
