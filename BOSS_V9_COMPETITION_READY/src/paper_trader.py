"""Local spot + collateralized-short simulator; it never submits exchange orders."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

TAKER_FEE = 0.001
MAKER_FEE = 0.0005
SHORT_FEE = 0.001  # Competition API documents a 0.1% fee on short open and close


@dataclass
class Position:
    coin: str
    qty: float
    avg_entry: float
    create_timestamp: int


@dataclass
class PaperShort:
    pair: str
    qty: float
    entry_price: float
    collateral: float
    create_timestamp: int


class PaperTrader:
    def __init__(self, starting_usd: float = 100_000.0):
        self.starting_usd = float(starting_usd)
        self.usd_balance = float(starting_usd)
        self._positions: dict[str, Position] = {}
        self._shorts: dict[str, PaperShort] = {}
        self._orders: list[dict] = []
        self._order_id_counter = 1

    def get_usd_balance(self) -> float:
        return self.usd_balance

    def get_coin_balance(self, coin: str) -> float:
        position = self._positions.get(coin.upper())
        return position.qty if position else 0.0

    def balance(self) -> dict:
        wallet = {"USD": {"Free": round(self.usd_balance, 8), "Lock": 0.0}}
        for coin, pos in self._positions.items():
            if pos.qty > 0:
                wallet[coin] = {"Free": round(pos.qty, 8), "Lock": 0.0}
        return {"Success": True, "ErrMsg": "", "SpotWallet": wallet}

    def place_order(self, pair: str, side: str, quantity: float, order_type: str = "MARKET",
                    price: Optional[float] = None) -> dict:
        if price is None or not math.isfinite(float(price)) or price <= 0:
            return {"Success": False, "ErrMsg": "price required for paper trading"}
        if not math.isfinite(float(quantity)) or quantity <= 0:
            return {"Success": False, "ErrMsg": "quantity must be positive"}

        coin = pair.split("/")[0].upper()
        side, order_type = side.upper(), order_type.upper()
        if side not in {"BUY", "SELL"}:
            return {"Success": False, "ErrMsg": f"unknown spot side: {side}"}
        fee_rate = TAKER_FEE if order_type == "MARKET" else MAKER_FEE
        order_value = float(quantity) * float(price)
        fee = order_value * fee_rate
        ts = int(time.time() * 1000)

        if side == "BUY":
            total_cost = order_value + fee
            if total_cost > self.usd_balance + 1e-8:
                return {"Success": False, "ErrMsg": f"insufficient paper USD (need ${total_cost:.2f}, have ${self.usd_balance:.2f})"}
            self.usd_balance -= total_cost
            existing = self._positions.get(coin)
            if existing:
                qty_sum = existing.qty + quantity
                self._positions[coin] = Position(coin, qty_sum, (existing.qty * existing.avg_entry + quantity * price) / qty_sum, existing.create_timestamp)
            else:
                self._positions[coin] = Position(coin, float(quantity), float(price), ts)
        else:
            existing = self._positions.get(coin)
            if not existing or existing.qty + 1e-12 < quantity:
                return {"Success": False, "ErrMsg": f"insufficient {coin} for paper SELL"}
            self.usd_balance += order_value - fee
            existing.qty -= quantity
            if existing.qty <= 1e-10:
                del self._positions[coin]

        detail = {
            "Pair": pair, "OrderID": self._next_order_id(), "Status": "FILLED", "Role": "TAKER",
            "CreateTimestamp": ts, "FinishTimestamp": ts, "Side": side, "Type": order_type,
            "Price": float(price), "Quantity": float(quantity), "FilledQuantity": float(quantity),
            "FilledAverPrice": float(price), "CommissionChargeValue": fee, "CommissionPercent": fee_rate,
        }
        self._orders.append(detail)
        return {"Success": True, "OrderDetail": detail}

    def short_open(self, pair: str, collateral: float, price: Optional[float] = None) -> dict:
        if price is None or not math.isfinite(float(price)) or price <= 0:
            return {"Success": False, "ErrMsg": "price required for paper short"}
        collateral = float(collateral)
        if collateral < 1.0 or not math.isfinite(collateral):
            return {"Success": False, "ErrMsg": "minimum collateral is $1"}
        quantity = math.floor((collateral / price) * 1e8 + 1e-8) / 1e8
        if quantity <= 0:
            return {"Success": False, "ErrMsg": "short quantity rounds to zero"}
        position_value = quantity * price
        fee = position_value * SHORT_FEE
        if collateral + fee > self.usd_balance + 1e-8:
            return {"Success": False, "ErrMsg": "insufficient paper USD for short collateral and fee"}
        self.usd_balance -= collateral + fee
        ts = int(time.time() * 1000)
        existing = self._shorts.get(pair)
        if existing:
            qty_sum = existing.qty + quantity
            existing.entry_price = (existing.qty * existing.entry_price + quantity * price) / qty_sum
            existing.qty = qty_sum
            existing.collateral += collateral
        else:
            self._shorts[pair] = PaperShort(pair, quantity, float(price), collateral, ts)

        detail = {
            "ID": self._next_order_id(), "Pair": pair, "Side": "SHORT_OPEN", "OrderType": "MARKET",
            "EntryPrice": float(price), "ShortQty": quantity, "Collateral": collateral,
            "OpenFee": fee, "Status": "OPEN", "CreateTimestamp": ts,
        }
        self._orders.append(detail)
        return {"Success": True, **detail}

    def short_close(self, pair: str, close_qty: Optional[float] = None, close_pct: Optional[float] = None,
                    price: Optional[float] = None) -> dict:
        position = self._shorts.get(pair)
        if position is None:
            return {"Success": False, "ErrMsg": "no open short position for this pair"}
        if price is None or not math.isfinite(float(price)) or price <= 0:
            return {"Success": False, "ErrMsg": "price required for paper short close"}
        if close_qty is not None:
            quantity = min(position.qty, max(0.0, float(close_qty)))
        elif close_pct is not None:
            if not (0 < float(close_pct) <= 100):
                return {"Success": False, "ErrMsg": "close_pct must be >0 and <=100"}
            quantity = position.qty * float(close_pct) / 100.0
        else:
            quantity = position.qty
        if quantity <= 0:
            return {"Success": False, "ErrMsg": "close quantity must be positive"}

        collateral_part = position.collateral * min(1.0, quantity / position.qty)
        if position.qty - quantity <= 1e-8:
            quantity = position.qty
            collateral_part = position.collateral
        pnl = quantity * (position.entry_price - price)
        pnl = max(-collateral_part, pnl)  # mirrors the simulator's collateral loss floor
        close_fee = quantity * price * SHORT_FEE
        returned = collateral_part + pnl - close_fee
        self.usd_balance += returned
        position.qty -= quantity
        position.collateral -= collateral_part
        fully_closed = position.qty <= 1e-8 or position.collateral <= 1e-8
        if fully_closed:
            del self._shorts[pair]
        result = {
            "Success": True, "Pair": pair, "Side": "SHORT_CLOSE", "ClosePrice": float(price),
            "RealizedPNL": pnl, "CloseFee": close_fee, "ReturnAmount": returned,
            "ClosedQty": quantity, "FullyClosed": fully_closed,
        }
        if not fully_closed:
            result["RemainingQty"] = position.qty
            result["RemainingCollateral"] = position.collateral
        self._orders.append({**result, "CreateTimestamp": int(time.time() * 1000)})
        return result

    def short_positions(self, tickers: Optional[dict] = None) -> dict:
        tickers = tickers or {}
        result = []
        for pair, position in self._shorts.items():
            price = float(tickers.get(pair, {}).get("MinAsk", tickers.get(pair, {}).get("LastPrice", position.entry_price)) or position.entry_price)
            unrealized = max(-position.collateral, position.qty * (position.entry_price - price))
            result.append({
                "ID": 0, "Pair": pair, "EntryPrice": position.entry_price, "ShortQty": position.qty,
                "Collateral": position.collateral, "CurrentPrice": price, "UnrealizedPNL": unrealized,
                "UnrealizedPNLPct": unrealized / position.collateral if position.collateral else 0.0,
                "PositionValue": max(0.0, position.collateral + unrealized),
                "CreateTimestamp": position.create_timestamp, "PositionStatus": "OPEN",
            })
        return {"Success": True, "Positions": result}

    def portfolio_value(self, tickers: dict) -> float:
        value = self.usd_balance
        for coin, position in self._positions.items():
            pair = f"{coin}/USD"
            value += position.qty * float(tickers.get(pair, {}).get("LastPrice", 0.0) or 0.0)
        shorts = self.short_positions(tickers).get("Positions", [])
        value += sum(float(item.get("PositionValue", 0.0) or 0.0) for item in shorts)
        return value

    def positions_summary(self, tickers: dict) -> list[dict]:
        result = []
        for coin, position in self._positions.items():
            price = float(tickers.get(f"{coin}/USD", {}).get("LastPrice", 0.0) or 0.0)
            value = position.qty * price
            cost = position.qty * position.avg_entry
            result.append({"coin": coin, "qty": position.qty, "avg_entry": position.avg_entry,
                           "current_price": price, "current_value": value, "unrealized_pnl": value - cost,
                           "unrealized_pct": (value - cost) / cost * 100 if cost else 0.0})
        return sorted(result, key=lambda item: item["current_value"], reverse=True)

    def order_history(self) -> list[dict]:
        return list(reversed(self._orders))

    def reset(self):
        self.usd_balance = self.starting_usd
        self._positions.clear()
        self._shorts.clear()
        self._orders.clear()
        self._order_id_counter = 1

    def _next_order_id(self) -> int:
        value = self._order_id_counter
        self._order_id_counter += 1
        return value
