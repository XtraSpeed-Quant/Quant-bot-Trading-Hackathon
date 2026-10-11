from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prices (
    timestamp_ms INTEGER NOT NULL,
    pair TEXT NOT NULL,
    price REAL NOT NULL,
    bid REAL NOT NULL,
    ask REAL NOT NULL,
    change_24h REAL NOT NULL,
    unit_trade_value REAL NOT NULL,
    spread_bps REAL NOT NULL,
    PRIMARY KEY(timestamp_ms, pair)
);
CREATE INDEX IF NOT EXISTS idx_prices_pair_ts ON prices(pair, timestamp_ms);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    fee REAL NOT NULL,
    mode TEXT NOT NULL,
    order_id TEXT,
    status TEXT NOT NULL,
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_trades_pair_ts ON trades(pair, timestamp_ms);
CREATE TABLE IF NOT EXISTS equity (timestamp_ms INTEGER PRIMARY KEY, equity REAL NOT NULL);
CREATE TABLE IF NOT EXISTS api_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    success INTEGER NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_events_ts ON api_events(timestamp_ms);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_ms INTEGER NOT NULL,
    pair TEXT NOT NULL,
    price REAL NOT NULL,
    target_time_ms INTEGER NOT NULL,
    predicted_return REAL NOT NULL,
    probability_up REAL NOT NULL,
    probability_flat REAL NOT NULL,
    probability_down REAL NOT NULL,
    confidence REAL NOT NULL,
    model_enabled INTEGER NOT NULL,
    realized_return REAL,
    realized_state INTEGER,
    settled INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_time_ms, settled);
CREATE INDEX IF NOT EXISTS idx_predictions_pair_ts ON predictions(pair, timestamp_ms);
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _day_start_ms(day_key: str | None = None) -> int:
    if day_key:
        dt = datetime.strptime(day_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        now = datetime.now(timezone.utc)
        dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp() * 1000)


class BotDB:
    def __init__(self, path: str = "data/boss_v7_live.sqlite3", history_limit: int = 2000):
        self.path = str(path)
        self.history_limit = int(history_limit)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.session_id = str(int(time.time()))
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
        if "session_id" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN session_id TEXT")
        conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def key(self, namespace: str, name: str) -> str:
        return f"{namespace}:{name}"

    def get_state(self, key: str, default=None):
        row = self._conn().execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return row[0]

    def set_state(self, key: str, value) -> None:
        encoded = json.dumps(value, separators=(",", ":"))
        self._conn().execute(
            "INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, encoded)
        )
        self._conn().commit()

    def delete_state(self, key: str) -> None:
        self._conn().execute("DELETE FROM state WHERE key=?", (key,))
        self._conn().commit()

    def add_market_snapshots(self, timestamp_ms: int, snapshots: Iterable[Mapping]) -> None:
        rows = []
        for pair, t in snapshots:
            price = float(t.get("LastPrice", 0) or 0)
            bid = float(t.get("MaxBid", 0) or 0)
            ask = float(t.get("MinAsk", 0) or 0)
            if price <= 0:
                continue
            spread = ((ask - bid) / price * 10000.0) if ask > 0 and bid > 0 else 0.0
            rows.append((
                int(timestamp_ms), str(pair), price, bid, ask,
                float(t.get("Change", 0) or 0), float(t.get("UnitTradeValue", 0) or 0), round(spread, 6),
            ))
        if rows:
            self._conn().executemany(
                "INSERT OR IGNORE INTO prices(timestamp_ms,pair,price,bid,ask,change_24h,unit_trade_value,spread_bps) VALUES(?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn().commit()

    def histories(self, pairs: list[str], limit: int | None = None) -> dict[str, list[float]]:
        if not pairs:
            return {}
        limit = int(limit or self.history_limit)
        placeholders = ",".join("?" for _ in pairs)
        sql = f"""
            SELECT pair, price, timestamp_ms FROM (
                SELECT pair, price, timestamp_ms,
                       ROW_NUMBER() OVER (PARTITION BY pair ORDER BY timestamp_ms DESC) AS rn
                FROM prices
                WHERE pair IN ({placeholders})
            ) WHERE rn <= ?
            ORDER BY pair ASC, timestamp_ms ASC
        """
        rows = self._conn().execute(sql, [*pairs, limit]).fetchall()
        result = {p: [] for p in pairs}
        for row in rows:
            result[row["pair"]].append(float(row["price"]))
        return result

    def history_summary(self, pairs: list[str] | None = None) -> dict[str, int | None]:
        """Return coverage statistics for the selected market history without mutating it."""
        if pairs is None:
            row = self._conn().execute(
                "SELECT COUNT(*), COUNT(DISTINCT timestamp_ms), COUNT(DISTINCT pair), "
                "MIN(timestamp_ms), MAX(timestamp_ms) FROM prices"
            ).fetchone()
        elif not pairs:
            return {"rows": 0, "snapshots": 0, "pairs": 0, "first_timestamp_ms": None, "last_timestamp_ms": None}
        else:
            placeholders = ",".join("?" for _ in pairs)
            row = self._conn().execute(
                f"SELECT COUNT(*), COUNT(DISTINCT timestamp_ms), COUNT(DISTINCT pair), "
                f"MIN(timestamp_ms), MAX(timestamp_ms) FROM prices WHERE pair IN ({placeholders})",
                list(pairs),
            ).fetchone()
        return {
            "rows": int(row[0] or 0),
            "snapshots": int(row[1] or 0),
            "pairs": int(row[2] or 0),
            "first_timestamp_ms": int(row[3]) if row[3] is not None else None,
            "last_timestamp_ms": int(row[4]) if row[4] is not None else None,
        }

    def latest_price_timestamp(self, pair: str) -> int | None:
        row = self._conn().execute(
            "SELECT timestamp_ms FROM prices WHERE pair=? ORDER BY timestamp_ms DESC LIMIT 1", (pair,)
        ).fetchone()
        return int(row[0]) if row else None

    def record_trade(self, timestamp_ms: int, pair: str, side: str, quantity: float, price: float,
                     fee: float, mode: str, order_id: str | None, status: str) -> None:
        self._conn().execute(
            """INSERT INTO trades(timestamp_ms,pair,side,quantity,price,notional,fee,mode,order_id,status,session_id)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (int(timestamp_ms), pair, side.upper(), float(quantity), float(price),
             float(quantity) * float(price), float(fee), mode, order_id, status, self.session_id),
        )
        self._conn().commit()

    def record_trade_from_order(self, result: Mapping, mode: str, fallback_timestamp_ms: int | None = None) -> bool:
        d = result.get("OrderDetail", result)
        if not isinstance(d, Mapping):
            return False
        pair = str(d.get("Pair", ""))
        side = str(d.get("Side", ""))
        qty = float(d.get("FilledQuantity") or d.get("Quantity") or d.get("ShortQty") or d.get("ClosedQty") or 0)
        price = float(d.get("FilledAverPrice") or d.get("Price") or d.get("EntryPrice") or d.get("ClosePrice") or 0)
        fee = float(d.get("CommissionChargeValue") or d.get("OpenFee") or d.get("CloseFee") or 0)
        order_id = str(d.get("OrderID") or d.get("ID") or "")
        status = str(d.get("Status", "FILLED"))
        ts = int(d.get("CreateTimestamp") or fallback_timestamp_ms or _now_ms())
        if not pair or not side or qty <= 0 or price <= 0:
            return False
        self.record_trade(ts, pair, side, qty, price, fee, mode, order_id, status)
        return True

    def record_equity(self, timestamp_ms: int, equity: float) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO equity(timestamp_ms,equity) VALUES(?,?)",
            (int(timestamp_ms), float(equity)),
        )
        self._conn().commit()

    def record_api_event(self, timestamp_ms: int, endpoint: str, success: bool, message: str = "") -> None:
        self._conn().execute(
            "INSERT INTO api_events(timestamp_ms,endpoint,success,message) VALUES(?,?,?,?)",
            (int(timestamp_ms), endpoint, int(bool(success)), message),
        )
        self._conn().commit()

    def _raw_day_turnover(self, day_key: str | None = None) -> float:
        start = _day_start_ms(day_key)
        row = self._conn().execute(
            "SELECT COALESCE(SUM(notional),0) FROM trades WHERE timestamp_ms>=? AND side IN ('BUY','SELL','SHORT_OPEN','SHORT_CLOSE')",
            (start,),
        ).fetchone()
        return float(row[0] or 0)

    def reset_day_turnover(self, namespace: str, day_key: str | None = None) -> tuple[float, float]:
        day_key = day_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = self._raw_day_turnover(day_key)
        self.set_state(self.key(namespace, "turnover_reset"), {
            "day": day_key,
            "baseline": raw,
            "reset_at_ms": _now_ms(),
        })
        return raw, 0.0

    def day_turnover(self, day_key: str | None = None, namespace: str | None = None) -> float:
        day_key = day_key or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        raw = self._raw_day_turnover(day_key)
        if namespace:
            reset = self.get_state(self.key(namespace, "turnover_reset"), None)
            if isinstance(reset, dict) and reset.get("day") == day_key:
                baseline = max(0.0, float(reset.get("baseline", 0.0) or 0.0))
                return max(0.0, raw - baseline)
        return raw

    def daily_trade_counts(self, day_key: str | None = None) -> dict[str, int]:
        start = _day_start_ms(day_key)
        rows = self._conn().execute(
            "SELECT side,COUNT(*) c FROM trades WHERE timestamp_ms>=? GROUP BY side", (start,)
        ).fetchall()
        return {str(r[0]): int(r[1]) for r in rows}

    def recent_trade_timestamp(self, pair: str) -> int | None:
        row = self._conn().execute(
            "SELECT timestamp_ms FROM trades WHERE pair=? ORDER BY timestamp_ms DESC LIMIT 1", (pair,)
        ).fetchone()
        return int(row[0]) if row else None

    def record_prediction(self, timestamp_ms: int, pair: str, price: float, target_time_ms: int,
                          predicted_return: float, probability_up: float, probability_flat: float,
                          probability_down: float, confidence: float, model_enabled: bool) -> None:
        self._conn().execute(
            """INSERT INTO predictions(timestamp_ms,pair,price,target_time_ms,predicted_return,
               probability_up,probability_flat,probability_down,confidence,model_enabled)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (int(timestamp_ms), pair, float(price), int(target_time_ms), float(predicted_return),
             float(probability_up), float(probability_flat), float(probability_down), float(confidence),
             int(bool(model_enabled))),
        )
        self._conn().commit()

    def settle_predictions(self, now_ms: int, current_prices: Mapping[str, float], class_threshold: float) -> int:
        rows = self._conn().execute(
            "SELECT id,pair,price,predicted_return,target_time_ms FROM predictions WHERE settled=0 AND target_time_ms<=? ORDER BY target_time_ms LIMIT 5000",
            (int(now_ms),),
        ).fetchall()
        updates = []
        for row in rows:
            current = float(current_prices.get(row["pair"], 0) or 0)
            if current <= 0 or row["price"] <= 0:
                continue
            realized = current / float(row["price"]) - 1.0
            state = 1 if realized > class_threshold else -1 if realized < -class_threshold else 0
            updates.append((realized, state, row["id"]))
        if updates:
            self._conn().executemany(
                "UPDATE predictions SET realized_return=?,realized_state=?,settled=1 WHERE id=?", updates
            )
            self._conn().commit()
        return len(updates)

    def unsettled_prediction_pairs(self, now_ms: int | None = None) -> set[str]:
        ts = int(now_ms or _now_ms())
        rows = self._conn().execute(
            "SELECT DISTINCT pair FROM predictions WHERE settled=0 AND target_time_ms>?", (ts,)
        ).fetchall()
        return {str(r[0]) for r in rows}

    def prediction_metrics(self, model_enabled: bool | None = None) -> dict[str, float | int]:
        args = []
        where = "WHERE settled=1"
        if model_enabled is not None:
            where += " AND model_enabled=?"
            args.append(int(model_enabled))
        rows = self._conn().execute(
            f"SELECT predicted_return,realized_return FROM predictions {where} ORDER BY target_time_ms", args
        ).fetchall()
        if not rows:
            return {"n": 0, "directional_accuracy": 0.0, "correlation": 0.0, "mean_realized": 0.0}
        import numpy as np
        pred = np.asarray([float(r[0]) for r in rows], dtype=float)
        real = np.asarray([float(r[1]) for r in rows], dtype=float)
        pa = np.where(pred >= 0, 1, -1)
        ra = np.where(real >= 0, 1, -1)
        corr = 0.0 if np.std(pred) <= 1e-12 or np.std(real) <= 1e-12 else float(np.corrcoef(pred, real)[0, 1])
        return {
            "n": int(len(rows)),
            "directional_accuracy": float(np.mean(pa == ra)),
            "correlation": corr,
            "mean_realized": float(np.mean(real)),
        }

    def stats(self) -> dict[str, int | float | None]:
        conn = self._conn()
        counts = {}
        for table in ("prices", "trades", "equity", "api_events", "predictions"):
            counts[f"{table}_rows"] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        row = conn.execute("SELECT equity FROM equity ORDER BY timestamp_ms DESC LIMIT 1").fetchone()
        counts["latest_equity"] = float(row[0]) if row else None
        return counts

    def position_cost_basis(self) -> dict[str, dict[str, float]]:
        rows = self._conn().execute(
            "SELECT pair,side,quantity,price,fee,timestamp_ms FROM trades WHERE status='FILLED' AND side IN ('BUY','SELL') ORDER BY timestamp_ms,id"
        ).fetchall()
        state: dict[str, dict[str, float]] = {}
        for row in rows:
            pair = str(row["pair"])
            qty = float(row["quantity"])
            price = float(row["price"])
            fee = float(row["fee"])
            s = state.setdefault(pair, {"qty": 0.0, "cost": 0.0, "entry_ts": 0.0})
            if row["side"] == "BUY":
                s["cost"] += qty * price + fee
                s["qty"] += qty
                if s["entry_ts"] <= 0:
                    s["entry_ts"] = float(row["timestamp_ms"])
            else:
                sell_qty = min(qty, max(s["qty"], 0.0))
                avg = s["cost"] / s["qty"] if s["qty"] > 0 else 0.0
                s["cost"] = max(0.0, s["cost"] - sell_qty * avg)
                s["qty"] = max(0.0, s["qty"] - sell_qty)
                if s["qty"] <= 1e-12:
                    s["entry_ts"] = 0.0
                    s["cost"] = 0.0
        return state
