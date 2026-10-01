from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping


class Store:
    def __init__(self, path: str, history_limit: int = 4320) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history_limit = history_limit
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.initialize()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS prices (
                timestamp_ms INTEGER NOT NULL, pair TEXT NOT NULL, price REAL NOT NULL,
                bid REAL NOT NULL, ask REAL NOT NULL, change_24h REAL NOT NULL,
                unit_trade_value REAL NOT NULL, spread_bps REAL NOT NULL,
                PRIMARY KEY(timestamp_ms, pair)
            );
            CREATE INDEX IF NOT EXISTS idx_prices_pair_ts ON prices(pair, timestamp_ms);
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL,
                pair TEXT NOT NULL, side TEXT NOT NULL, quantity REAL NOT NULL,
                price REAL NOT NULL, notional REAL NOT NULL, fee REAL NOT NULL,
                mode TEXT NOT NULL, order_id TEXT, status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp_ms);
            CREATE TABLE IF NOT EXISTS equity (timestamp_ms INTEGER PRIMARY KEY, equity REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS api_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL,
                endpoint TEXT NOT NULL, success INTEGER NOT NULL, message TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp_ms INTEGER NOT NULL,
                pair TEXT NOT NULL, price REAL NOT NULL, target_time_ms INTEGER NOT NULL,
                predicted_return REAL NOT NULL, probability_up REAL NOT NULL,
                probability_flat REAL NOT NULL, probability_down REAL NOT NULL,
                confidence REAL NOT NULL, model_enabled INTEGER NOT NULL,
                realized_return REAL, realized_state INTEGER, settled INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target_time_ms, settled);
            CREATE INDEX IF NOT EXISTS idx_predictions_pair ON predictions(pair, timestamp_ms);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def key(self, namespace: str, name: str) -> str:
        return f"{namespace}:{name}"

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_state(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, separators=(",", ":"))),
        )
        self.conn.commit()

    def add_market_snapshots(self, timestamp_ms: int, snapshots: Iterable[Mapping[str, Any]]) -> None:
        rows = []
        for s in snapshots:
            price, bid, ask = float(s["price"]), float(s["bid"]), float(s["ask"])
            if price <= 0 or bid <= 0 or ask <= 0:
                continue
            rows.append((timestamp_ms, str(s["pair"]), price, bid, ask, float(s["change_24h"]), float(s["quote_volume_24h"]), float(s["spread_bps"])))
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices(timestamp_ms,pair,price,bid,ask,change_24h,unit_trade_value,spread_bps) VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def histories(self, pairs: list[str], limit: int) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        actual_limit = min(limit, self.history_limit)
        for pair in pairs:
            rows = self.conn.execute(
                "SELECT price FROM prices WHERE pair=? ORDER BY timestamp_ms DESC LIMIT ?",
                (pair, actual_limit),
            ).fetchall()
            result[pair] = [float(r["price"]) for r in reversed(rows)]
        return result

    def latest_price_timestamp(self, pair: str) -> int | None:
        row = self.conn.execute("SELECT MAX(timestamp_ms) AS ts FROM prices WHERE pair=?", (pair,)).fetchone()
        return None if row is None or row["ts"] is None else int(row["ts"])

    def record_trade(self, timestamp_ms: int, pair: str, side: str, quantity: float, price: float, fee: float, mode: str, order_id: str | None, status: str) -> None:
        self.conn.execute(
            "INSERT INTO trades(timestamp_ms,pair,side,quantity,price,notional,fee,mode,order_id,status) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (timestamp_ms, pair, side, float(quantity), float(price), float(quantity) * float(price), float(fee), mode, order_id, status),
        )
        self.conn.commit()

    def record_equity(self, timestamp_ms: int, equity: float) -> None:
        self.conn.execute("INSERT OR REPLACE INTO equity(timestamp_ms,equity) VALUES(?,?)", (timestamp_ms, float(equity)))
        self.conn.commit()

    def record_api_event(self, timestamp_ms: int, endpoint: str, success: bool, message: str) -> None:
        self.conn.execute("INSERT INTO api_events(timestamp_ms,endpoint,success,message) VALUES(?,?,?,?)", (timestamp_ms, endpoint, 1 if success else 0, message[:500]))
        self.conn.commit()

    def day_turnover(self, day_key: str) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(notional),0) AS turnover FROM trades WHERE date(timestamp_ms/1000,'unixepoch')=? AND status IN ('FILLED','PARTIALLY_FILLED')",
            (day_key,),
        ).fetchone()
        return float(row["turnover"] or 0.0)

    def daily_trade_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT date(timestamp_ms/1000,'unixepoch') AS day, COUNT(*) AS count FROM trades WHERE status IN ('FILLED','PARTIALLY_FILLED') GROUP BY day ORDER BY day"
        ).fetchall()
        return {str(r["day"]): int(r["count"]) for r in rows}

    def recent_trade_timestamp(self, pair: str) -> int | None:
        row = self.conn.execute("SELECT MAX(timestamp_ms) AS ts FROM trades WHERE pair=? AND status IN ('FILLED','PARTIALLY_FILLED')", (pair,)).fetchone()
        return None if row is None or row["ts"] is None else int(row["ts"])

    def record_prediction(self, timestamp_ms: int, pair: str, price: float, target_time_ms: int, predicted_return: float, probability_up: float, probability_flat: float, probability_down: float, confidence: float, model_enabled: bool) -> None:
        self.conn.execute(
            "INSERT INTO predictions(timestamp_ms,pair,price,target_time_ms,predicted_return,probability_up,probability_flat,probability_down,confidence,model_enabled,settled) VALUES(?,?,?,?,?,?,?,?,?,?,0)",
            (timestamp_ms, pair, float(price), int(target_time_ms), float(predicted_return), float(probability_up), float(probability_flat), float(probability_down), float(confidence), 1 if model_enabled else 0),
        )
        self.conn.commit()

    def settle_predictions(self, now_ms: int, current_prices: Mapping[str, float], class_threshold: float) -> int:
        rows = self.conn.execute(
            "SELECT id,pair,price,predicted_return FROM predictions WHERE settled=0 AND target_time_ms<=?",
            (int(now_ms),),
        ).fetchall()
        settled = 0
        for row in rows:
            pair = str(row["pair"])
            current = current_prices.get(pair)
            if current is None or float(row["price"]) <= 0:
                continue
            realized = float(current) / float(row["price"]) - 1.0
            cls = 1 if realized > class_threshold else -1 if realized < -class_threshold else 0
            self.conn.execute("UPDATE predictions SET realized_return=?,realized_state=?,settled=1 WHERE id=?", (realized, cls, int(row["id"])))
            settled += 1
        if settled:
            self.conn.commit()
        return settled

    def prediction_metrics(self) -> dict[str, float | int]:
        rows = self.conn.execute(
            "SELECT predicted_return,realized_return,predicted_return,probability_up,probability_down FROM predictions WHERE settled=1 AND realized_return IS NOT NULL ORDER BY timestamp_ms"
        ).fetchall()
        if not rows:
            return {"samples": 0, "directional_accuracy": 0.0, "correlation": 0.0}
        pred = [float(r[0]) for r in rows]
        real = [float(r[1]) for r in rows]
        import numpy as np
        pa, ra = np.asarray(pred), np.asarray(real)
        corr = float(np.corrcoef(pa, ra)[0, 1]) if len(pa) > 1 and np.std(pa) > 1e-12 and np.std(ra) > 1e-12 else 0.0
        acc = float(np.mean(np.where(pa >= 0, 1, -1) == np.where(ra >= 0, 1, -1)))
        return {"samples": len(rows), "directional_accuracy": acc, "correlation": corr}
