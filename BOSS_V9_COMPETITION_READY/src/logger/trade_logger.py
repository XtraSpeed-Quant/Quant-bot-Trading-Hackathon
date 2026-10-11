from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


class TradeLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_path = self.log_dir / "trades.csv"
        self.performance_path = self.log_dir / "performance.jsonl"
        self._snapshots: list[float] = []
        if not self.trade_path.exists():
            with self.trade_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["timestamp","pair","side","quantity","price","fee","order_id","status","note"])

    def log_order(self, result: dict, note: str = "") -> None:
        d = result.get("OrderDetail", result)
        row = [
            datetime.now(IST).isoformat(), d.get("Pair", ""), d.get("Side", ""),
            d.get("FilledQuantity") or d.get("Quantity") or d.get("ShortQty") or d.get("ClosedQty", 0),
            d.get("FilledAverPrice") or d.get("Price") or d.get("EntryPrice") or d.get("ClosePrice", 0),
            d.get("CommissionChargeValue") or d.get("OpenFee") or d.get("CloseFee", 0),
            d.get("OrderID") or d.get("ID", ""), d.get("Status", ""), note,
        ]
        with self.trade_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def snapshot(self, equity: float, extra: dict | None = None) -> None:
        self._snapshots.append(float(equity))
        row = {"timestamp": datetime.now(IST).isoformat(), "portfolio_usd": round(float(equity), 4)}
        if extra:
            row.update(extra)
        with self.performance_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")

    def summary(self, initial: float, current: float) -> dict:
        if len(self._snapshots) < 2:
            return {"status": "insufficient_data", "return_pct": (current - initial) / initial * 100 if initial else 0.0}
        import math
        rets = [(self._snapshots[i] - self._snapshots[i-1]) / self._snapshots[i-1] for i in range(1, len(self._snapshots)) if self._snapshots[i-1] != 0]
        mean = sum(rets) / len(rets) if rets else 0.0
        std = math.sqrt(sum((r-mean)**2 for r in rets)/len(rets)) if rets else 0.0
        downside = [r for r in rets if r < 0]
        downstd = math.sqrt(sum(r*r for r in downside)/len(downside)) if downside else 0.0
        peak = initial
        maxdd = 0.0
        for v in self._snapshots:
            peak = max(peak, v)
            maxdd = max(maxdd, (peak-v)/peak if peak else 0.0)
        scale = math.sqrt(len(rets)) if rets else 0.0
        total = (current-initial)/initial if initial else 0.0
        return {
            "return_pct": total * 100,
            "sharpe": mean/std*scale if std else 0.0,
            "sortino": mean/downstd*scale if downstd else 0.0,
            "max_drawdown_pct": maxdd*100,
        }
