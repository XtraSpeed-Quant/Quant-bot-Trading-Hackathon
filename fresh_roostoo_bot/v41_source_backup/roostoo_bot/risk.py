from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Config
from .store import Store


@dataclass(frozen=True)
class RiskDecision:
    block_buys: bool
    liquidate: bool
    reason: str
    day_key: str
    day_start_equity: float
    peak_equity: float
    daily_pnl_pct: float
    drawdown_pct: float
    turnover_used: float


class RiskManager:
    def __init__(self, config: Config, store: Store) -> None:
        self.config = config
        self.store = store
        self.ns = config.bot_namespace

    def _key(self, name: str) -> str:
        return self.store.key(self.ns, name)

    @staticmethod
    def day_key(timestamp_ms: int) -> str:
        return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat()

    def assess(self, timestamp_ms: int, equity: float) -> RiskDecision:
        day = self.day_key(timestamp_ms)
        stored_day = self.store.get_state(self._key("risk_day"))
        day_start = float(self.store.get_state(self._key("day_start_equity"), equity))
        if stored_day != day:
            day_start = equity
            self.store.set_state(self._key("risk_day"), day)
            self.store.set_state(self._key("day_start_equity"), day_start)

        peak = max(float(self.store.get_state(self._key("peak_equity"), equity)), equity)
        self.store.set_state(self._key("peak_equity"), peak)
        daily_pnl = equity / day_start - 1.0 if day_start > 0 else 0.0
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        turnover = self.store.day_turnover(day)

        if drawdown <= -self.config.max_drawdown_limit:
            self.store.set_state(self._key("global_lock"), True)
        locked = bool(self.store.get_state(self._key("global_lock"), False))
        if locked:
            return RiskDecision(True, True, "global_drawdown_lock", day, day_start, peak, daily_pnl, drawdown, turnover)
        if daily_pnl <= -self.config.daily_loss_limit:
            return RiskDecision(True, True, "daily_loss_lock", day, day_start, peak, daily_pnl, drawdown, turnover)
        return RiskDecision(False, False, "ok", day, day_start, peak, daily_pnl, drawdown, turnover)

    def turnover_remaining(self, decision: RiskDecision, equity: float) -> float:
        return max(0.0, equity * self.config.max_daily_turnover - decision.turnover_used)
