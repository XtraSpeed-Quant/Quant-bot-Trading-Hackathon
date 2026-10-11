from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskSnapshot:
    equity: float = 0.0
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    halted: bool = False


class RiskManager:
    def __init__(self, starting_equity: float, daily_loss_limit: float = 0.0, max_drawdown: float = 0.0):
        self.starting_equity = float(starting_equity)
        self.day_start_equity = float(starting_equity)
        self.peak_equity = float(starting_equity)
        self.daily_loss_limit = float(daily_loss_limit)
        self.max_drawdown = float(max_drawdown)
        self.daily_loss_latched = False
        self.drawdown_latched = False
        self.snapshot = RiskSnapshot(equity=starting_equity, day_start_equity=starting_equity, peak_equity=starting_equity)

    def new_day(self, equity: float) -> None:
        self.day_start_equity = float(equity)
        self.daily_loss_latched = False

    def update(self, equity: float, day_start_equity: float | None = None) -> RiskSnapshot:
        equity = float(equity)
        if day_start_equity is not None:
            self.day_start_equity = float(day_start_equity)
        self.peak_equity = max(self.peak_equity, equity)
        daily_pnl = (equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity else 0.0
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0.0
        # A threshold of 0 disables the optional portfolio-wide trading halt.
        # Position-level exit strategies continue to run independently in the planner.
        if self.daily_loss_limit > 0 and daily_pnl <= -self.daily_loss_limit:
            self.daily_loss_latched = True
        if self.max_drawdown > 0 and dd >= self.max_drawdown:
            self.drawdown_latched = True
        halted = self.daily_loss_latched or self.drawdown_latched
        self.snapshot = RiskSnapshot(equity, self.day_start_equity, self.peak_equity, daily_pnl, dd, halted)
        return self.snapshot

    def position_size(self, portfolio_value: float, target_weight: float) -> float:
        return max(0.0, float(portfolio_value) * float(target_weight))
