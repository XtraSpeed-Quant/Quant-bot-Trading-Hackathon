"""
Momentum Strategy v2
EMA crossover + RSI with hardened entry filters:

  Changes from v1:
  - 2-tick confirmation: fast EMA must be above slow for 2 consecutive ticks before BUY
  - Tighter RSI buy zone: 45-58 (was 40-60)
  - Stronger EMA separation: 0.05% (was 0.02%)
  - Per-pair cooldown: 3-cycle lockout after a losing trade
  - Max open positions: enforced by caller (main.py)
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass
class MomentumConfig:
    fast_ema: int   = 8
    slow_ema: int   = 21
    rsi_period: int = 14
    min_history: int = 50

    # Entry filters (tighter than v1)
    rsi_buy_min: float  = 45.0    # was 40 -- avoid weak/falling momentum
    rsi_buy_max: float  = 58.0    # was 60 -- avoid overbought entries
    rsi_sell_min: float = 45.0    # don't sell on signal if RSI < 45

    # EMA filters
    ema_separation_pct: float = 0.05   # was 0.02 -- crossover must be meaningful
    confirm_ticks: int = 2              # NEW: fast must be above slow for N ticks

    # Hold & exit filters
    min_hold_cycles: int   = 3
    min_profit_pct: float  = 0.25
    stop_loss_pct: float   = 2.0
    take_profit_pct: float = 3.0

    # Cooldown after a losing trade
    loss_cooldown_cycles: int = 3       # NEW: lockout after stop-loss


@dataclass
class PairState:
    prices: deque = field(default_factory=lambda: deque(maxlen=200))
    last_signal: Signal = "HOLD"
    entry_price: float  = 0.0
    hold_cycles: int    = 0
    just_restored: bool = False

    # Confirmation tracking
    ticks_above_slow: int = 0   # consecutive ticks where fast > slow

    # Cooldown after loss
    cooldown_cycles: int  = 0   # cycles remaining before next entry allowed


def _ema(prices: list, period: int) -> float:
    if len(prices) < period:
        return float("nan")
    k = 2 / (period + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(prices: list, period: int) -> float:
    if len(prices) < period + 1:
        return 50.0
    relevant = prices[-(period + 1):]
    deltas   = [relevant[i+1] - relevant[i] for i in range(len(relevant) - 1)]
    gains    = [d for d in deltas if d > 0]
    losses   = [-d for d in deltas if d < 0]
    avg_gain = sum(gains)  / period if gains  else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


class MomentumStrategy:
    def __init__(self, config: MomentumConfig = None):
        self.config     = config or MomentumConfig()
        self._states: dict[str, PairState] = {}
        self._restoring = False

    def _state(self, pair: str) -> PairState:
        if pair not in self._states:
            self._states[pair] = PairState()
        return self._states[pair]

    def update(self, pair: str, price: float) -> Signal:
        if price <= 0:
            return "HOLD"

        state  = self._state(pair)
        state.prices.append(price)
        prices = list(state.prices)
        cfg    = self.config

        if len(prices) < cfg.min_history:
            return "HOLD"

        # First live tick after restore -- skip signals, clear flag
        if state.just_restored:
            state.just_restored = False
            return "HOLD"

        fast      = _ema(prices, cfg.fast_ema)
        slow      = _ema(prices, cfg.slow_ema)
        rsi       = _rsi(prices, cfg.rsi_period)
        prev      = prices[:-1]
        prev_fast = _ema(prev, cfg.fast_ema)
        prev_slow = _ema(prev, cfg.slow_ema)

        separation_pct = ((fast - slow) / slow * 100) if slow > 0 else 0

        # Track consecutive ticks above slow EMA
        if fast > slow:
            state.ticks_above_slow += 1
        else:
            state.ticks_above_slow = 0

        # Count down cooldown
        if state.cooldown_cycles > 0:
            state.cooldown_cycles -= 1

        signal: Signal = "HOLD"

        # -- BUY -------------------------------------------------------
        # Entry requires a FRESH crossover event:
        # fast EMA must have crossed above slow EMA within the last
        # `confirm_ticks` live ticks (not stale DB-restored history).
        # This prevents buying 6 coins at once on startup just because
        # the DB shows they've been in uptrend for days.
        crossed_up = prev_fast <= prev_slow and fast > slow

        # trend_confirmed: crossover happened AND has been above for
        # exactly confirm_ticks (not more) -- ensures signal is recent
        trend_confirmed = (
            state.ticks_above_slow >= cfg.confirm_ticks
            and state.ticks_above_slow <= cfg.confirm_ticks + 2  # only fires in narrow window
            and cfg.rsi_buy_min <= rsi <= cfg.rsi_buy_max
            and separation_pct >= cfg.ema_separation_pct
            and state.cooldown_cycles == 0
            and not state.just_restored
            and state.entry_price == 0
            and state.last_signal != "BUY"
        )

        if trend_confirmed or (
            crossed_up
            and fast > slow
            and cfg.rsi_buy_min <= rsi <= cfg.rsi_buy_max
            and separation_pct >= cfg.ema_separation_pct
            and state.cooldown_cycles == 0
            and not state.just_restored
            and state.entry_price == 0      # BUG FIX: don't overwrite an open position's entry
            and state.last_signal != "BUY"  # BUG FIX: don't re-fire if already in BUY state
        ):
            signal = "BUY"
            state.entry_price = price
            state.hold_cycles = 0

        # -- SELL ------------------------------------------------------
        elif state.last_signal == "BUY" or state.hold_cycles > 0:
            state.hold_cycles += 1
            pnl_pct = ((price - state.entry_price) / state.entry_price * 100) if state.entry_price > 0 else 0

            # 1. Hard stop-loss
            if not self._restoring and pnl_pct <= -cfg.stop_loss_pct:
                signal = "SELL"
                state.cooldown_cycles = cfg.loss_cooldown_cycles  # lockout
                logger.warning("%s STOP-LOSS: pnl=%.2f%% -- cooldown %d cycles",
                               pair, pnl_pct, cfg.loss_cooldown_cycles)

            # 2. Take profit
            elif not self._restoring and pnl_pct >= cfg.take_profit_pct:
                signal = "SELL"
                logger.info("%s TAKE-PROFIT: pnl=%.2f%%", pair, pnl_pct)

            # 3. EMA cross down -- only if profitable and held long enough
            elif (
                prev_fast >= prev_slow and fast < slow
                and rsi > cfg.rsi_sell_min
                and state.hold_cycles >= cfg.min_hold_cycles
                and pnl_pct >= cfg.min_profit_pct
            ):
                signal = "SELL"
                logger.info("%s EMA-SELL: pnl=%.2f%% cycles=%d",
                            pair, pnl_pct, state.hold_cycles)

        if signal not in ("HOLD",):
            logger.info("%s signal=%s | fast=%.6f slow=%.6f rsi=%.1f sep=%.4f%% cooldown=%d confirm=%d",
                        pair, signal, fast, slow, rsi, separation_pct,
                        state.cooldown_cycles, state.ticks_above_slow)

        if signal == "SELL":
            state.entry_price = 0.0
            state.hold_cycles = 0

        state.last_signal = signal
        return signal

    def notify_bought(self, pair: str, price: float):
        state = self._state(pair)
        state.entry_price = price
        state.hold_cycles = 0
        state.last_signal = "BUY"

    def notify_sold(self, pair: str, was_loss: bool = False):
        state = self._state(pair)
        state.entry_price = 0.0
        state.hold_cycles = 0
        state.last_signal = "HOLD"
        if was_loss:
            state.cooldown_cycles = self.config.loss_cooldown_cycles

    def indicators(self, pair: str) -> dict:
        state  = self._state(pair)
        prices = list(state.prices)
        cfg    = self.config
        if len(prices) < cfg.min_history:
            return {"warming_up": True, "prices_collected": len(prices)}
        fast = _ema(prices, cfg.fast_ema)
        slow = _ema(prices, cfg.slow_ema)
        sep  = ((fast - slow) / slow * 100) if slow > 0 else 0
        return {
            "pair":             pair,
            "last_price":       prices[-1],
            "fast_ema":         round(fast, 6),
            "slow_ema":         round(slow, 6),
            "ema_sep_pct":      round(sep, 4),
            "rsi":              round(_rsi(prices, cfg.rsi_period), 2),
            "hold_cycles":      state.hold_cycles,
            "entry_price":      state.entry_price,
            "ticks_above_slow": state.ticks_above_slow,
            "cooldown_cycles":  state.cooldown_cycles,
            "prices_collected": len(prices),
        }

    def reset(self, pair: str = None):
        if pair:
            self._states.pop(pair, None)
        else:
            self._states.clear()
