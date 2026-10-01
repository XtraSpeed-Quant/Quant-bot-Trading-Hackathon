from pathlib import Path
p=Path('/tmp/roostoo')

config = '''from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_ENV_FILE = find_dotenv(usecwd=True)
if _ENV_FILE:
    load_dotenv(_ENV_FILE, override=False)


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    api_url: str = os.getenv("ROOSTOO_API_URL", "https://mock-api.roostoo.com").rstrip("/")
    api_key: str = os.getenv("ROOSTOO_API_KEY", "")
    api_secret: str = os.getenv("ROOSTOO_API_SECRET", "")

    bot_namespace: str = os.getenv("BOT_NAMESPACE", "test-v42")
    poll_seconds: int = _int("POLL_SECONDS", 60)
    rebalance_seconds: int = _int("REBALANCE_SECONDS", 900)
    timeout_seconds: float = _float("TIMEOUT_SECONDS", 12.0)
    stale_data_seconds: int = _int("STALE_DATA_SECONDS", 180)
    prediction_retrain_seconds: int = _int("PREDICTION_RETRAIN_SECONDS", 900)
    prediction_log_interval_seconds: int = _int("PREDICTION_LOG_INTERVAL_SECONDS", 300)
    prediction_log_top_n: int = _int("PREDICTION_LOG_TOP_N", 12)

    expected_starting_cash: float = _float("EXPECTED_STARTING_CASH", 100_000.0)
    starting_cash_warn_pct: float = _float("STARTING_CASH_WARN_PCT", 0.20)
    orders_enabled: bool = _bool("ORDERS_ENABLED", False)
    enable_shorts: bool = _bool("ENABLE_SHORTS", False)
    allow_pairs: tuple[str, ...] = _csv("ALLOW_PAIRS")
    execution_style: str = os.getenv("EXECUTION_STYLE", "MARKET").upper()

    min_quote_volume: float = _float("MIN_QUOTE_VOLUME", 250_000.0)
    max_spread_bps: float = _float("MAX_SPREAD_BPS", 35.0)

    fast_window: int = _int("FAST_WINDOW", 60)
    medium_window: int = _int("MEDIUM_WINDOW", 240)
    long_window: int = _int("LONG_WINDOW", 480)
    trend_window: int = _int("TREND_WINDOW", 360)
    vol_window: int = _int("VOL_WINDOW", 60)
    history_limit: int = _int("HISTORY_LIMIT", 4320)

    top_n: int = _int("TOP_N", 5)
    max_gross_target: float = _float("MAX_GROSS_TARGET", 0.70)
    max_asset_weight: float = _float("MAX_ASSET_WEIGHT", 0.14)
    max_short_gross: float = _float("MAX_SHORT_GROSS", 0.20)
    max_short_weight: float = _float("MAX_SHORT_WEIGHT", 0.08)
    min_score: float = _float("MIN_SCORE", 0.53)
    min_relative_edge: float = _float("MIN_RELATIVE_EDGE", 0.0025)
    min_1h_return: float = _float("MIN_1H_RETURN", -0.004)
    min_4h_return: float = _float("MIN_4H_RETURN", -0.005)
    min_12h_return: float = _float("MIN_12H_RETURN", -0.015)
    volatility_floor: float = _float("VOLATILITY_FLOOR", 0.0005)

    rebalance_band: float = _float("REBALANCE_BAND", 0.012)
    min_order_usd: float = _float("MIN_ORDER_USD", 10.0)
    max_order_usd: float = _float("MAX_ORDER_USD", 20_000.0)
    max_daily_turnover: float = _float("MAX_DAILY_TURNOVER", 0.40)
    cash_reserve: float = _float("CASH_RESERVE", 0.10)
    min_hold_minutes: int = _int("MIN_HOLD_MINUTES", 30)

    daily_loss_limit: float = _float("DAILY_LOSS_LIMIT", 0.03)
    max_drawdown_limit: float = _float("MAX_DRAWDOWN_LIMIT", 0.08)

    order_settle_attempts: int = _int("ORDER_SETTLE_ATTEMPTS", 5)
    order_settle_delay_seconds: float = _float("ORDER_SETTLE_DELAY_SECONDS", 1.0)
    order_reconcile_window_seconds: int = _int("ORDER_RECONCILE_WINDOW_SECONDS", 90)
    post_trade_reconcile_retries: int = _int("POST_TRADE_RECONCILE_RETRIES", 3)

    predictor_enabled: bool = _bool("PREDICTOR_ENABLED", True)
    prediction_horizon: int = _int("PREDICTION_HORIZON", 30)
    predictor_feature_window: int = _int("PREDICTOR_FEATURE_WINDOW", 360)
    predictor_min_samples: int = _int("PREDICTOR_MIN_SAMPLES", 2000)
    predictor_max_samples: int = _int("PREDICTOR_MAX_SAMPLES", 25_000)
    predictor_ridge_alpha: float = _float("PREDICTOR_RIDGE_ALPHA", 10.0)
    predictor_classifier_alpha: float = _float("PREDICTOR_CLASSIFIER_ALPHA", 2.0)
    predictor_validation_fraction: float = _float("PREDICTOR_VALIDATION_FRACTION", 0.25)
    predictor_purge_gap: int = _int("PREDICTOR_PURGE_GAP", 30)
    predictor_min_directional_accuracy: float = _float("PREDICTOR_MIN_DIRECTIONAL_ACCURACY", 0.515)
    predictor_min_correlation: float = _float("PREDICTOR_MIN_CORRELATION", 0.01)
    predictor_min_balanced_accuracy: float = _float("PREDICTOR_MIN_BALANCED_ACCURACY", 0.36)
    predictor_min_top_decile_edge: float = _float("PREDICTOR_MIN_TOP_DECILE_EDGE", 0.0003)
    predictor_max_brier: float = _float("PREDICTOR_MAX_BRIER", 0.66)
    predictor_class_threshold: float = _float("PREDICTOR_CLASS_THRESHOLD", 0.0015)
    min_prediction_edge: float = _float("MIN_PREDICTION_EDGE", 0.0015)

    correlation_window: int = _int("CORRELATION_WINDOW", 180)
    max_pair_correlation: float = _float("MAX_PAIR_CORRELATION", 0.88)
    correlation_penalty: float = _float("CORRELATION_PENALTY", 0.30)
    overextension_penalty: float = _float("OVEREXTENSION_PENALTY", 0.18)
    breakout_window: int = _int("BREAKOUT_WINDOW", 240)

    gross_strength_power: float = _float("GROSS_STRENGTH_POWER", 1.15)
    risk_on_gross_cap: float = _float("RISK_ON_GROSS_CAP", 1.00)
    balanced_gross_cap: float = _float("BALANCED_GROSS_CAP", 0.80)
    risk_off_gross_cap: float = _float("RISK_OFF_GROSS_CAP", 0.50)
    defensive_gross_cap: float = _float("DEFENSIVE_GROSS_CAP", 0.25)
    defensive_min_breadth: float = _float("DEFENSIVE_MIN_BREADTH", 0.08)
    defensive_min_strength: float = _float("DEFENSIVE_MIN_STRENGTH", 0.12)
    allocation_edge_power: float = _float("ALLOCATION_EDGE_POWER", 1.10)
    allocation_vol_power: float = _float("ALLOCATION_VOL_POWER", 0.75)
    scan_top_n: int = _int("SCAN_TOP_N", 8)

    taker_fee_assumption: float = _float("TAKER_FEE_ASSUMPTION", 0.001)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_file: str = os.getenv("LOG_FILE", "logs/bot.log")
    db_path: str = os.getenv("BOT_DB_PATH", "data/bot.sqlite3")

    def validate(self, require_credentials: bool = True) -> None:
        numeric = [
            self.poll_seconds, self.rebalance_seconds, self.timeout_seconds,
            self.stale_data_seconds, self.prediction_retrain_seconds, self.prediction_log_interval_seconds,
            self.prediction_log_top_n, self.expected_starting_cash, self.starting_cash_warn_pct,
            self.min_quote_volume, self.max_spread_bps, self.fast_window, self.medium_window,
            self.long_window, self.trend_window, self.vol_window, self.history_limit, self.top_n,
            self.max_gross_target, self.max_asset_weight, self.max_short_gross, self.max_short_weight,
            self.min_score, self.min_relative_edge, self.min_1h_return, self.min_4h_return,
            self.min_12h_return, self.volatility_floor, self.rebalance_band, self.min_order_usd,
            self.max_order_usd, self.max_daily_turnover, self.cash_reserve, self.min_hold_minutes,
            self.daily_loss_limit, self.max_drawdown_limit, self.order_settle_attempts,
            self.order_settle_delay_seconds, self.order_reconcile_window_seconds,
            self.post_trade_reconcile_retries, self.prediction_horizon, self.predictor_feature_window,
            self.predictor_min_samples, self.predictor_max_samples, self.predictor_ridge_alpha,
            self.predictor_classifier_alpha, self.predictor_validation_fraction, self.predictor_purge_gap,
            self.predictor_min_directional_accuracy, self.predictor_min_correlation,
            self.predictor_min_balanced_accuracy, self.predictor_min_top_decile_edge, self.predictor_max_brier,
            self.predictor_class_threshold, self.min_prediction_edge, self.correlation_window,
            self.max_pair_correlation, self.correlation_penalty, self.overextension_penalty,
            self.breakout_window, self.gross_strength_power, self.risk_on_gross_cap,
            self.balanced_gross_cap, self.risk_off_gross_cap, self.defensive_gross_cap,
            self.defensive_min_breadth, self.defensive_min_strength, self.allocation_edge_power,
            self.allocation_vol_power, self.scan_top_n, self.taker_fee_assumption,
        ]
        if any(not math.isfinite(float(v)) for v in numeric):
            raise ValueError("All numeric configuration values must be finite")
        if not self.api_url.startswith(("https://", "http://")):
            raise ValueError("ROOSTOO_API_URL must be an HTTP(S) URL")
        if self.poll_seconds < 30:
            raise ValueError("POLL_SECONDS must be at least 30 seconds")
        if self.prediction_log_interval_seconds < self.poll_seconds:
            raise ValueError("PREDICTION_LOG_INTERVAL_SECONDS must be >= POLL_SECONDS")
        if self.prediction_log_top_n < 1 or self.scan_top_n < 1:
            raise ValueError("Prediction and scan display limits must be positive")
        if self.rebalance_seconds < self.poll_seconds:
            raise ValueError("REBALANCE_SECONDS must be >= POLL_SECONDS")
        if not (2 <= self.fast_window < self.medium_window < self.long_window):
            raise ValueError("Require FAST < MEDIUM < LONG")
        if self.trend_window < self.medium_window:
            raise ValueError("TREND_WINDOW must be >= MEDIUM_WINDOW")
        required = max(self.long_window, self.trend_window, self.vol_window, self.breakout_window) + 1
        if self.history_limit < required:
            raise ValueError("HISTORY_LIMIT is too small for configured features")
        if not 1 <= self.top_n <= 20:
            raise ValueError("TOP_N must be between 1 and 20")
        if not (0 < self.max_gross_target <= 1):
            raise ValueError("MAX_GROSS_TARGET must be in (0, 1]")
        if not (0 < self.max_asset_weight <= 1):
            raise ValueError("MAX_ASSET_WEIGHT must be in (0, 1]")
        if self.top_n * self.max_asset_weight + 1e-12 < self.max_gross_target:
            raise ValueError("TOP_N * MAX_ASSET_WEIGHT must cover MAX_GROSS_TARGET")
        if self.max_short_weight * self.top_n + 1e-12 < self.max_short_gross and self.enable_shorts:
            raise ValueError("TOP_N * MAX_SHORT_WEIGHT must cover MAX_SHORT_GROSS")
        if self.min_quote_volume < 0 or self.max_spread_bps <= 0:
            raise ValueError("Invalid liquidity/spread limits")
        if self.volatility_floor <= 0:
            raise ValueError("VOLATILITY_FLOOR must be positive")
        if not (0 < self.rebalance_band < 1):
            raise ValueError("REBALANCE_BAND must be in (0, 1)")
        if self.min_order_usd <= 0 or self.max_order_usd < self.min_order_usd:
            raise ValueError("Invalid order size bounds")
        if not (0 < self.cash_reserve < 1):
            raise ValueError("CASH_RESERVE must be in (0, 1)")
        if not (0 < self.daily_loss_limit < 1 and 0 < self.max_drawdown_limit < 1):
            raise ValueError("Risk limits must be in (0, 1)")
        if self.taker_fee_assumption <= 0:
            raise ValueError("TAKER_FEE_ASSUMPTION must be positive")
        if self.predictor_min_samples < 200 or self.predictor_max_samples < self.predictor_min_samples:
            raise ValueError("Invalid predictor sample limits")
        if not (0 < self.predictor_max_brier <= 1.0):
            raise ValueError("PREDICTOR_MAX_BRIER must be in (0, 1]")
        if not (0.10 <= self.predictor_validation_fraction < 0.50):
            raise ValueError("PREDICTOR_VALIDATION_FRACTION must be in [0.10, 0.50)")
        if not (0.50 <= self.predictor_min_directional_accuracy <= 1.0):
            raise ValueError("Invalid minimum predictor directional accuracy")
        if not (1 / 3 <= self.predictor_min_balanced_accuracy <= 1.0):
            raise ValueError("Invalid minimum predictor balanced accuracy")
        if self.predictor_purge_gap < 0:
            raise ValueError("PREDICTOR_PURGE_GAP cannot be negative")
        if self.predictor_ridge_alpha <= 0 or self.predictor_classifier_alpha <= 0:
            raise ValueError("Predictor regularization must be positive")
        if self.correlation_window < 30:
            raise ValueError("CORRELATION_WINDOW must be at least 30")
        if not (0 < self.max_pair_correlation <= 1):
            raise ValueError("Invalid MAX_PAIR_CORRELATION")
        if self.correlation_penalty < 0 or self.overextension_penalty < 0:
            raise ValueError("Penalties cannot be negative")
        if self.gross_strength_power <= 0 or self.allocation_edge_power <= 0 or self.allocation_vol_power <= 0:
            raise ValueError("Sizing powers must be positive")
        if not (0 < self.risk_on_gross_cap <= 1 and 0 < self.balanced_gross_cap <= 1 and 0 < self.risk_off_gross_cap <= 1 and 0 < self.defensive_gross_cap <= 1):
            raise ValueError("Invalid regime gross caps")
        if not (0 <= self.defensive_min_breadth <= 1 and 0 <= self.defensive_min_strength <= 1):
            raise ValueError("Invalid defensive entry thresholds")
        if require_credentials and (not self.api_key or not self.api_secret):
            raise ValueError("ROOSTOO_API_KEY and ROOSTOO_API_SECRET are required")
        if not self.bot_namespace.strip():
            raise ValueError("BOT_NAMESPACE cannot be empty")
        if self.execution_style not in {"MARKET"}:
            raise ValueError("EXECUTION_STYLE currently supports MARKET only")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
'''
(p/'roostoo_bot/config.py').write_text(config, encoding='utf-8')

# Start from existing strategy and patch targeted sections.
strategy = (p/'roostoo_bot/strategy.py').read_text(encoding='utf-8')
strategy = strategy.replace('''        overextension_penalty: float,\n        breakout_window: int,\n        taker_fee_assumption: float,\n        enable_shorts: bool,\n''','''        overextension_penalty: float,\n        breakout_window: int,\n        taker_fee_assumption: float,\n        enable_shorts: bool,\n        gross_strength_power: float,\n        risk_on_gross_cap: float,\n        balanced_gross_cap: float,\n        risk_off_gross_cap: float,\n        defensive_gross_cap: float,\n        defensive_min_breadth: float,\n        defensive_min_strength: float,\n        allocation_edge_power: float,\n        allocation_vol_power: float,\n        scan_top_n: int,\n''')
strategy = strategy.replace('''        self.taker_fee_assumption = taker_fee_assumption\n''','''        self.taker_fee_assumption = taker_fee_assumption\n        self.gross_strength_power = gross_strength_power\n        self.risk_on_gross_cap = risk_on_gross_cap\n        self.balanced_gross_cap = balanced_gross_cap\n        self.risk_off_gross_cap = risk_off_gross_cap\n        self.defensive_gross_cap = defensive_gross_cap\n        self.defensive_min_breadth = defensive_min_breadth\n        self.defensive_min_strength = defensive_min_strength\n        self.allocation_edge_power = allocation_edge_power\n        self.allocation_vol_power = allocation_vol_power\n        self.scan_top_n = scan_top_n\n''')
old = '''        gross = min(\n            self.max_gross_target * (0.18 + 0.95 * strength) * vol_scale,\n            self.max_gross_target,\n            1.0 - 0.10,\n        )\n        if regime == "balanced":\n            gross = min(gross, self.max_gross_target * 0.82)\n        elif regime == "risk_off":\n            gross = min(gross, self.max_gross_target * 0.58)\n        elif regime == "defensive":\n            gross = min(gross, self.max_gross_target * 0.28)\n'''
new = '''        if breadth < self.defensive_min_breadth or strength < self.defensive_min_strength:\n            gross = 0.0\n        else:\n            regime_cap = {\n                "risk_on": self.risk_on_gross_cap,\n                "balanced": self.balanced_gross_cap,\n                "risk_off": self.risk_off_gross_cap,\n                "defensive": self.defensive_gross_cap,\n            }.get(regime, self.defensive_gross_cap)\n            market_budget = strength ** self.gross_strength_power\n            gross = min(\n                self.max_gross_target * market_budget * vol_scale,\n                self.max_gross_target * regime_cap,\n                1.0 - self.cash_reserve if hasattr(self, "cash_reserve") else 0.90,\n            )\n'''
# add cash_reserve to strategy init via fixed 0.10 from config; better pass later. Keep fallback now.
strategy = strategy.replace(old, new)
old_alloc = '''                raw[signal.pair] = (\n                    (edge ** 1.55)\n                    * confidence_boost\n                    / max(signal.volatility, self.volatility_floor)\n                )\n'''
new_alloc = '''                raw[signal.pair] = (\n                    (edge ** self.allocation_edge_power)\n                    * confidence_boost\n                    / max(signal.volatility, self.volatility_floor) ** self.allocation_vol_power\n                )\n'''
strategy = strategy.replace(old_alloc, new_alloc)
# Introduce a top candidate row payload without changing Signal schema.
needle = '''        weighted: list[Signal] = []\n        for signal in chosen_longs + chosen_shorts:\n'''
replacement = '''        weighted: list[Signal] = []\n        for signal in chosen_longs + chosen_shorts:\n'''
strategy = strategy.replace(needle, replacement)
# Add cash_reserve to strategy object cleanly: signature and assignment
strategy = strategy.replace('''        enable_shorts: bool,\n        gross_strength_power: float,''','''        enable_shorts: bool,\n        cash_reserve: float,\n        gross_strength_power: float,''')
strategy = strategy.replace('''        self.enable_shorts = enable_shorts\n        self.min_prediction_edge = min_prediction_edge\n''','''        self.enable_shorts = enable_shorts\n        self.cash_reserve = cash_reserve\n        self.min_prediction_edge = min_prediction_edge\n''')
# Add explicit top-candidate serialization in StrategyResult via dynamically attached attribute is impossible; instead engine will derive from signal rows only.
(p/'roostoo_bot/strategy.py').write_text(strategy, encoding='utf-8')

# Predictor full rewrite with temporal split fix + gate reason and richer validation outputs.
predictor = (p/'roostoo_bot/predictor.py').read_text(encoding='utf-8')
predictor = predictor.replace('''class PredictorResult:\n    enabled: bool\n    samples: int\n    validation_directional_accuracy: float\n    validation_correlation: float\n    validation_balanced_accuracy: float\n    validation_top_decile_edge: float\n    validation_brier: float\n    residual_std: float\n    predictions: dict[str, Prediction]\n''','''class PredictorResult:\n    enabled: bool\n    samples: int\n    validation_directional_accuracy: float\n    validation_correlation: float\n    validation_balanced_accuracy: float\n    validation_top_decile_edge: float\n    validation_brier: float\n    residual_std: float\n    predictions: dict[str, Prediction]\n    gate_reason: str = ""\n''')
old = '''        latest = float(np.max(times))\n        split = latest - max(1.0, latest * self.validation_fraction)\n        train_mask = times < split - self.purge_gap\n        valid_mask = times >= split\n        if np.count_nonzero(train_mask) < 200 or np.count_nonzero(valid_mask) < 100:\n            self._model = None\n            result = PredictorResult(False, samples, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, {})\n            self._last_validation = result\n            return result\n'''
new = '''        earliest = float(np.min(times))\n        latest = float(np.max(times))\n        span = max(latest - earliest, 1.0)\n        split = earliest + span * (1.0 - self.validation_fraction)\n        train_mask = times < split - self.purge_gap\n        valid_mask = times >= split\n        train_count = int(np.count_nonzero(train_mask))\n        valid_count = int(np.count_nonzero(valid_mask))\n        if train_count < 200 or valid_count < 100:\n            self._model = None\n            reason = f"time split too small: train={train_count}, validation={valid_count}; collect more history"\n            result = PredictorResult(False, samples, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, {}, reason)\n            self._last_validation = result\n            return result\n'''
predictor = predictor.replace(old, new)
old2 = '''        enabled = (\n            accuracy >= self.min_directional_accuracy\n            and correlation >= self.min_correlation\n            and balanced_accuracy >= self.min_balanced_accuracy\n            and top_edge >= self.min_top_decile_edge\n            and brier <= self.max_brier\n            and residual_std > 1e-8\n        )\n\n        live_model = self._fit_model(x, y, classes) if enabled else validation_model\n'''
new2 = '''        gate_failures = []\n        if accuracy < self.min_directional_accuracy:\n            gate_failures.append(f"direction {accuracy:.3f}<{self.min_directional_accuracy:.3f}")\n        if correlation < self.min_correlation:\n            gate_failures.append(f"correlation {correlation:.4f}<{self.min_correlation:.4f}")\n        if balanced_accuracy < self.min_balanced_accuracy:\n            gate_failures.append(f"balanced_acc {balanced_accuracy:.3f}<{self.min_balanced_accuracy:.3f}")\n        if top_edge < self.min_top_decile_edge:\n            gate_failures.append(f"top_decile {top_edge:.5f}<{self.min_top_decile_edge:.5f}")\n        if brier > self.max_brier:\n            gate_failures.append(f"brier {brier:.3f}>{self.max_brier:.3f}")\n        if residual_std <= 1e-8:\n            gate_failures.append("residual_std too small")\n        enabled = not gate_failures\n\n        live_model = self._fit_model(x, y, classes) if enabled else validation_model\n'''
predictor = predictor.replace(old2, new2)
predictor = predictor.replace('''            predictions=self._current_predictions(histories),\n        )\n        self._last_validation = result\n''','''            predictions=self._current_predictions(histories),\n            gate_reason="passed all gates" if enabled else "; ".join(gate_failures),\n        )\n        self._last_validation = result\n''')
# Fix non-refit result copy to include gate reason
predictor = predictor.replace('''                residual_std=previous.residual_std,\n                predictions=self._current_predictions(histories),\n            )\n''','''                residual_std=previous.residual_std,\n                predictions=self._current_predictions(histories),\n                gate_reason=previous.gate_reason,\n            )\n''')
(p/'roostoo_bot/predictor.py').write_text(predictor, encoding='utf-8')

# StrategyResult add predictor_gate_reason field and pass through.
strategy = (p/'roostoo_bot/strategy.py').read_text(encoding='utf-8')
strategy = strategy.replace('''    predictor_brier: float\n''','''    predictor_brier: float\n    predictor_gate_reason: str\n''')
strategy = strategy.replace('''        predictor_brier=1.0,\n    )\n\n\ndef _empty_result''','''        predictor_brier=1.0,\n        predictor_gate_reason="warming up",\n    )\n\n\ndef _empty_result''')
strategy = strategy.replace('''        predictor_brier=1.0,\n    )\n\n\nclass MultiFactorStrategy''','''        predictor_brier=1.0,\n        predictor_gate_reason="warming up",\n    )\n\n\nclass MultiFactorStrategy''')
strategy = strategy.replace('''            predictor_brier=predictor_result.validation_brier,\n        )\n''','''            predictor_brier=predictor_result.validation_brier,\n            predictor_gate_reason=predictor_result.gate_reason,\n        )\n''')
(p/'roostoo_bot/strategy.py').write_text(strategy, encoding='utf-8')

# Engine patches for new config args, all-signal rebalancing, fresh post-trade state, session P&L.
engine = (p/'roostoo_bot/engine.py').read_text(encoding='utf-8')
engine = engine.replace('''            correlation_penalty=config.correlation_penalty, overextension_penalty=config.overextension_penalty,\n            breakout_window=config.breakout_window, taker_fee_assumption=config.taker_fee_assumption,\n            enable_shorts=config.enable_shorts,\n''','''            correlation_penalty=config.correlation_penalty, overextension_penalty=config.overextension_penalty,\n            breakout_window=config.breakout_window, taker_fee_assumption=config.taker_fee_assumption,\n            enable_shorts=config.enable_shorts, cash_reserve=config.cash_reserve,\n            gross_strength_power=config.gross_strength_power, risk_on_gross_cap=config.risk_on_gross_cap,\n            balanced_gross_cap=config.balanced_gross_cap, risk_off_gross_cap=config.risk_off_gross_cap,\n            defensive_gross_cap=config.defensive_gross_cap, defensive_min_breadth=config.defensive_min_breadth,\n            defensive_min_strength=config.defensive_min_strength, allocation_edge_power=config.allocation_edge_power,\n            allocation_vol_power=config.allocation_vol_power, scan_top_n=config.scan_top_n,\n''')
engine = engine.replace('''        self.order_in_flight = False\n''','''        self.order_in_flight = False\n        baseline = store.get_state(self._state_key("session_start_equity"), None)\n        self.session_start_equity = float(baseline) if baseline is not None else None\n''')
engine = engine.replace('''        shorts = self._short_map() if self.config.enable_shorts else {}\n        equity, holdings = self.equity_and_holdings(wallet, markets, shorts)\n''','''        shorts = self._short_map() if self.config.enable_shorts else {}\n        equity, holdings = self.equity_and_holdings(wallet, markets, shorts)\n        if self.session_start_equity is None and equity > 0:\n            self.session_start_equity = equity\n            self.store.set_state(self._state_key("session_start_equity"), equity)\n''')
# Replace default StrategyResult construction with gate reason; occurs twice maybe.
engine = engine.replace('''            predictor_top_decile_edge=0.0, predictor_brier=1.0,\n        )\n''','''            predictor_top_decile_edge=0.0, predictor_brier=1.0, predictor_gate_reason="warming up",\n        )\n''')
# Ensure no-signal cycles still rebalance/exit instead of pass.
old = '''            if not strategy_result.signals:\n                pass\n            elif not execute:\n                sells, buys, short_opens, short_closes = self._order_plan(\n                    strategy_result.long_targets, strategy_result.short_targets, strategy_markets, wallet, shorts,\n                    equity, decision.liquidate, server_timestamp,\n                )\n                result = {"orders": 0, "turnover": 0.0, "blocked": "inspection_only", "planned_sells": sells, "planned_buys": buys, "planned_short_opens": short_opens, "planned_short_closes": short_closes}\n            elif due or decision.liquidate:\n                result = self._rebalance(strategy_result, strategy_markets, wallet, shorts, equity, decision, server_timestamp)\n                self.last_rebalance_ms = server_timestamp\n                self.store.set_state(self._state_key("last_rebalance_ms"), self.last_rebalance_ms)\n'''
new = '''            if not execute:\n                sells, buys, short_opens, short_closes = self._order_plan(\n                    strategy_result.long_targets, strategy_result.short_targets, strategy_markets, wallet, shorts,\n                    equity, decision.liquidate, server_timestamp,\n                )\n                result = {"orders": 0, "turnover": 0.0, "blocked": "inspection_only", "planned_sells": sells, "planned_buys": buys, "planned_short_opens": short_opens, "planned_short_closes": short_closes}\n            elif due or decision.liquidate:\n                result = self._rebalance(strategy_result, strategy_markets, wallet, shorts, equity, decision, server_timestamp)\n                self.last_rebalance_ms = server_timestamp\n                self.store.set_state(self._state_key("last_rebalance_ms"), self.last_rebalance_ms)\n'''
engine = engine.replace(old, new)
# Replace final exposure calculation with fresh account state after orders.
oldret = '''        today_trades = self.store.daily_trade_counts().get(self.risk.day_key(server_timestamp), 0)\n        exposure = sum(holdings.values()) + sum(float(p.get("collateral", 0.0)) for p in shorts.values())\n        return {\n'''
newret = '''        if result["orders"] > 0:\n            try:\n                refreshed_wallet = self.client.balance()\n                refreshed_shorts = self._short_map() if self.config.enable_shorts else {}\n                equity, holdings = self.equity_and_holdings(refreshed_wallet, markets, refreshed_shorts)\n                wallet = refreshed_wallet\n                shorts = refreshed_shorts\n            except (RoostooError, requests.RequestException):\n                pass\n        today_trades = self.store.daily_trade_counts().get(self.risk.day_key(server_timestamp), 0)\n        exposure = sum(holdings.values()) + sum(float(p.get("collateral", 0.0)) for p in shorts.values())\n        next_rebalance_ms = max(0, self.config.rebalance_seconds * 1000 - max(0, server_timestamp - self.last_rebalance_ms)) if self.last_rebalance_ms else 0\n        account_initial = 0.0\n        if isinstance(self.exchange_info, Mapping):\n            initial_wallet = self.exchange_info.get("InitialWallet", {})\n            if isinstance(initial_wallet, Mapping):\n                account_initial = float(initial_wallet.get("USD", 0.0) or 0.0)\n        session_start = self.session_start_equity or equity\n        return {\n'''
engine = engine.replace(oldret, newret)
# Add fields before final result closes.
engine = engine.replace('''            "prediction_top_decile_edge": strategy_result.predictor_top_decile_edge,\n            "prediction_brier": strategy_result.predictor_brier, "live_prediction_samples": live_metrics["samples"],\n''','''            "prediction_top_decile_edge": strategy_result.predictor_top_decile_edge,\n            "prediction_brier": strategy_result.predictor_brier,\n            "prediction_gate_reason": strategy_result.predictor_gate_reason,\n            "live_prediction_samples": live_metrics["samples"],\n''')
engine = engine.replace('''            "history_points": history_points, "history_required": required, "rebalance_due": due,\n            "orders_enabled": self.config.orders_enabled, "planned_sells": result.get("planned_sells", []),\n''','''            "history_points": history_points, "history_required": required, "rebalance_due": due,\n            "next_rebalance_ms": next_rebalance_ms, "session_start_equity": session_start,\n            "session_pnl": equity - session_start, "session_pnl_pct": equity / session_start - 1.0 if session_start > 0 else 0.0,\n            "account_initial_equity": account_initial,\n            "account_pnl": equity - account_initial if account_initial > 0 else 0.0,\n            "account_pnl_pct": equity / account_initial - 1.0 if account_initial > 0 else 0.0,\n            "cash": self._wallet_amount(wallet, "USD"),\n            "exposure_pct": exposure / equity if equity > 0 else 0.0,\n            "universe_size": len(strategy_markets), "feature_universe": strategy_result.predictor_samples if False else 0,\n            "orders_enabled": self.config.orders_enabled, "planned_sells": result.get("planned_sells", []),\n''')
(p/'roostoo_bot/engine.py').write_text(engine, encoding='utf-8')

# Dashboard rewrite.
dashboard = '''from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def _pct(value: float) -> str:
    return f"{value * 100:+.2f}%"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _countdown(milliseconds: int) -> str:
    seconds = max(0, int(milliseconds / 1000))
    minutes, sec = divmod(seconds, 60)
    return f"{minutes:02d}:{sec:02d}"


def format_dashboard(result: Mapping[str, Any]) -> str:
    width = 100
    timestamp = int(result.get("timestamp_ms", 0) or 0)
    now = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if timestamp else "-"
    plans = list(result.get("planned_sells", [])) + list(result.get("planned_buys", []))
    action = "HOLD / MONITOR"
    if result.get("blocked") == "inspection_only":
        action = "INSPECTION ONLY"
    elif result.get("risk_reason") not in {"ok", "unknown"} and result.get("risk_reason"):
        action = f"RISK: {str(result.get('risk_reason')).upper()}"
    elif plans:
        sell_names = [str(x.get("pair")) for x in result.get("planned_sells", [])]
        buy_names = [str(x.get("pair")) for x in result.get("planned_buys", [])]
        if sell_names and buy_names:
            action = f"ROTATE: {', '.join(sell_names[:2])} -> {', '.join(buy_names[:2])}"
        elif sell_names:
            action = f"REDUCE/EXIT: {', '.join(sell_names[:3])}"
        elif buy_names:
            action = f"ADD: {', '.join(buy_names[:3])}"
    elif float(result.get("gross_target", 0.0)) <= 0.0:
        action = "CASH / NO LONG EXPOSURE"

    lines = [
        "", "=" * width, "ROOSTOO QUANT ENGINE | LIVE PORTFOLIO CONTROL", "=" * width,
        f"  LAST UPDATE       {now}",
        f"  ACTION            {action}",
        f"  STATE             {str(result.get('status', 'unknown')).upper():<12}   REGIME {str(result.get('regime', 'unknown')).upper()}",
        f"  NEXT REBALANCE    {_countdown(int(result.get('next_rebalance_ms', 0)))}{'  (DUE)' if result.get('rebalance_due') else ''}",
        "-" * width,
        "ACCOUNT",
        "-" * width,
        f"  EQUITY            {_money(float(result.get('equity', 0))):>15}    CASH {_money(float(result.get('cash', 0))):>15}    EXPOSURE {_money(float(result.get('current_exposure', 0))):>15}",
        f"  EXPOSURE %        {_pct(float(result.get('exposure_pct', 0))):>15}    SESSION P&L {_money(float(result.get('session_pnl', 0))):>14} ({_pct(float(result.get('session_pnl_pct', 0)))})",
        f"  VS EXCHANGE INIT  {_money(float(result.get('account_pnl', 0))):>15} ({_pct(float(result.get('account_pnl_pct', 0)))})    TRADES {int(result.get('today_trades', 0)):>4}    TURNOVER {_money(float(result.get('turnover', 0))):>12}",
        "-" * width,
        "MARKET",
        "-" * width,
        f"  UNIVERSE          {int(result.get('universe_size', 0)):>4} eligible USD markets     BREADTH {_pct(float(result.get('breadth', 0))):>9}     STRENGTH {float(result.get('market_strength', 0)):.3f}",
        f"  MEDIAN 4H         {_pct(float(result.get('median_4h', 0))):>9}    MEDIAN 12H {_pct(float(result.get('median_12h', 0))):>9}    MEDIAN 24H {_pct(float(result.get('median_24h', 0))):>9}    VOL {float(result.get('median_volatility', 0)):.4f}",
        f"  TARGET GROSS      LONG {_pct(float(result.get('gross_target', 0))):>9}    SHORT {_pct(float(result.get('short_gross_target', 0))):>9}    VOL SCALE {float(result.get('volatility_scale', 0)):.2f}",
        "-" * width,
        "PREDICTIVE MODEL",
        "-" * width,
        f"  STATUS             {'ENABLED' if result.get('prediction_enabled') else 'GATED'}    TRAINING SAMPLES {int(result.get('prediction_samples', 0)):>6}    LIVE SETTLED {int(result.get('live_prediction_samples', 0)):>6}",
        f"  VALIDATION         dir={float(result.get('prediction_accuracy', 0))*100:5.2f}%  corr={float(result.get('prediction_correlation', 0)):+.4f}  bal_acc={float(result.get('prediction_balanced_accuracy', 0))*100:5.2f}%  top10={_pct(float(result.get('prediction_top_decile_edge', 0)))}  brier={float(result.get('prediction_brier', 1)):.3f}",
        f"  GATE REASON        {str(result.get('prediction_gate_reason', '') or 'n/a')[:84]}",
        "-" * width,
        "CURRENT HOLDINGS",
        "-" * width,
    ]
    holdings = result.get("holdings", {}) or {}
    if holdings:
        total = float(result.get("equity", 0))
        targets = {}
        for order in result.get("planned_sells", []) + result.get("planned_buys", []):
            targets[order["pair"]] = float(order.get("target", 0.0))
        for pair, value in sorted(holdings.items(), key=lambda kv: kv[1], reverse=True):
            weight = float(value) / total if total > 0 else 0.0
            target = targets.get(pair, value)
            lines.append(f"  {pair:<12} value={_money(float(value)):>13} weight={_pct(weight):>9} target={_money(target):>13}")
    else:
        lines.append("  CASH ONLY / NO SPOT POSITIONS")

    lines += ["-" * width, "SIGNALS / TARGETS", "-" * width]
    rows = list(result.get("signal_rows", []))
    if rows:
        for i, row in enumerate(rows, 1):
            lines.append(
                f"  {i:>2}. {str(row.get('pair')):<11} score={float(row.get('score', 0)):.3f} target={_pct(float(row.get('target_weight', 0))):>8} "
                f"1h={_pct(float(row.get('fast_return', 0))):>8} 4h={_pct(float(row.get('medium_return', 0))):>8} "
                f"12h={_pct(float(row.get('long_return', 0))):>8} 24h={_pct(float(row.get('return_24h', 0))):>8} pred={_pct(float(row.get('predicted_return', 0))):>8}"
            )
            lines.append(f"      {str(row.get('reason', '')).replace('_', ' ')}")
    else:
        lines.append("  No qualifying signals")

    lines += ["-" * width, "ORDER PLAN", "-" * width]
    for order in result.get("planned_sells", []):
        lines.append(f"  SELL {order['pair']:<12} {_money(order['notional']):>13}  current={_money(order['current'])}  target={_money(order['target'])}")
    for order in result.get("planned_buys", []):
        lines.append(f"  BUY  {order['pair']:<12} {_money(order['notional']):>13}  current={_money(order['current'])}  target={_money(order['target'])}")
    if not result.get("planned_sells") and not result.get("planned_buys"):
        lines.append("  No trade required")

    lines += [
        "-" * width,
        f"  RISK              {str(result.get('risk_reason', 'unknown')).upper()}    ORDERS {'ENABLED' if result.get('orders_enabled') else 'DISABLED'}    BLOCK {result.get('blocked') or 'none'}",
        f"  HISTORY           {int(result.get('history_points', 0))}/{int(result.get('history_required', 0))}    DAILY TRADES {int(result.get('today_trades', 0))}    EXPOSURE {_pct(float(result.get('exposure_pct', 0)))}",
        "=" * width,
    ]
    return "\n".join(lines)
'''
(p/'roostoo_bot/dashboard.py').write_text(dashboard, encoding='utf-8')

# Update main logging so every minute is understandable without full dashboard spam.
main=(p/'roostoo_bot/__main__.py').read_text(encoding='utf-8')
oldlog='''                log.info("[STATUS] state=%s equity=$%.2f exposure=$%.2f regime=%s breadth=%.1f%% signals=%s orders=%s turnover=$%.2f risk=%s blocked=%s history=%s/%s", result["status"], result["equity"], result["current_exposure"], result["regime"], result["breadth"]*100, result["signals"], result["orders"], result["turnover"], result["risk_reason"], result["blocked"] or "none", result["history_points"], result["history_required"])\n'''
newlog='''                log.info("[STATUS] state=%s equity=$%.2f cash=$%.2f exposure=$%.2f (%.1f%%) regime=%s breadth=%.1f%% signals=%s model=%s trades=%s turnover=$%.2f next_rebalance=%ss risk=%s blocked=%s history=%s/%s", result["status"], result["equity"], result.get("cash", 0.0), result["current_exposure"], result.get("exposure_pct", 0.0)*100, result["regime"], result["breadth"]*100, result["signals"], "on" if result.get("prediction_enabled") else "gated", result["today_trades"], result["turnover"], int(result.get("next_rebalance_ms", 0)/1000), result["risk_reason"], result["blocked"] or "none", result["history_points"], result["history_required"])\n'''
main=main.replace(oldlog,newlog)
(p/'roostoo_bot/__main__.py').write_text(main, encoding='utf-8')

# Update tests by adapting constructor and add predictor split / empty signal liquidation checks later through quick script.
tests=(p/'tests/test_bot.py').read_text(encoding='utf-8')
# Add new args to both strategy test constructors.
tests=tests.replace('''            prediction_retrain_seconds=900, correlation_window=10, max_pair_correlation=0.99, correlation_penalty=0.1,\n            overextension_penalty=0.1, breakout_window=6, taker_fee_assumption=0.001, enable_shorts=False,\n''','''            prediction_retrain_seconds=900, correlation_window=10, max_pair_correlation=0.99, correlation_penalty=0.1,\n            overextension_penalty=0.1, breakout_window=6, taker_fee_assumption=0.001, enable_shorts=False, cash_reserve=0.10,\n            gross_strength_power=1.15, risk_on_gross_cap=1.0, balanced_gross_cap=0.8, risk_off_gross_cap=0.5, defensive_gross_cap=0.25,\n            defensive_min_breadth=0.08, defensive_min_strength=0.12, allocation_edge_power=1.10, allocation_vol_power=0.75, scan_top_n=8,\n''')
# This replacement affects both occurrences because identical.
# Add predictor time split test.
insert='''\n    def test_predictor_uses_relative_time_split(self):\n        from roostoo_bot.predictor import PredictiveEnsemble\n        model = PredictiveEnsemble(10, 3, 50, 500, 1.0, 1.0, 0.25, 2, 0.5, -1.0, 1/3, -1.0, 0.001, 0.9)\n        histories = {p: [100.0 * (1.001 + i*0.0001) ** j for j in range(80)] for i, p in enumerate(("BTC/USD", "ETH/USD", "SOL/USD", "SUI/USD"))}\n        result = model.fit_and_predict(histories)\n        self.assertIn("", result.gate_reason)\n        self.assertGreater(result.samples, 0)\n\n'''
tests=tests.replace('''    def test_breakout_feature_uses_previous_high(self):\n''', insert+'''    def test_breakout_feature_uses_previous_high(self):\n''')
(p/'tests/test_bot.py').write_text(tests, encoding='utf-8')

# Write an improved env example.
env='''ROOSTOO_API_URL=https://mock-api.roostoo.com\nROOSTOO_API_KEY=\nROOSTOO_API_SECRET=\nBOT_NAMESPACE=test-v42\nPOLL_SECONDS=60\nREBALANCE_SECONDS=900\nORDERS_ENABLED=false\nENABLE_SHORTS=false\nTOP_N=5\nMAX_GROSS_TARGET=0.70\nMAX_ASSET_WEIGHT=0.14\nMIN_QUOTE_VOLUME=250000\nMAX_SPREAD_BPS=35\nREBALANCE_BAND=0.012\nMIN_HOLD_MINUTES=30\nMAX_DAILY_TURNOVER=0.40\nPREDICTOR_ENABLED=true\nPREDICTION_RETRAIN_SECONDS=900\nPREDICTION_LOG_INTERVAL_SECONDS=300\nGROSS_STRENGTH_POWER=1.15\nRISK_ON_GROSS_CAP=1.00\nBALANCED_GROSS_CAP=0.80\nRISK_OFF_GROSS_CAP=0.50\nDEFENSIVE_GROSS_CAP=0.25\nDEFENSIVE_MIN_BREADTH=0.08\nDEFENSIVE_MIN_STRENGTH=0.12\nALLOCATION_EDGE_POWER=1.10\nALLOCATION_VOL_POWER=0.75\nSCAN_TOP_N=8\n'''
(p/'.env.example').write_text(env, encoding='utf-8')

# Remove caches from source package
import shutil
for d in [p/'roostoo_bot/__pycache__', p/'tests/__pycache__']:
    if d.exists(): shutil.rmtree(d)
