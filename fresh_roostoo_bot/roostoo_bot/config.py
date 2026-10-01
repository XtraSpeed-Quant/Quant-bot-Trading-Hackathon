from __future__ import annotations

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
