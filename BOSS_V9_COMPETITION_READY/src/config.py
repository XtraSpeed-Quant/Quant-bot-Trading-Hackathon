from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=False)


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in os.getenv(name, default).split(",") if x.strip())


def minutes_to_bars(minutes: int | float, poll_seconds: int) -> int:
    """Convert a real-time window in minutes to the current sampling cadence."""
    return max(1, int(round(float(minutes) * 60.0 / max(int(poll_seconds), 1))))


@dataclass(frozen=True)
class Config:
    # Roostoo uses the same host for spot and v6 short endpoints. Never put keys in source control.
    api_url: str = os.getenv("ROOSTOO_API_URL", "https://mock-api.roostoo.com").rstrip("/")
    api_key: str = os.getenv("ROOSTOO_API_KEY", "")
    api_secret: str = os.getenv("ROOSTOO_API_SECRET", "")

    bot_namespace: str = os.getenv("BOT_NAMESPACE", "boss-v9-quant-hackathon")
    paper_mode: bool = _bool("PAPER_MODE", False)
    orders_enabled: bool = _bool("ORDERS_ENABLED", True)
    expected_starting_cash: float = _float("EXPECTED_STARTING_CASH", 100_000.0)

    poll_seconds: int = _int("POLL_SECONDS", 60)
    rebalance_seconds: int = _int("REBALANCE_SECONDS", 120)
    timeout_seconds: float = _float("TIMEOUT_SECONDS", 12.0)
    history_limit: int = _int("HISTORY_LIMIT", 2000)
    predictor_history_limit: int = _int("PREDICTOR_HISTORY_LIMIT", 25000)
    exchange_info_refresh_seconds: int = _int("EXCHANGE_INFO_REFRESH_SECONDS", 900)
    account_refresh_seconds: int = _int("ACCOUNT_REFRESH_SECONDS", 120)
    prediction_retrain_seconds: int = _int("PREDICTION_RETRAIN_SECONDS", 1800)
    prediction_log_interval_seconds: int = _int("PREDICTION_LOG_INTERVAL_SECONDS", 300)
    # Windows are expressed in minutes, not raw samples. They are scaled at runtime.
    top_n: int = _int("TOP_N", 20)
    max_gross_target: float = _float("MAX_GROSS_TARGET", 0.70)
    max_asset_weight: float = _float("MAX_ASSET_WEIGHT", 0.10)
    shorts_enabled: bool = _bool("SHORTS_ENABLED", True)
    short_top_n: int = _int("SHORT_TOP_N", 20)
    max_short_gross_target: float = _float("MAX_SHORT_GROSS_TARGET", 0.42)
    max_short_asset_weight: float = _float("MAX_SHORT_ASSET_WEIGHT", 0.075)
    max_total_gross_target: float = _float("MAX_TOTAL_GROSS_TARGET", 0.98)
    allow_shorts_risk_on: bool = _bool("ALLOW_SHORTS_RISK_ON", True)
    short_strength_power: float = _float("SHORT_STRENGTH_POWER", 1.25)
    cash_reserve: float = _float("CASH_RESERVE", 0.02)

    min_score: float = _float("MIN_SCORE", 0.51)
    min_short_score: float = _float("MIN_SHORT_SCORE", 0.53)
    min_relative_edge: float = _float("MIN_RELATIVE_EDGE", 0.0025)
    min_short_relative_edge: float = _float("MIN_SHORT_RELATIVE_EDGE", 0.0025)
    min_1h_return: float = _float("MIN_1H_RETURN", -0.010)
    min_4h_return: float = _float("MIN_4H_RETURN", -0.010)
    min_12h_return: float = _float("MIN_12H_RETURN", -0.020)
    max_short_1h_return: float = _float("MAX_SHORT_1H_RETURN", 0.0020)
    max_short_4h_return: float = _float("MAX_SHORT_4H_RETURN", -0.0020)
    max_short_12h_return: float = _float("MAX_SHORT_12H_RETURN", 0.0030)
    min_quote_volume: float = _float("MIN_QUOTE_VOLUME", 250_000.0)
    max_spread_bps: float = _float("MAX_SPREAD_BPS", 20.0)

    fast_window: int = _int("FAST_WINDOW", 60)
    medium_window: int = _int("MEDIUM_WINDOW", 240)
    long_window: int = _int("LONG_WINDOW", 720)
    trend_window: int = _int("TREND_WINDOW", 360)
    vol_window: int = _int("VOL_WINDOW", 60)
    breakout_window: int = _int("BREAKOUT_WINDOW", 240)

    rebalance_band: float = _float("REBALANCE_BAND", 0.0)
    min_order_usd: float = _float("MIN_ORDER_USD", 1.0)
    max_order_usd: float = _float("MAX_ORDER_USD", 0.0)
    # Zero disables the internal turnover cap. A positive value remains available as an optional limit.
    max_daily_turnover: float = _float("MAX_DAILY_TURNOVER", 0.0)
    min_hold_minutes: int = _int("MIN_HOLD_MINUTES", 0)
    commission_rate: float = _float("TAKER_FEE_ASSUMPTION", 0.001)
    maker_fee_assumption: float = _float("MAKER_FEE_ASSUMPTION", 0.0005)
    short_fee_assumption: float = _float("SHORT_FEE_ASSUMPTION", 0.001)

    stop_loss_pct: float = _float("STOP_LOSS_PCT", 0.025)
    trailing_activation_pct: float = _float("TRAILING_ACTIVATION_PCT", 0.018)
    trailing_stop_pct: float = _float("TRAILING_STOP_PCT", 0.009)
    proactive_loss_pct: float = _float("PROACTIVE_LOSS_PCT", 0.012)
    max_loss_before_stop_pct: float = _float("MAX_LOSS_BEFORE_STOP_PCT", 0.025)
    take_profit_pct: float = _float("TAKE_PROFIT_PCT", 0.035)
    short_stop_loss_pct: float = _float("SHORT_STOP_LOSS_PCT", 0.030)
    short_trailing_activation_pct: float = _float("SHORT_TRAILING_ACTIVATION_PCT", 0.020)
    short_trailing_stop_pct: float = _float("SHORT_TRAILING_STOP_PCT", 0.010)
    adaptive_exits_enabled: bool = _bool("ADAPTIVE_EXITS_ENABLED", True)
    stop_vol_multiple: float = _float("STOP_VOL_MULTIPLE", 2.5)
    trailing_activation_vol_multiple: float = _float("TRAILING_ACTIVATION_VOL_MULTIPLE", 1.8)
    trailing_stop_vol_multiple: float = _float("TRAILING_STOP_VOL_MULTIPLE", 1.0)
    take_profit_vol_multiple: float = _float("TAKE_PROFIT_VOL_MULTIPLE", 3.5)
    short_stop_vol_multiple: float = _float("SHORT_STOP_VOL_MULTIPLE", 3.0)
    short_trailing_activation_vol_multiple: float = _float("SHORT_TRAILING_ACTIVATION_VOL_MULTIPLE", 2.0)
    short_trailing_stop_vol_multiple: float = _float("SHORT_TRAILING_STOP_VOL_MULTIPLE", 1.0)
    loss_cooldown_minutes: int = _int("LOSS_COOLDOWN_MINUTES", 0)
    daily_loss_limit_pct: float = _float("DAILY_LOSS_LIMIT_PCT", 0.0)
    max_drawdown_limit_pct: float = _float("MAX_DRAWDOWN_LIMIT_PCT", 0.0)

    defensive_min_breadth: float = _float("DEFENSIVE_MIN_BREADTH", 0.08)
    defensive_min_strength: float = _float("DEFENSIVE_MIN_STRENGTH", 0.12)
    risk_on_gross_cap: float = _float("RISK_ON_GROSS_CAP", 1.00)
    balanced_gross_cap: float = _float("BALANCED_GROSS_CAP", 0.90)
    risk_off_gross_cap: float = _float("RISK_OFF_GROSS_CAP", 0.75)
    defensive_gross_cap: float = _float("DEFENSIVE_GROSS_CAP", 0.50)
    gross_strength_power: float = _float("GROSS_STRENGTH_POWER", 1.15)
    allocation_edge_power: float = _float("ALLOCATION_EDGE_POWER", 1.10)
    allocation_vol_power: float = _float("ALLOCATION_VOL_POWER", 0.75)
    # Joint long/short allocation objective; no turnover quota is imposed.
    portfolio_risk_aversion: float = _float("PORTFOLIO_RISK_AVERSION", 60.0)
    portfolio_cost_penalty: float = _float("PORTFOLIO_COST_PENALTY", 1.0)
    portfolio_covariance_window_minutes: int = _int("PORTFOLIO_COVARIANCE_WINDOW_MINUTES", 360)
    portfolio_covariance_half_life_minutes: int = _int("PORTFOLIO_COVARIANCE_HALF_LIFE_MINUTES", 180)
    portfolio_risk_horizon_minutes: int = _int("PORTFOLIO_RISK_HORIZON_MINUTES", 60)
    portfolio_covariance_shrinkage: float = _float("PORTFOLIO_COVARIANCE_SHRINKAGE", 0.65)
    portfolio_optimizer_iterations: int = _int("PORTFOLIO_OPTIMIZER_ITERATIONS", 250)
    correlation_window: int = _int("CORRELATION_WINDOW", 240)
    max_pair_correlation: float = _float("MAX_PAIR_CORRELATION", 0.90)
    correlation_penalty: float = _float("CORRELATION_PENALTY", 0.12)
    overextension_penalty: float = _float("OVEREXTENSION_PENALTY", 0.06)

    predictor_enabled: bool = _bool("PREDICTOR_ENABLED", True)
    # ML horizon and all feature windows are real minutes and cadence-scaled.
    prediction_horizon: int = _int("PREDICTION_HORIZON", 60)
    predictor_feature_window: int = _int("PREDICTOR_FEATURE_WINDOW", 360)
    predictor_min_samples: int = _int("PREDICTOR_MIN_SAMPLES", 5000)
    predictor_max_samples: int = _int("PREDICTOR_MAX_SAMPLES", 60_000)
    predictor_ridge_alpha: float = _float("PREDICTOR_RIDGE_ALPHA", 4.0)
    predictor_classifier_alpha: float = _float("PREDICTOR_CLASSIFIER_ALPHA", 6.0)
    predictor_validation_fraction: float = _float("PREDICTOR_VALIDATION_FRACTION", 0.25)
    predictor_purge_gap: int = _int("PREDICTOR_PURGE_GAP", 60)
    predictor_min_directional_accuracy: float = _float("PREDICTOR_MIN_DIRECTIONAL_ACCURACY", 0.515)
    predictor_min_correlation: float = _float("PREDICTOR_MIN_CORRELATION", 0.01)
    predictor_min_rank_ic: float = _float("PREDICTOR_MIN_RANK_IC", 0.01)
    predictor_min_balanced_accuracy: float = _float("PREDICTOR_MIN_BALANCED_ACCURACY", 0.36)
    predictor_min_top_decile_edge: float = _float("PREDICTOR_MIN_TOP_DECILE_EDGE", 0.00025)
    predictor_min_top_decile_net_return: float = _float("PREDICTOR_MIN_TOP_DECILE_NET_RETURN", 0.00025)
    slippage_bps: float = _float("SLIPPAGE_BPS", _float("PREDICTOR_SLIPPAGE_BPS", 10.0))
    predictor_max_brier: float = _float("PREDICTOR_MAX_BRIER", 0.23)
    predictor_class_threshold: float = _float("PREDICTOR_CLASS_THRESHOLD", 0.0015)
    predictor_retrain_seconds: int = _int("PREDICTION_RETRAIN_SECONDS", 1800)
    prediction_log_interval_seconds: int = _int("PREDICTION_LOG_INTERVAL_SECONDS", 300)
    prediction_log_top_n: int = _int("PREDICTION_LOG_TOP_N", 100)
    bearish_veto_enabled: bool = _bool("PREDICTOR_BEARISH_VETO_ENABLED", True)
    bearish_veto_return: float = _float("PREDICTOR_BEARISH_VETO_RETURN", -0.005)
    bearish_veto_probability: float = _float("PREDICTOR_BEARISH_VETO_PROBABILITY", 0.60)
    bearish_veto_confidence: float = _float("PREDICTOR_BEARISH_VETO_CONFIDENCE", 0.60)

    rotation_enabled: bool = _bool("ROTATION_ENABLED", True)
    rotation_score_gap: float = _float("ROTATION_SCORE_GAP", 0.02)
    rotation_max_per_rebalance: int = _int("ROTATION_MAX_PER_REBALANCE", 0)

    allow_pairs: tuple[str, ...] = _csv("ALLOW_PAIRS")
    log_file: str = os.getenv("LOG_FILE", "logs/bot.log")
    db_path: str = os.getenv("BOT_DB_PATH", "data/boss_v9_live.sqlite3")
    # Minimum spacing between outbound requests; unrelated to turnover/order count.
    api_min_request_interval_seconds: float = _float("API_MIN_REQUEST_INTERVAL_SECONDS", 0.20)
    # Used only for dashboard diagnostics; it never truncates the tradable universe.
    diagnostic_top_n: int = _int("DIAGNOSTIC_TOP_N", _int("SCAN_TOP_N", 100))
    target_hourly_vol: float = _float("TARGET_HOURLY_VOL", 0.010)

    def bars(self, minutes: int | float) -> int:
        return minutes_to_bars(minutes, self.poll_seconds)

    def validate(self, require_credentials: bool = True, require_api_url: bool = True) -> None:
        if require_credentials and not self.paper_mode and (not self.api_key or not self.api_secret):
            raise ValueError("ROOSTOO_API_KEY and ROOSTOO_API_SECRET must be set in .env")
        if require_api_url and not self.api_url:
            raise ValueError("ROOSTOO_API_URL must be set to the competition's actual API host in .env")

        numeric = [
            self.expected_starting_cash, self.poll_seconds, self.rebalance_seconds, self.timeout_seconds,
            self.history_limit, self.predictor_history_limit, self.exchange_info_refresh_seconds,
            self.account_refresh_seconds, self.prediction_retrain_seconds, self.prediction_log_interval_seconds,
            self.api_min_request_interval_seconds, self.maker_fee_assumption, self.short_fee_assumption,
            self.top_n, self.diagnostic_top_n, self.max_gross_target, self.max_asset_weight,
            self.max_short_gross_target, self.max_short_asset_weight, self.max_total_gross_target,
            self.short_top_n, self.cash_reserve, self.min_score, self.min_short_score,
            self.min_relative_edge, self.min_short_relative_edge, self.min_1h_return, self.min_4h_return,
            self.min_12h_return, self.max_short_1h_return, self.max_short_4h_return,
            self.max_short_12h_return, self.min_quote_volume, self.max_spread_bps, self.fast_window,
            self.medium_window, self.long_window, self.trend_window, self.vol_window, self.breakout_window,
            self.rebalance_band, self.min_order_usd, self.max_order_usd, self.max_daily_turnover,
            self.min_hold_minutes, self.commission_rate, self.stop_loss_pct, self.trailing_activation_pct,
            self.daily_loss_limit_pct, self.max_drawdown_limit_pct, self.slippage_bps, self.target_hourly_vol,
            self.trailing_stop_pct, self.proactive_loss_pct, self.max_loss_before_stop_pct,
            self.take_profit_pct, self.short_stop_loss_pct, self.short_trailing_activation_pct,
            self.short_trailing_stop_pct, self.stop_vol_multiple, self.trailing_activation_vol_multiple,
            self.trailing_stop_vol_multiple, self.take_profit_vol_multiple, self.short_stop_vol_multiple,
            self.short_trailing_activation_vol_multiple, self.short_trailing_stop_vol_multiple,
            self.loss_cooldown_minutes, self.defensive_min_breadth,
            self.defensive_min_strength, self.risk_on_gross_cap, self.balanced_gross_cap,
            self.risk_off_gross_cap, self.defensive_gross_cap, self.gross_strength_power,
            self.short_strength_power, self.allocation_edge_power, self.allocation_vol_power,
            self.portfolio_risk_aversion, self.portfolio_cost_penalty,
            self.portfolio_covariance_window_minutes, self.portfolio_risk_horizon_minutes,
            self.portfolio_covariance_shrinkage, self.portfolio_optimizer_iterations,
            self.correlation_window, self.max_pair_correlation, self.correlation_penalty,
            self.overextension_penalty, self.prediction_horizon, self.predictor_feature_window,
            self.predictor_min_samples, self.predictor_max_samples, self.predictor_ridge_alpha,
            self.predictor_classifier_alpha, self.predictor_validation_fraction, self.predictor_purge_gap,
            self.predictor_min_directional_accuracy, self.predictor_min_correlation,
            self.predictor_min_rank_ic, self.predictor_min_balanced_accuracy,
            self.predictor_min_top_decile_edge, self.predictor_min_top_decile_net_return,
            self.predictor_max_brier, self.predictor_class_threshold, self.predictor_retrain_seconds,
            self.prediction_log_interval_seconds, self.prediction_log_top_n, self.bearish_veto_return,
            self.bearish_veto_probability, self.bearish_veto_confidence, self.rotation_score_gap,
            self.rotation_max_per_rebalance,
        ]
        if any(not math.isfinite(float(v)) for v in numeric):
            raise ValueError("Non-finite configuration value")
        if self.expected_starting_cash <= 0:
            raise ValueError("EXPECTED_STARTING_CASH must be positive")
        if self.poll_seconds < 1:
            raise ValueError("POLL_SECONDS must be at least 1 second")
        if self.rebalance_seconds < self.poll_seconds:
            raise ValueError("REBALANCE_SECONDS must be >= POLL_SECONDS")
        if not (2 <= self.fast_window < self.medium_window < self.long_window):
            raise ValueError("Require FAST_WINDOW < MEDIUM_WINDOW < LONG_WINDOW (minutes)")
        required_history = max(
            self.bars(self.long_window), self.bars(self.trend_window), self.bars(self.vol_window),
            self.bars(self.breakout_window), self.bars(self.predictor_feature_window) + self.bars(self.prediction_horizon),
        ) + 2
        if self.history_limit < min(required_history, self.bars(self.long_window) + 2):
            raise ValueError(f"HISTORY_LIMIT too small; need at least {self.bars(self.long_window) + 2} samples at this poll interval")
        if self.predictor_history_limit < self.bars(self.predictor_feature_window) + self.bars(self.prediction_horizon) + 2:
            raise ValueError("PREDICTOR_HISTORY_LIMIT must cover feature plus prediction horizon")
        if self.exchange_info_refresh_seconds < 1 or self.account_refresh_seconds < 1:
            raise ValueError("EXCHANGE_INFO_REFRESH_SECONDS and ACCOUNT_REFRESH_SECONDS must be positive")
        if self.prediction_retrain_seconds < 1 or self.prediction_log_interval_seconds < 1:
            raise ValueError("Prediction retrain/log intervals must be positive")
        if not (0 <= self.maker_fee_assumption < 0.1 and 0 <= self.short_fee_assumption < 0.1):
            raise ValueError("Fee assumptions must be plausible nonnegative fractions")
        if self.top_n < 1 or self.short_top_n < 1:
            raise ValueError("TOP_N and SHORT_TOP_N must be positive")
        if self.diagnostic_top_n < 1:
            raise ValueError("DIAGNOSTIC_TOP_N must be positive")
        if not (0 < self.max_gross_target <= 1 and 0 < self.max_asset_weight <= 1):
            raise ValueError("Long gross and per-asset weight must be in (0,1]")
        if self.top_n * self.max_asset_weight + 1e-12 < self.max_gross_target:
            raise ValueError("TOP_N * MAX_ASSET_WEIGHT must cover MAX_GROSS_TARGET")
        if not (0 <= self.max_short_gross_target <= 1 and 0 < self.max_short_asset_weight <= 1):
            raise ValueError("Short gross and per-asset weight are invalid")
        if self.short_top_n * self.max_short_asset_weight + 1e-12 < self.max_short_gross_target:
            raise ValueError("SHORT_TOP_N * MAX_SHORT_ASSET_WEIGHT must cover MAX_SHORT_GROSS_TARGET")
        if self.max_total_gross_target <= 0 or self.max_total_gross_target > 1 - self.cash_reserve + 1e-12:
            raise ValueError("MAX_TOTAL_GROSS_TARGET must be positive and leave the configured cash reserve")
        if not (0 <= self.rebalance_band < 1):
            raise ValueError("REBALANCE_BAND must be in [0,1)")
        if self.max_order_usd < 0 or self.min_order_usd <= 0 or (0 < self.max_order_usd < self.min_order_usd):
            raise ValueError("MAX_ORDER_USD must be 0 (unlimited) or at least MIN_ORDER_USD")
        if not (0 < self.cash_reserve < 1):
            raise ValueError("CASH_RESERVE must be in (0,1)")
        if min(self.stop_vol_multiple, self.trailing_activation_vol_multiple, self.trailing_stop_vol_multiple,
               self.take_profit_vol_multiple, self.short_stop_vol_multiple,
               self.short_trailing_activation_vol_multiple, self.short_trailing_stop_vol_multiple) <= 0:
            raise ValueError("Adaptive exit volatility multiples must be positive")
        if self.max_daily_turnover < 0:
            raise ValueError("MAX_DAILY_TURNOVER must be zero (unlimited) or positive")
        if self.daily_loss_limit_pct < 0 or self.max_drawdown_limit_pct < 0:
            raise ValueError("Portfolio halt thresholds must be zero (disabled) or positive fractions")
        if self.slippage_bps < 0 or self.target_hourly_vol <= 0:
            raise ValueError("Invalid predictor slippage or hourly volatility target")
        if self.portfolio_risk_aversion < 0 or self.portfolio_cost_penalty < 0:
            raise ValueError("Portfolio risk aversion and cost penalty must be nonnegative")
        if (self.portfolio_covariance_window_minutes < 60 or self.portfolio_risk_horizon_minutes < 1
                or self.portfolio_covariance_half_life_minutes < 1):
            raise ValueError("Portfolio covariance window/horizon/half-life values are invalid")
        if not (0 <= self.portfolio_covariance_shrinkage <= 1) or self.portfolio_optimizer_iterations < 10:
            raise ValueError("Invalid portfolio covariance shrinkage or optimizer iterations")
        if self.prediction_log_interval_seconds <= 0:
            raise ValueError("PREDICTION_LOG_INTERVAL_SECONDS must be positive")
        if self.predictor_min_samples < 200 or self.predictor_max_samples < self.predictor_min_samples:
            raise ValueError("Invalid predictor sample limits")
        if not (0.10 <= self.predictor_validation_fraction < 0.50):
            raise ValueError("Invalid validation fraction")
        if not (0.50 <= self.predictor_min_directional_accuracy <= 1.0):
            raise ValueError("Invalid directional-accuracy threshold")
        if not (1/3 <= self.predictor_min_balanced_accuracy <= 1.0):
            raise ValueError("Invalid balanced-accuracy threshold")
        if not (0 < self.predictor_max_brier <= 1):
            raise ValueError("Invalid Brier threshold")
        if self.predictor_purge_gap < 0 or self.prediction_horizon <= 0:
            raise ValueError("Invalid predictor horizon/purge")
        if not (0 < self.bearish_veto_probability <= 1 and 0 <= self.bearish_veto_confidence <= 1):
            raise ValueError("Invalid bearish veto thresholds")
        if self.api_url and not self.api_url.startswith(("http://", "https://")):
            raise ValueError("ROOSTOO_API_URL must be HTTP(S)")
        if self.api_min_request_interval_seconds < 0:
            raise ValueError("API_MIN_REQUEST_INTERVAL_SECONDS must be nonnegative")
        # Roostoo documents mock-api.roostoo.com as the official REST host; the hackathon
        # itself is hosted on this real-time mock exchange. Local hosts are allowed for paper tests.
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
