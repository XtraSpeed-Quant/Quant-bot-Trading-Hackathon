from __future__ import annotations

import math
from collections import Counter
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..config import Config
from ..predictor import Prediction, PredictorResult, PredictiveEnsemble
from .portfolio_optimizer import optimize_signed_portfolio


@dataclass(frozen=True)
class Market:
    pair: str
    coin: str
    price: float
    bid: float
    ask: float
    change_24h: float
    quote_volume_24h: float
    amount_precision: int
    min_order: float
    can_trade: bool

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        if mid <= 0 or self.ask <= 0 or self.bid <= 0:
            return math.inf
        return (self.ask - self.bid) / mid * 10_000.0


@dataclass(frozen=True)
class Signal:
    pair: str
    score: float
    fast_return: float
    medium_return: float
    long_return: float
    return_24h: float
    trend: float
    volatility: float
    relative_4h: float
    relative_12h: float
    pullback: float
    breakout: float
    overextension: float
    predicted_return: float
    probability_up: float
    probability_down: float
    prediction_confidence: float
    expected_edge: float
    target_weight: float = 0.0
    correlation_penalty: float = 0.0
    bearish_veto: bool = False
    reason: str = ""
    direction: str = "LONG"
    gross_expected_edge: float = 0.0
    round_trip_cost: float = 0.0
    one_way_cost: float = 0.0


@dataclass(frozen=True)
class StrategyResult:
    long_targets: dict[str, float]
    signals: list[Signal]
    all_scores: dict[str, float]
    bearish_veto_pairs: tuple[str, ...]
    rejected: list[dict[str, object]]
    regime: str
    breadth: float
    median_4h: float
    median_12h: float
    median_24h: float
    market_strength: float
    gross_target: float
    median_volatility: float
    volatility_scale: float
    predictor_enabled: bool
    predictor_samples: int
    predictor_directional_accuracy: float
    predictor_correlation: float
    predictor_rank_ic: float
    predictor_balanced_accuracy: float
    predictor_top_decile_edge: float
    predictor_top_decile_net_return: float
    predictor_brier: float
    predictor_gate_reason: str
    predictor_predictions: dict[str, Prediction] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)
    short_targets: dict[str, float] = field(default_factory=dict)
    short_signals: list[Signal] = field(default_factory=list)
    predictor_directional_enabled: bool = False
    short_gross_target: float = 0.0


class BossStrategy:
    """Cross-sectional momentum strategy with independently gated long and short books."""

    def __init__(self, cfg: Config, shorts_available: bool = True):
        self.cfg = cfg
        self.shorts_available = bool(shorts_available)
        self.predictor = PredictiveEnsemble(
            feature_window=cfg.predictor_feature_window,
            prediction_horizon=cfg.prediction_horizon,
            min_samples=cfg.predictor_min_samples,
            max_training_samples=cfg.predictor_max_samples,
            ridge_alpha=cfg.predictor_ridge_alpha,
            classifier_alpha=cfg.predictor_classifier_alpha,
            validation_fraction=cfg.predictor_validation_fraction,
            purge_gap=cfg.predictor_purge_gap,
            min_directional_accuracy=cfg.predictor_min_directional_accuracy,
            min_correlation=cfg.predictor_min_correlation,
            min_rank_ic=cfg.predictor_min_rank_ic,
            min_balanced_accuracy=cfg.predictor_min_balanced_accuracy,
            min_top_decile_edge=cfg.predictor_min_top_decile_edge,
            class_threshold=cfg.predictor_class_threshold,
            fee_assumption=cfg.commission_rate,
            max_brier=cfg.predictor_max_brier,
            min_top_decile_net_return=cfg.predictor_min_top_decile_net_return,
            poll_seconds=cfg.poll_seconds,
            slippage_bps=cfg.slippage_bps,
        ) if cfg.predictor_enabled else None
        self._last_predictor_fit_ms = 0
        self._last_predictor_result: PredictorResult | None = None

    def _bars(self, minutes: int | float) -> int:
        return self.cfg.bars(minutes)

    @staticmethod
    def _ema(values: Sequence[float], window: int) -> float:
        if len(values) < window:
            return math.nan
        alpha = 2.0 / (window + 1.0)
        value = float(values[0])
        for p in values[1:]:
            value = alpha * float(p) + (1 - alpha) * value
        return value

    def _rolling_vol_bars(self, values: Sequence[float], window: int) -> float:
        if len(values) < window + 1:
            return math.nan
        arr = np.asarray(values[-window - 1:], dtype=float)
        returns = np.diff(np.log(np.maximum(arr, 1e-12)))
        return float(np.std(returns) * math.sqrt(max(window, 1)))

    def _rolling_vol(self, values: Sequence[float], window_minutes: int) -> float:
        return self._rolling_vol_bars(values, self._bars(window_minutes))

    @staticmethod
    def _pct_return_bars(values: Sequence[float], window: int) -> float:
        if len(values) <= window:
            return math.nan
        return float(values[-1]) / max(float(values[-window - 1]), 1e-12) - 1.0

    def _pct_return(self, values: Sequence[float], window_minutes: int) -> float:
        return self._pct_return_bars(values, self._bars(window_minutes))

    @staticmethod
    def _previous_high_bars(values: Sequence[float], window: int) -> float:
        if len(values) <= window:
            return math.nan
        arr = np.asarray(values[-window - 1:-1], dtype=float)
        return float(np.max(arr)) if len(arr) else math.nan

    def _previous_high(self, values: Sequence[float], window_minutes: int) -> float:
        return self._previous_high_bars(values, self._bars(window_minutes))

    @staticmethod
    def _previous_low_bars(values: Sequence[float], window: int) -> float:
        if len(values) <= window:
            return math.nan
        arr = np.asarray(values[-window - 1:-1], dtype=float)
        return float(np.min(arr)) if len(arr) else math.nan

    def _previous_low(self, values: Sequence[float], window_minutes: int) -> float:
        return self._previous_low_bars(values, self._bars(window_minutes))

    @staticmethod
    def _rank01(values: Mapping[str, float]) -> dict[str, float]:
        if not values:
            return {}
        items = sorted(values.items(), key=lambda kv: (float(kv[1]), kv[0]))
        n = len(items)
        if n == 1:
            return {items[0][0]: 1.0}
        out = {}
        for i, (key, _) in enumerate(items):
            out[key] = i / (n - 1)
        return out

    @staticmethod
    def _safe_corr(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) < 2 or len(b) < 2:
            return 0.0
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
            return 0.0
        value = float(np.corrcoef(aa, bb)[0, 1])
        return value if math.isfinite(value) else 0.0

    def predictor_training_due(self, now_ms: int,
                               histories: Mapping[str, Sequence[float]] | None = None) -> bool:
        """Whether the full persisted training window should be loaded for a refit."""
        if self.predictor is None:
            return False
        if histories is not None and not any(
            len(values) >= self._bars(self.cfg.fast_window) + 1 for values in histories.values()
        ):
            return False
        return (self._last_predictor_result is None or
                now_ms - self._last_predictor_fit_ms >= self.cfg.predictor_retrain_seconds * 1000)

    def _predictor(self, histories: Mapping[str, Sequence[float]], now_ms: int,
                   training_histories: Mapping[str, Sequence[float]] | None = None) -> PredictorResult:
        if self.predictor is None:
            return PredictorResult(
                enabled=False, samples=0, validation_directional_accuracy=0.0,
                validation_correlation=0.0, validation_balanced_accuracy=0.0,
                validation_top_decile_edge=0.0, validation_brier=1.0, residual_std=0.0,
                predictions={}, gate_reason="disabled", validation_rank_ic=0.0,
                directional_enabled=False,
            )
        refit = self._last_predictor_result is None or now_ms - self._last_predictor_fit_ms >= self.cfg.predictor_retrain_seconds * 1000
        result = self.predictor.fit_and_predict(
            histories, refit=refit, fit_timestamp_ms=now_ms,
            training_histories=training_histories,
        )
        if refit:
            self._last_predictor_fit_ms = now_ms
        self._last_predictor_result = result
        return result

    def _features(self, market: Market, history: Sequence[float]) -> dict[str, float] | None:
        # Warm up on the fastest full horizon, then lengthen the 4h/12h features as
        # clean live observations accumulate. This avoids a forced 12-hour no-trade
        # period on a fresh AWS database without pretending partial windows are full windows.
        if len(history) < self._bars(self.cfg.fast_window) + 1:
            return None
        available = len(history) - 1
        fast_bars = min(self._bars(self.cfg.fast_window), available)
        medium_bars = min(self._bars(self.cfg.medium_window), available)
        long_bars = min(self._bars(self.cfg.long_window), available)
        trend_bars = min(self._bars(self.cfg.trend_window), available)
        vol_bars = min(self._bars(self.cfg.vol_window), available)
        breakout_bars = min(self._bars(self.cfg.breakout_window), available)
        price = float(history[-1])
        fast = self._pct_return_bars(history, fast_bars)
        medium = self._pct_return_bars(history, medium_bars)
        long_ret = self._pct_return_bars(history, long_bars)
        trend_ema = self._ema(history, trend_bars)
        vol = self._rolling_vol_bars(history, vol_bars)
        high_prev = self._previous_high_bars(history, breakout_bars)
        low_prev = self._previous_low_bars(history, breakout_bars)
        mean_fast = float(np.mean(np.asarray(history[-fast_bars:], dtype=float)))
        if any(not math.isfinite(v) for v in [fast, medium, long_ret, trend_ema, vol, high_prev, low_prev, mean_fast]):
            return None
        trend = price / max(trend_ema, 1e-12) - 1.0
        breakout = price / max(high_prev, 1e-12) - 1.0
        breakdown = low_prev / max(price, 1e-12) - 1.0
        extension = max(0.0, fast - 0.35 * max(medium, 0.0))
        downside_extension = max(0.0, -fast - 0.35 * max(-medium, 0.0))
        pullback = max(0.0, -fast) if medium > 0 and trend > 0 else 0.0
        bearish_bounce = max(0.0, fast) if medium < 0 and trend < 0 else 0.0
        mean_reversion = price / max(mean_fast, 1e-12) - 1.0
        return {
            "fast": fast, "medium": medium, "long": long_ret,
            "return_24h": float(market.change_24h), "trend": trend,
            "vol": max(vol, 0.0001), "liquidity": math.log1p(max(market.quote_volume_24h, 0.0)),
            "spread": market.spread_bps, "extension": extension,
            "downside_extension": downside_extension, "pullback": pullback,
            "bearish_bounce": bearish_bounce, "breakout": breakout,
            "breakdown": breakdown, "mean_reversion": mean_reversion,
        }

    @staticmethod
    def _market_strength(breadth: float, median_4h: float, median_12h: float, median_24h: float, median_vol: float) -> float:
        breadth_component = np.clip((breadth - 0.20) / 0.60, 0.0, 1.0)
        m4 = np.clip((median_4h + 0.008) / 0.028, 0.0, 1.0)
        m12 = np.clip((median_12h + 0.015) / 0.050, 0.0, 1.0)
        m24 = np.clip((median_24h + 0.025) / 0.070, 0.0, 1.0)
        vol_drag = np.clip((median_vol - 0.004) / 0.010, 0.0, 1.0)
        return float(np.clip(0.40 * breadth_component + 0.25 * m4 + 0.18 * m12 + 0.17 * m24 - 0.12 * vol_drag, 0.0, 1.0))

    @staticmethod
    def _regime(strength: float) -> str:
        if strength >= 0.72:
            return "risk_on"
        if strength >= 0.50:
            return "balanced"
        if strength >= 0.28:
            return "risk_off"
        return "defensive"

    def _pair_corr(self, histories: Mapping[str, Sequence[float]], a: str, b: str) -> float:
        va = np.asarray(histories.get(a, ()), dtype=float)
        vb = np.asarray(histories.get(b, ()), dtype=float)
        window = self._bars(self.cfg.correlation_window)
        if len(va) < window + 1 or len(vb) < window + 1:
            return 0.0
        ra = np.diff(np.log(np.maximum(va[-window - 1:], 1e-12)))
        rb = np.diff(np.log(np.maximum(vb[-window - 1:], 1e-12)))
        return self._safe_corr(ra, rb)

    def _diversify(self, candidates: list[Signal], histories: Mapping[str, Sequence[float]], limit: int) -> list[Signal]:
        selected: list[Signal] = []
        for candidate in candidates:
            if len(selected) >= limit:
                break
            if not selected:
                selected.append(candidate)
                continue
            corrs = [abs(self._pair_corr(histories, candidate.pair, item.pair)) for item in selected]
            max_corr = max(corrs, default=0.0)
            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            weakest = min(selected, key=lambda item: item.score)
            if max_corr > self.cfg.max_pair_correlation and candidate.score - weakest.score < 0.06:
                continue
            penalty = self.cfg.correlation_penalty * max(0.0, avg_corr - 0.50)
            selected.append(Signal(**{**candidate.__dict__, "score": max(0.0, candidate.score - penalty),
                                      "correlation_penalty": penalty,
                                      "reason": candidate.reason + ("+correlation_penalty" if penalty > 0 else "")}))
        return selected

    def _allocate(self, selected: list[Signal], total: float, cap: float, score_floor: float) -> dict[str, float]:
        if not selected or total <= 0:
            return {}
        raw = {}
        for signal in selected:
            edge = max(signal.score - score_floor, 0.015)
            conf_boost = 1.0 + 0.25 * signal.prediction_confidence if self._last_predictor_result and self._last_predictor_result.enabled else 1.0
            raw[signal.pair] = (edge ** self.cfg.allocation_edge_power) * conf_boost / max(signal.volatility, 1e-4) ** self.cfg.allocation_vol_power
        weights = {key: 0.0 for key in raw}
        active = set(raw)
        remaining = total
        while active and remaining > 1e-12:
            denom = sum(raw[pair] for pair in active)
            if denom <= 1e-12:
                break
            capped = set()
            for pair in list(active):
                proposed = remaining * raw[pair] / denom
                if proposed >= cap:
                    weights[pair] = cap
                    remaining -= cap
                    capped.add(pair)
            if not capped:
                for pair in active:
                    weights[pair] = remaining * raw[pair] / denom
                break
            active -= capped
        return {pair: float(weight) for pair, weight in weights.items() if weight > 1e-8}

    def analyze(self, histories: Mapping[str, Sequence[float]], markets: Mapping[str, Market], now_ms: int,
                training_histories: Mapping[str, Sequence[float]] | None = None,
                current_weights: Mapping[str, float] | None = None) -> StrategyResult:
        feats = {
            pair: f for pair, market in markets.items()
            if (not self.cfg.allow_pairs or pair.upper() in self.cfg.allow_pairs)
            and (f := self._features(market, histories.get(pair, ()))) is not None
        }
        # Use named fields here: this result is used on a fresh database before any
        # market has enough history. Positional construction silently shifts metrics
        # when StrategyResult gains a new field (previously putting "warming up" in
        # the numeric Brier-score slot and crashing --inspect during formatting).
        empty = StrategyResult(
            long_targets={},
            signals=[],
            all_scores={},
            bearish_veto_pairs=(),
            rejected=[],
            regime="defensive",
            breadth=0.0,
            median_4h=0.0,
            median_12h=0.0,
            median_24h=0.0,
            market_strength=0.0,
            gross_target=0.0,
            median_volatility=0.0,
            volatility_scale=0.0,
            predictor_enabled=False,
            predictor_samples=0,
            predictor_directional_accuracy=0.0,
            predictor_correlation=0.0,
            predictor_rank_ic=0.0,
            predictor_balanced_accuracy=0.0,
            predictor_top_decile_edge=0.0,
            predictor_top_decile_net_return=0.0,
            predictor_brier=1.0,
            predictor_gate_reason="warming up",
            predictor_predictions={},
        )
        if not feats:
            return empty

        median_4h = float(np.median([value["medium"] for value in feats.values()]))
        median_12h = float(np.median([value["long"] for value in feats.values()]))
        median_24h = float(np.median([value["return_24h"] for value in feats.values()]))
        median_vol = float(np.median([value["vol"] for value in feats.values()]))
        breadth = float(np.mean([value["medium"] > 0 and value["long"] > 0 and value["return_24h"] > 0 and value["trend"] > 0 for value in feats.values()]))
        strength = self._market_strength(breadth, median_4h, median_12h, median_24h, median_vol)
        regime = self._regime(strength)
        vol_scale = float(np.clip(self.cfg.target_hourly_vol / max(median_vol, 0.0010), 0.35, 1.25))

        predictor = self._predictor(histories, now_ms, training_histories=training_histories)
        pred_rank = self._rank01({pair: pred.predicted_return for pair, pred in predictor.predictions.items()})
        relative4 = {pair: value["medium"] - median_4h for pair, value in feats.items()}
        relative12 = {pair: value["long"] - median_12h for pair, value in feats.items()}
        ranks = {name: self._rank01({pair: value[key] for pair, value in feats.items()}) for name, key in {
            "fast": "fast", "medium": "medium", "long": "long", "ret24": "return_24h", "trend": "trend",
            "liq": "liquidity", "spread": "spread", "vol": "vol", "pull": "pullback", "break": "breakout",
            "downbreak": "breakdown", "bounce": "bearish_bounce", "downext": "downside_extension"}.items()}
        ranks["relative4"] = self._rank01(relative4)
        ranks["relative12"] = self._rank01(relative12)
        down_ranks = {name: self._rank01({pair: -value[key] for pair, value in feats.items()}) for name, key in {
            "fast": "fast", "medium": "medium", "long": "long", "ret24": "return_24h", "trend": "trend"}.items()}
        short_rel4 = self._rank01({pair: -value for pair, value in relative4.items()})
        short_rel12 = self._rank01({pair: -value for pair, value in relative12.items()})

        long_candidates: list[Signal] = []
        short_candidates: list[Signal] = []
        rejected: list[dict[str, object]] = []
        veto_pairs: list[str] = []
        all_scores: dict[str, float] = {}
        rejection_counts: Counter[str] = Counter()

        for pair, item in feats.items():
            market = markets[pair]
            prediction = predictor.predictions.get(pair, Prediction(0.0, 0.5, 1.0, 0.5, 0.0, 0.0))
            estimated_spread_cost = (market.spread_bps / 10_000.0 if math.isfinite(market.spread_bps) else 0.0)
            estimated_slippage_cost = 2.0 * self.cfg.slippage_bps / 10_000.0
            round_trip_cost = 2.0 * self.cfg.commission_rate + estimated_spread_cost + estimated_slippage_cost
            long_edge_gross = (0.30 * relative4[pair] + 0.24 * relative12[pair] + 0.08 * item["fast"] +
                               0.16 * item["trend"] + 0.10 * item["return_24h"] + 0.12 * item["breakout"])
            short_edge_gross = (-0.30 * relative4[pair] - 0.24 * relative12[pair] - 0.08 * item["fast"] -
                                0.16 * item["trend"] - 0.10 * item["return_24h"] + 0.12 * max(item["breakdown"], 0.0))
            # The edge filter is net of one entry + one exit fee and current quoted spread.
            long_edge = long_edge_gross - round_trip_cost
            short_edge = short_edge_gross - round_trip_cost
            long_prediction_edge = prediction.predicted_return - round_trip_cost
            short_prediction_edge = -prediction.predicted_return - round_trip_cost

            trend_score = (0.08 * ranks["fast"][pair] + 0.30 * ranks["medium"][pair] + 0.20 * ranks["long"][pair] +
                           0.22 * ranks["trend"][pair] + 0.20 * ranks["ret24"][pair])
            relative_score = 0.56 * ranks["relative4"][pair] + 0.44 * ranks["relative12"][pair]
            quality = (0.45 * ranks["liq"][pair] + 0.25 * (1 - ranks["spread"][pair]) +
                       0.20 * (1 - ranks["vol"][pair]) + 0.10 * (1 - ranks["pull"][pair]))
            continuation = 0.60 * ranks["pull"][pair] + 0.40 * ranks["break"][pair]
            long_prediction_component = 0.5
            short_prediction_component = 0.5
            if predictor.enabled and pred_rank:
                confidence = float(np.clip(prediction.confidence, 0.0, 1.0))
                long_prediction_component = float(np.clip(0.5 + 2.0 * (pred_rank.get(pair, 0.5) - 0.5) * confidence, 0.0, 1.0))
                short_prediction_component = float(np.clip(0.5 + 2.0 * ((1.0 - pred_rank.get(pair, 0.5)) - 0.5) * confidence, 0.0, 1.0))
            if predictor.directional_enabled:
                long_prediction_component = 0.70 * long_prediction_component + 0.30 * prediction.probability_up
                short_prediction_component = 0.70 * short_prediction_component + 0.30 * prediction.probability_down

            if regime == "risk_on":
                long_score = (0.28 * trend_score + 0.25 * relative_score + 0.10 * continuation +
                              0.16 * long_prediction_component + 0.11 * quality + 0.10 * ranks["break"][pair])
            elif regime == "balanced":
                long_score = (0.25 * trend_score + 0.26 * relative_score + 0.12 * continuation +
                              0.19 * long_prediction_component + 0.18 * quality)
            elif regime == "risk_off":
                long_score = (0.18 * trend_score + 0.30 * relative_score + 0.10 * continuation +
                              0.12 * long_prediction_component + 0.30 * quality)
            else:
                long_score = (0.16 * trend_score + 0.30 * relative_score + 0.05 * continuation +
                              0.05 * long_prediction_component + 0.44 * quality)
            long_score -= self.cfg.overextension_penalty * max(0.0, item["extension"] / 0.008)
            long_score = float(np.clip(long_score, 0.0, 1.0))
            all_scores[pair] = long_score

            bearish_veto = (self.cfg.bearish_veto_enabled and predictor.directional_enabled and pair in predictor.predictions and
                            prediction.predicted_return <= self.cfg.bearish_veto_return and
                            prediction.probability_down >= self.cfg.bearish_veto_probability and
                            prediction.confidence >= self.cfg.bearish_veto_confidence)
            if bearish_veto:
                veto_pairs.append(pair)

            long_reasons: list[str] = []
            if item["fast"] < self.cfg.min_1h_return: long_reasons.append("1h_floor")
            if item["medium"] < self.cfg.min_4h_return: long_reasons.append("4h_floor")
            if item["long"] < self.cfg.min_12h_return: long_reasons.append("12h_floor")
            if item["trend"] <= 0: long_reasons.append("trend")
            if long_edge < self.cfg.min_relative_edge: long_reasons.append("edge")
            if long_score < self.cfg.min_score: long_reasons.append("score")
            if predictor.enabled and prediction.confidence >= 0.45 and long_prediction_edge < 0.0: long_reasons.append("prediction_edge")
            if bearish_veto: long_reasons.append("ml_bearish_veto")
            if market.quote_volume_24h < self.cfg.min_quote_volume: long_reasons.append("liquidity")
            if not math.isfinite(market.spread_bps) or market.spread_bps > self.cfg.max_spread_bps: long_reasons.append("spread")
            if long_reasons:
                for reason in long_reasons: rejection_counts[reason] += 1
                rejected.append({"pair": pair, "direction": "LONG", "score": long_score, "expected_edge": long_edge,
                                 "predicted_return": prediction.predicted_return, "reasons": "+".join(long_reasons)})
            else:
                reason = "trend+relative+quality"
                if item["breakout"] > 0: reason += "+breakout"
                if predictor.enabled and long_prediction_edge >= 0: reason += "+ml_rank"
                long_candidates.append(Signal(pair, long_score, item["fast"], item["medium"], item["long"], item["return_24h"],
                                              item["trend"], item["vol"], relative4[pair], relative12[pair], item["pullback"],
                                              item["breakout"], item["extension"], prediction.predicted_return,
                                              prediction.probability_up, prediction.probability_down, prediction.confidence,
                                              long_edge, 0.0, 0.0, bearish_veto, reason, "LONG",
                                              gross_expected_edge=(max(0.0, prediction.predicted_return)
                                                                   if predictor.enabled else long_edge_gross),
                                              round_trip_cost=round_trip_cost,
                                              one_way_cost=self.cfg.commission_rate +
                                                  (market.spread_bps / 20_000.0 if math.isfinite(market.spread_bps) else 0.0) +
                                                  self.cfg.slippage_bps / 10_000.0))

            # Short-side candidates use independent downside ranks and stricter regime gates.
            short_trend_score = (0.08 * down_ranks["fast"][pair] + 0.30 * down_ranks["medium"][pair] +
                                 0.20 * down_ranks["long"][pair] + 0.22 * down_ranks["trend"][pair] +
                                 0.20 * down_ranks["ret24"][pair])
            short_relative_score = 0.56 * short_rel4[pair] + 0.44 * short_rel12[pair]
            short_continuation = 0.65 * ranks["downbreak"][pair] + 0.35 * down_ranks["fast"][pair]
            if regime == "risk_on":
                short_score = (0.27 * short_trend_score + 0.24 * short_relative_score + 0.09 * short_continuation +
                               0.12 * short_prediction_component + 0.22 * quality + 0.06 * ranks["downbreak"][pair])
            elif regime == "balanced":
                short_score = (0.24 * short_trend_score + 0.28 * short_relative_score + 0.10 * short_continuation +
                               0.14 * short_prediction_component + 0.19 * quality + 0.05 * ranks["downbreak"][pair])
            elif regime == "risk_off":
                short_score = (0.22 * short_trend_score + 0.30 * short_relative_score + 0.12 * short_continuation +
                               0.12 * short_prediction_component + 0.20 * quality + 0.04 * ranks["downbreak"][pair])
            else:
                short_score = (0.18 * short_trend_score + 0.32 * short_relative_score + 0.15 * short_continuation +
                               0.10 * short_prediction_component + 0.20 * quality + 0.03 * ranks["downbreak"][pair] +
                               0.02 * ranks["bounce"][pair])
            short_score -= self.cfg.overextension_penalty * max(0.0, item["downside_extension"] / 0.008)
            short_score = float(np.clip(short_score, 0.0, 1.0))
            short_reasons: list[str] = []
            if not self.cfg.shorts_enabled or not self.shorts_available: short_reasons.append("shorts_unavailable")
            if regime == "risk_on" and not self.cfg.allow_shorts_risk_on: short_reasons.append("risk_on_short_gate")
            if item["fast"] > self.cfg.max_short_1h_return: short_reasons.append("1h_ceiling")
            if item["medium"] > self.cfg.max_short_4h_return: short_reasons.append("4h_ceiling")
            if item["long"] > self.cfg.max_short_12h_return: short_reasons.append("12h_ceiling")
            if item["trend"] >= 0 and item["breakdown"] < 0.003: short_reasons.append("trend_or_breakdown")
            if short_edge < self.cfg.min_short_relative_edge: short_reasons.append("short_edge")
            if short_score < self.cfg.min_short_score: short_reasons.append("short_score")
            if predictor.enabled and prediction.confidence >= 0.45 and short_prediction_edge < 0.0:
                short_reasons.append("prediction_edge")
            if predictor.directional_enabled and prediction.confidence >= 0.55 and prediction.probability_up >= 0.67 and prediction.predicted_return > 0.004:
                short_reasons.append("ml_bullish_veto")
            if market.quote_volume_24h < self.cfg.min_quote_volume: short_reasons.append("liquidity")
            if not math.isfinite(market.spread_bps) or market.spread_bps > self.cfg.max_spread_bps: short_reasons.append("spread")
            if short_reasons:
                if "shorts_unavailable" not in short_reasons and "risk_on_short_gate" not in short_reasons:
                    for reason in short_reasons: rejection_counts["short_" + reason] += 1
            else:
                reason = "downtrend+relative+quality"
                if item["breakdown"] > 0: reason += "+breakdown"
                if predictor.enabled and short_prediction_edge >= 0: reason += "+ml_rank"
                short_candidates.append(Signal(pair, short_score, item["fast"], item["medium"], item["long"], item["return_24h"],
                                               item["trend"], item["vol"], relative4[pair], relative12[pair], item["bearish_bounce"],
                                               item["breakdown"], item["downside_extension"], prediction.predicted_return,
                                               prediction.probability_up, prediction.probability_down, prediction.confidence,
                                               short_edge, 0.0, 0.0, False, reason, "SHORT",
                                               gross_expected_edge=(max(0.0, -prediction.predicted_return)
                                                                    if predictor.enabled else short_edge_gross),
                                               round_trip_cost=round_trip_cost,
                                               one_way_cost=self.cfg.commission_rate +
                                                   (market.spread_bps / 20_000.0 if math.isfinite(market.spread_bps) else 0.0) +
                                                   self.cfg.slippage_bps / 10_000.0))

        long_candidates.sort(key=lambda signal: (signal.score, signal.expected_edge), reverse=True)
        selected_longs = self._diversify(long_candidates, histories, self.cfg.top_n)
        regime_cap = {"risk_on": self.cfg.risk_on_gross_cap, "balanced": self.cfg.balanced_gross_cap,
                      "risk_off": self.cfg.risk_off_gross_cap, "defensive": self.cfg.defensive_gross_cap}[regime]
        # Smooth exposure down in weak regimes instead of abruptly forcing the long book to zero.
        # Entry filters still require positive trend and positive cost-adjusted edge.
        strength_for_size = 0.35 + 0.65 * strength
        long_gross = min(self.cfg.max_gross_target * (strength_for_size ** self.cfg.gross_strength_power) * vol_scale,
                         self.cfg.max_gross_target * regime_cap, self.cfg.max_total_gross_target, 1.0 - self.cfg.cash_reserve)
        # Jointly optimize long and short targets instead of allocating each side
        # independently by score/volatility. Candidate direction is fixed, while
        # expected gross edge, covariance risk and marginal trading cost determine size.
        long_budget = long_gross
        long_pairs = {signal.pair for signal in selected_longs}
        short_candidates = [signal for signal in short_candidates if signal.pair not in long_pairs]
        short_candidates.sort(key=lambda signal: (signal.score, signal.expected_edge), reverse=True)
        selected_shorts = self._diversify(short_candidates, histories, self.cfg.short_top_n)
        short_strength_for_size = 0.25 + 0.75 * (1.0 - strength)
        short_budget = min(
            self.cfg.max_short_gross_target * (short_strength_for_size ** self.cfg.short_strength_power) * vol_scale,
            self.cfg.max_short_gross_target,
            self.cfg.max_total_gross_target,
            max(0.0, 1.0 - self.cfg.cash_reserve),
        )
        optimizer_longs: list[object] = list(selected_longs)
        optimizer_shorts: list[object] = list(selected_shorts)
        candidate_pairs = {signal.pair for signal in selected_longs} | {signal.pair for signal in selected_shorts}
        # Held assets that temporarily fall outside the entry list remain in the
        # objective with zero new alpha. This prices the cost of closing them and
        # avoids churn-driven sell/rebuy loops; hard exits are still enforced later
        # by the dedicated per-position exit logic.
        for pair, signed_weight in (current_weights or {}).items():
            pair = str(pair)
            if pair in candidate_pairs or abs(float(signed_weight)) <= 1e-8:
                continue
            market = markets.get(pair)
            spread_cost = market.spread_bps / 20_000.0 if market and math.isfinite(market.spread_bps) else 0.0
            one_way = self.cfg.commission_rate + spread_cost + self.cfg.slippage_bps / 10_000.0
            held = SimpleNamespace(pair=pair, gross_expected_edge=0.0, expected_edge=0.0,
                                   round_trip_cost=2.0 * one_way, one_way_cost=one_way,
                                   volatility=0.01)
            (optimizer_longs if float(signed_weight) > 0 else optimizer_shorts).append(held)

        optimizer_result = optimize_signed_portfolio(
            optimizer_longs, optimizer_shorts, histories, current_weights,
            long_budget=long_budget, short_budget=short_budget,
            total_gross_cap=min(self.cfg.max_total_gross_target, 1.0 - self.cfg.cash_reserve),
            long_asset_cap=self.cfg.max_asset_weight,
            short_asset_cap=self.cfg.max_short_asset_weight,
            covariance_window_bars=self._bars(self.cfg.portfolio_covariance_window_minutes),
            risk_horizon_bars=self._bars(self.cfg.prediction_horizon if predictor.enabled
                                         else self.cfg.portfolio_risk_horizon_minutes),
            covariance_shrinkage=self.cfg.portfolio_covariance_shrinkage,
            volatility_window_bars=self._bars(self.cfg.vol_window),
            covariance_half_life_bars=self._bars(self.cfg.portfolio_covariance_half_life_minutes),
            risk_aversion=self.cfg.portfolio_risk_aversion,
            cost_penalty=self.cfg.portfolio_cost_penalty,
            iterations=self.cfg.portfolio_optimizer_iterations,
        )
        long_targets = {pair: weight for pair, weight in optimizer_result.weights.items() if weight > 0.0}
        short_targets = {pair: -weight for pair, weight in optimizer_result.weights.items() if weight < 0.0}
        enriched_longs = [Signal(**{**signal.__dict__, "target_weight": long_targets.get(signal.pair, 0.0)})
                          for signal in selected_longs if long_targets.get(signal.pair, 0.0) > 0.0]
        enriched_shorts = [Signal(**{**signal.__dict__, "target_weight": short_targets.get(signal.pair, 0.0)})
                           for signal in selected_shorts if short_targets.get(signal.pair, 0.0) > 0.0]

        rejected.sort(key=lambda item: (float(item["score"]), str(item["pair"])), reverse=True)
        diagnostics = {
            "feature_count": len(feats),
            "tradable_market_count": len(markets),
            "funnel": {"features": len(feats), "long_candidates": len(long_candidates), "long_selected": len(enriched_longs),
                       "short_candidates": len(short_candidates), "short_selected": len(enriched_shorts)},
            "rejection_counts": dict(rejection_counts),
            "top_long_candidates": [(signal.pair, round(signal.score, 4)) for signal in long_candidates[:self.cfg.diagnostic_top_n]],
            "top_short_candidates": [(signal.pair, round(signal.score, 4)) for signal in short_candidates[:self.cfg.diagnostic_top_n]],
            "next_fit_ms": self._last_predictor_fit_ms + self.cfg.predictor_retrain_seconds * 1000 if self.predictor else 0,
            "poll_seconds": self.cfg.poll_seconds,
            "feature_windows_bars": {"1h": self._bars(60), "4h": self._bars(240), "12h": self._bars(720)},
            "portfolio_optimizer": {
                "method": optimizer_result.method,
                "objective": optimizer_result.objective,
                "expected_edge_contribution": optimizer_result.expected_edge_contribution,
                "horizon_variance": optimizer_result.variance,
                "estimated_rebalance_cost": optimizer_result.transaction_cost_estimate,
                "covariance_condition": optimizer_result.covariance_condition,
                "iterations": optimizer_result.iterations,
                "converged": optimizer_result.converged,
                "risk_aversion": self.cfg.portfolio_risk_aversion,
                "cost_penalty": self.cfg.portfolio_cost_penalty,
                "risk_horizon_minutes": (self.cfg.prediction_horizon if predictor.enabled
                                          else self.cfg.portfolio_risk_horizon_minutes),
                "alpha_source": "validated_predictor" if predictor.enabled else "momentum_edge_proxy",
            },
        }
        return StrategyResult(
            long_targets=long_targets,
            signals=enriched_longs,
            all_scores=all_scores,
            bearish_veto_pairs=tuple(sorted(set(veto_pairs))),
            rejected=rejected,
            regime=regime,
            breadth=breadth,
            median_4h=median_4h,
            median_12h=median_12h,
            median_24h=median_24h,
            market_strength=strength,
            gross_target=sum(long_targets.values()),
            median_volatility=median_vol,
            volatility_scale=vol_scale,
            predictor_enabled=predictor.enabled,
            predictor_samples=predictor.samples,
            predictor_directional_accuracy=predictor.validation_directional_accuracy,
            predictor_correlation=predictor.validation_correlation,
            predictor_rank_ic=predictor.validation_rank_ic,
            predictor_balanced_accuracy=predictor.validation_balanced_accuracy,
            predictor_top_decile_edge=predictor.validation_top_decile_edge,
            predictor_top_decile_net_return=predictor.validation_top_decile_net_return,
            predictor_brier=predictor.validation_brier,
            predictor_gate_reason=predictor.gate_reason,
            predictor_predictions=predictor.predictions,
            diagnostics=diagnostics,
            short_targets=short_targets,
            short_signals=enriched_shorts,
            predictor_directional_enabled=predictor.directional_enabled,
            short_gross_target=sum(short_targets.values()),
        )
