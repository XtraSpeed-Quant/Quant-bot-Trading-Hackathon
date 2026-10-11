from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Prediction:
    predicted_return: float
    probability_up: float
    probability_flat: float
    probability_down: float
    confidence: float
    z_score: float


@dataclass(frozen=True)
class PredictorResult:
    enabled: bool
    samples: int
    validation_directional_accuracy: float
    validation_correlation: float
    validation_balanced_accuracy: float
    validation_top_decile_edge: float
    validation_brier: float
    residual_std: float
    predictions: dict[str, Prediction]
    gate_reason: str = ""
    fit_seconds: float = 0.0
    prediction_seconds: float = 0.0
    last_fit_ms: int = 0
    validation_top_decile_mean_return: float = 0.0
    validation_top_decile_net_return: float = 0.0
    validation_rank_ic: float = 0.0
    directional_enabled: bool = False


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)


def _returns(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) > window:
        out[window:] = values[window:] / values[:-window] - 1.0
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return out
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    out[window - 1 :] = (prefix[window:] - prefix[:-window]) / window
    return out


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < window:
        return out
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    prefix2 = np.concatenate(([0.0], np.cumsum(values * values, dtype=float)))
    mean = (prefix[window:] - prefix[:-window]) / window
    mean2 = (prefix2[window:] - prefix2[:-window]) / window
    out[window - 1 :] = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    return out


def _rolling_max_previous(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) <= window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values[:-1], window)
    maxima = np.max(windows, axis=1)
    out[window:] = maxima[: len(out) - window]
    return out


def _ema_array(values: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    if len(values) == 0:
        return out
    alpha = 2.0 / (window + 1.0)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _feature_arrays(values: Sequence[float], bar_seconds: int = 60) -> dict[str, np.ndarray]:
    """Build features on real-time horizons, independently of polling cadence."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        return {}

    def bars(minutes: float) -> int:
        return max(1, int(round(float(minutes) * 60.0 / max(int(bar_seconds), 1))))

    b5, b15, b30 = bars(5), bars(15), bars(30)
    b60, b240, b360 = bars(60), bars(240), bars(360)
    log_returns = np.diff(np.log(arr))
    vol15_returns = _rolling_std(log_returns, b15)
    vol60_returns = _rolling_std(log_returns, b60)
    vol15 = np.full(len(arr), np.nan, dtype=float)
    vol60 = np.full(len(arr), np.nan, dtype=float)
    # Convert per-bar standard deviation into the volatility over the corresponding time window.
    vol15[b15:] = vol15_returns[b15 - 1:] * math.sqrt(b15)
    vol60[b60:] = vol60_returns[b60 - 1:] * math.sqrt(b60)

    result = {
        "r5": _returns(arr, b5),
        "r15": _returns(arr, b15),
        "r30": _returns(arr, b30),
        "r60": _returns(arr, b60),
        "r240": _returns(arr, b240),
        "r360": _returns(arr, b360),
        "ema30": _ema_array(arr, bars(30)),
        "ema120": _ema_array(arr, bars(120)),
        "ema360": _ema_array(arr, b360),
        "vol15": vol15,
        "vol60": vol60,
        "high240_prev": _rolling_max_previous(arr, b240),
        "mean60": _rolling_mean(arr, b60),
    }
    result["trend30"] = arr / np.maximum(result["ema30"], 1e-12) - 1.0
    result["trend120"] = arr / np.maximum(result["ema120"], 1e-12) - 1.0
    result["trend360"] = arr / np.maximum(result["ema360"], 1e-12) - 1.0
    result["extension"] = np.maximum(0.0, result["r60"] - 0.25 * np.maximum(result["r240"], 0.0))
    result["accel"] = result["r15"] - result["r60"] / max(b60 / b15, 1.0)
    result["decel"] = result["r60"] - result["r240"] / max(b240 / b60, 1.0)
    result["vol_ratio"] = result["vol15"] / np.maximum(result["vol60"], 1e-8)
    result["mean_gap"] = arr / np.maximum(result["mean60"], 1e-12) - 1.0
    return result


class PredictiveEnsemble:
    """Walk-forward-gated ridge regression plus three-state classification."""

    def __init__(
        self,
        feature_window: int,
        prediction_horizon: int,
        min_samples: int,
        max_training_samples: int,
        ridge_alpha: float,
        classifier_alpha: float,
        validation_fraction: float,
        purge_gap: int,
        min_directional_accuracy: float,
        min_correlation: float,
        min_balanced_accuracy: float,
        min_top_decile_edge: float,
        class_threshold: float,
        fee_assumption: float,
        max_brier: float = 0.23,
        min_top_decile_net_return: float = 0.00025,
        poll_seconds: int = 60,
        min_rank_ic: float = 0.01,
        slippage_bps: float = 10.0,
    ) -> None:
        self.feature_window = feature_window
        self.prediction_horizon = prediction_horizon
        self.min_samples = min_samples
        self.max_training_samples = max_training_samples
        self.ridge_alpha = ridge_alpha
        self.classifier_alpha = classifier_alpha
        self.validation_fraction = validation_fraction
        self.purge_gap = purge_gap
        self.min_directional_accuracy = min_directional_accuracy
        self.min_correlation = min_correlation
        self.min_rank_ic = min_rank_ic
        self.min_balanced_accuracy = min_balanced_accuracy
        self.min_top_decile_edge = min_top_decile_edge
        self.class_threshold = class_threshold
        self.fee_assumption = fee_assumption
        self.slippage_assumption = 2.0 * max(0.0, float(slippage_bps)) / 10_000.0
        self.max_brier = max_brier
        self.min_top_decile_net_return = min_top_decile_net_return
        self.poll_seconds = max(1, int(poll_seconds))
        self.feature_window_bars = max(1, int(round(self.feature_window * 60 / self.poll_seconds)))
        self.prediction_horizon_bars = max(1, int(round(self.prediction_horizon * 60 / self.poll_seconds)))
        self.purge_gap_bars = max(0, int(round(self.purge_gap * 60 / self.poll_seconds)))
        self._model: dict[str, np.ndarray | float] | None = None
        self._last_validation: PredictorResult | None = None
        self._classifier_enabled = False

    @property
    def minimum_history(self) -> int:
        return self.feature_window_bars + self.prediction_horizon_bars + 1

    @property
    def last_validation(self) -> PredictorResult | None:
        return self._last_validation

    def _build_samples(self, histories: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        valid_pairs = [
            (pair, np.asarray(values, dtype=float))
            for pair, values in histories.items()
            if len(values) >= self.minimum_history
        ]
        if not valid_pairs:
            return np.empty((0, 18)), np.empty(0), np.empty(0), np.empty(0, dtype=int)

        common_len = min(len(values) for _, values in valid_pairs)
        bar60 = max(1, int(round(60 * 60 / self.poll_seconds)))
        bar240 = max(1, int(round(240 * 60 / self.poll_seconds)))
        matrices_r60: list[np.ndarray] = []
        matrices_r240: list[np.ndarray] = []
        arrays: list[np.ndarray] = []
        for _, values in valid_pairs:
            arr = values[-common_len:]
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                continue
            arrays.append(arr)
            matrices_r60.append(_returns(arr, bar60))
            matrices_r240.append(_returns(arr, bar240))

        if not arrays:
            return np.empty((0, 18)), np.empty(0), np.empty(0), np.empty(0, dtype=int)

        market_r60 = np.zeros(common_len, dtype=float)
        market_r240 = np.zeros(common_len, dtype=float)
        stacked60 = np.vstack(matrices_r60)
        stacked240 = np.vstack(matrices_r240)
        if common_len > bar60:
            market_r60[bar60:] = np.nanmedian(stacked60[:, bar60:], axis=0)
        if common_len > bar240:
            market_r240[bar240:] = np.nanmedian(stacked240[:, bar240:], axis=0)
        rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        times: list[np.ndarray] = []
        classes: list[np.ndarray] = []
        end = common_len - self.prediction_horizon_bars
        full_indices = np.arange(self.feature_window_bars, end, dtype=int)
        # Choose a uniform set of timestamps BEFORE creating feature matrices. This
        # keeps every selected cross-section but bounds peak memory for a week of
        # many markets; post-hoc truncation would first materialize millions of rows.
        cross_section_size = max(1, len(arrays))
        max_timepoints = max(1, self.max_training_samples // cross_section_size)
        if len(full_indices) > max_timepoints:
            sampled_positions = np.linspace(0, len(full_indices) - 1, max_timepoints, dtype=int)
            idx = full_indices[np.unique(sampled_positions)]
        else:
            idx = full_indices

        for arr in arrays:
            features = _feature_arrays(arr, self.poll_seconds)
            if not features or len(idx) == 0:
                continue
            r60 = features["r60"][idx]
            r240 = features["r240"][idx]
            vol15 = features["vol15"][idx]
            vol60 = features["vol60"][idx]
            matrix = np.column_stack([
                features["r5"][idx], features["r15"][idx], features["r30"][idx],
                r60, r240, features["r360"][idx],
                features["trend30"][idx], features["trend120"][idx], features["trend360"][idx],
                features["accel"][idx], features["decel"][idx],
                vol15, vol60, features["vol_ratio"][idx],
                features["extension"][idx], features["mean_gap"][idx],
                r60 - market_r60[idx], r240 - market_r240[idx],
            ])
            current = arr[idx]
            future = arr[idx + self.prediction_horizon_bars]
            target = future / np.maximum(current, 1e-12) - 1.0
            threshold = np.maximum(
                self.class_threshold,
                2.0 * self.fee_assumption + 0.5 * np.nan_to_num(vol60, nan=0.0),
            )
            cls = np.where(target > threshold, 1, np.where(target < -threshold, -1, 0)).astype(int)
            finite = np.isfinite(matrix).all(axis=1) & np.isfinite(target)
            if np.any(finite):
                rows.append(matrix[finite])
                targets.append(target[finite])
                times.append(idx[finite].astype(float))
                classes.append(cls[finite])

        if not rows:
            return np.empty((0, 18)), np.empty(0), np.empty(0), np.empty(0, dtype=int)
        x = np.vstack(rows)
        y = np.concatenate(targets)
        t = np.concatenate(times)
        c = np.concatenate(classes)
        return x, y, t, c

    @staticmethod
    def _balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        recalls = []
        for cls in (-1, 0, 1):
            mask = actual == cls
            if np.any(mask):
                recalls.append(float(np.mean(predicted[mask] == cls)))
        return float(np.mean(recalls)) if recalls else 0.0

    @staticmethod
    def _cross_sectional_validation_metrics(predicted: np.ndarray, actual: np.ndarray,
                                           times: np.ndarray) -> tuple[float, float, float]:
        """Return mean per-timestamp rank IC, top-decile mean return, and top-decile edge."""
        rank_ics: list[float] = []
        top_means: list[float] = []
        universe_means: list[float] = []
        for timestamp in np.unique(times):
            mask = times == timestamp
            p, y = predicted[mask], actual[mask]
            if len(p) < 10:
                continue
            p_ranks = np.argsort(np.argsort(p)).astype(float)
            y_ranks = np.argsort(np.argsort(y)).astype(float)
            rank_ics.append(_safe_corr(p_ranks, y_ranks))
            top_n = max(1, int(math.ceil(len(p) * 0.10)))
            top_idx = np.argsort(p)[-top_n:]
            top_means.append(float(np.mean(y[top_idx])))
            universe_means.append(float(np.mean(y)))
        if not rank_ics:
            return 0.0, float(np.mean(actual)) if len(actual) else 0.0, 0.0
        top_mean = float(np.mean(top_means))
        edge = float(np.mean(np.asarray(top_means) - np.asarray(universe_means)))
        return float(np.mean(rank_ics)), top_mean, edge

    def _fit_model(self, x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict[str, np.ndarray | float]:
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        sigma = np.where(sigma < 1e-8, 1.0, sigma)
        xs = (x - mu) / sigma
        y_mu = float(np.mean(y))
        y_scale = max(float(np.std(y)), 1e-8)
        ys = (y - y_mu) / y_scale
        beta_reg = np.linalg.solve(
            xs.T @ xs + self.ridge_alpha * np.eye(xs.shape[1]),
            xs.T @ ys,
        )
        one_hot = np.column_stack([
            (classes == -1).astype(float),
            (classes == 0).astype(float),
            (classes == 1).astype(float),
        ])
        coef_class = np.linalg.solve(
            xs.T @ xs + self.classifier_alpha * np.eye(xs.shape[1]),
            xs.T @ one_hot,
        )
        return {
            "mu": mu,
            "sigma": sigma,
            "y_mu": y_mu,
            "y_scale": y_scale,
            "beta_reg": beta_reg,
            "coef_class": coef_class,
        }

    @staticmethod
    def _predict_matrix(model: Mapping[str, np.ndarray | float], x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        mu = np.asarray(model["mu"], dtype=float)
        sigma = np.asarray(model["sigma"], dtype=float)
        xs = (x - mu) / sigma
        y_mu = float(model["y_mu"])
        y_scale = float(model["y_scale"])
        beta_reg = np.asarray(model["beta_reg"], dtype=float)
        coef_class = np.asarray(model["coef_class"], dtype=float)
        pred = xs @ beta_reg * y_scale + y_mu
        probs = _softmax(xs @ coef_class)
        return pred, probs, y_scale

    def _current_predictions(self, histories: Mapping[str, Sequence[float]]) -> dict[str, Prediction]:
        if self._model is None:
            return {}
        arrays = {
            pair: _feature_arrays(values, self.poll_seconds)
            for pair, values in histories.items()
            if len(values) >= self.feature_window_bars + 1
        }
        usable = [arr for arr in arrays.values() if arr]
        if not usable:
            return {}
        current_r60 = np.asarray([float(arr["r60"][-1]) for arr in usable if np.isfinite(arr["r60"][-1])], dtype=float)
        current_r240 = np.asarray([float(arr["r240"][-1]) for arr in usable if np.isfinite(arr["r240"][-1])], dtype=float)
        market_r60 = float(np.median(current_r60)) if current_r60.size else 0.0
        market_r240 = float(np.median(current_r240)) if current_r240.size else 0.0
        predictions: dict[str, Prediction] = {}
        for pair, f in arrays.items():
            if not f:
                continue
            arr = np.asarray(histories[pair], dtype=float)
            i = len(arr) - 1
            values = [
                f["r5"][i], f["r15"][i], f["r30"][i], f["r60"][i], f["r240"][i], f["r360"][i],
                f["trend30"][i], f["trend120"][i], f["trend360"][i], f["accel"][i], f["decel"][i],
                f["vol15"][i], f["vol60"][i], f["vol_ratio"][i], f["extension"][i], f["mean_gap"][i],
                f["r60"][i] - market_r60, f["r240"][i] - market_r240,
            ]
            x = np.asarray(values, dtype=float)
            if not np.isfinite(x).all():
                continue
            pred, probs, y_scale = self._predict_matrix(self._model, x.reshape(1, -1))
            regression_pred = float(pred[0])
            p_down, p_flat, p_up = [float(v) for v in probs[0]]
            ensemble = 0.70 * regression_pred + 0.30 * (p_up - p_down) * y_scale
            residual_std = max(float(self._model["residual_std"]), 1e-8)
            z = ensemble / residual_std
            z_component = 1.0 - math.exp(-abs(z) / 2.5)
            prob_component = abs(p_up - p_down) if self._classifier_enabled else 0.0
            confidence = min(0.98, (0.35 * z_component + 0.35 * prob_component + 0.30 * 0.5) if self._classifier_enabled else (0.65 * z_component + 0.35 * 0.5))
            predictions[pair] = Prediction(ensemble, p_up, p_flat, p_down, confidence, z)
        return predictions

    def fit_and_predict(
        self,
        histories: Mapping[str, Sequence[float]],
        refit: bool = True,
        fit_timestamp_ms: int = 0,
        training_histories: Mapping[str, Sequence[float]] | None = None,
    ) -> PredictorResult:
        prediction_started = time.perf_counter()
        if not refit and self._last_validation is not None:
            # If a previous scheduled fit failed the minimum-sample gate, retain that
            # result until the next scheduled full-history retrain instead of fitting
            # the smaller signal window on every market poll.
            previous = self._last_validation
            predictions = self._current_predictions(histories) if self._model is not None else {}
            prediction_seconds = time.perf_counter() - prediction_started
            return PredictorResult(
                enabled=previous.enabled,
                samples=previous.samples,
                validation_directional_accuracy=previous.validation_directional_accuracy,
                validation_correlation=previous.validation_correlation,
                validation_balanced_accuracy=previous.validation_balanced_accuracy,
                validation_top_decile_edge=previous.validation_top_decile_edge,
                validation_top_decile_mean_return=previous.validation_top_decile_mean_return,
                validation_top_decile_net_return=previous.validation_top_decile_net_return,
                validation_brier=previous.validation_brier,
                residual_std=previous.residual_std,
                predictions=predictions,
                gate_reason=previous.gate_reason,
                fit_seconds=previous.fit_seconds,
                prediction_seconds=prediction_seconds,
                last_fit_ms=previous.last_fit_ms,
                validation_rank_ic=previous.validation_rank_ic,
                directional_enabled=previous.directional_enabled,
            )

        fit_started = time.perf_counter()
        fit_histories = training_histories if training_histories is not None else histories
        x, y, times, classes = self._build_samples(fit_histories)
        samples = len(y)
        if samples < self.min_samples:
            self._model = None
            self._classifier_enabled = False
            elapsed = time.perf_counter() - fit_started
            result = PredictorResult(
                enabled=False, samples=samples, validation_directional_accuracy=0.0,
                validation_correlation=0.0, validation_balanced_accuracy=0.0,
                validation_top_decile_edge=0.0, validation_brier=1.0, residual_std=0.0,
                predictions={}, gate_reason=f"insufficient samples: {samples}/{self.min_samples}",
                fit_seconds=elapsed, prediction_seconds=0.0, last_fit_ms=fit_timestamp_ms,
            )
            self._last_validation = result
            return result

        earliest = float(np.min(times))
        latest = float(np.max(times))
        span = max(latest - earliest, 1.0)
        split = earliest + span * (1.0 - self.validation_fraction)
        train_mask = times < split - self.purge_gap_bars
        valid_mask = times >= split
        train_count = int(np.count_nonzero(train_mask))
        valid_count = int(np.count_nonzero(valid_mask))
        if train_count < 200 or valid_count < 100:
            self._model = None
            self._classifier_enabled = False
            elapsed = time.perf_counter() - fit_started
            reason = f"time split too small: train={train_count}, validation={valid_count}; collect more history"
            result = PredictorResult(
                enabled=False, samples=samples, validation_directional_accuracy=0.0,
                validation_correlation=0.0, validation_balanced_accuracy=0.0,
                validation_top_decile_edge=0.0, validation_brier=1.0, residual_std=0.0,
                predictions={}, gate_reason=reason, fit_seconds=elapsed,
                prediction_seconds=0.0, last_fit_ms=fit_timestamp_ms,
            )
            self._last_validation = result
            return result

        validation_model = self._fit_model(x[train_mask], y[train_mask], classes[train_mask])
        pred_valid, probs, _ = self._predict_matrix(validation_model, x[valid_mask])
        c_valid = classes[valid_mask]
        y_valid = y[valid_mask]
        mapping = np.array([-1, 0, 1])
        pred_class = mapping[np.argmax(probs, axis=1)]
        accuracy = float(
            np.mean(
                np.where(pred_valid >= 0, 1, -1)
                == np.where(y_valid >= 0, 1, -1)
            )
        )
        correlation = _safe_corr(pred_valid, y_valid)
        balanced_accuracy = self._balanced_accuracy(c_valid, pred_class)
        brier = float(
            np.mean(
                (probs[:, 0] - (c_valid == -1)) ** 2
                + (probs[:, 1] - (c_valid == 0)) ** 2
                + (probs[:, 2] - (c_valid == 1)) ** 2
            ) / 3.0
        )
        rank_ic, top_mean_return, top_edge = self._cross_sectional_validation_metrics(
            pred_valid, y_valid, times[valid_mask]
        )
        top_net_return = float(top_mean_return - 2.0 * self.fee_assumption - self.slippage_assumption)
        residual_std = float(np.std(y_valid - pred_valid))
        rank_gate_failures = []
        if correlation < self.min_correlation:
            rank_gate_failures.append(f"correlation {correlation:.4f}<{self.min_correlation:.4f}")
        if rank_ic < self.min_rank_ic:
            rank_gate_failures.append(f"rank_ic {rank_ic:.4f}<{self.min_rank_ic:.4f}")
        if top_edge < self.min_top_decile_edge:
            rank_gate_failures.append(f"top_decile_edge {top_edge:.5f}<{self.min_top_decile_edge:.5f}")
        if top_net_return < self.min_top_decile_net_return:
            rank_gate_failures.append(f"top_decile_net {top_net_return:.5f}<{self.min_top_decile_net_return:.5f}")
        if brier > self.max_brier:
            rank_gate_failures.append(f"brier {brier:.3f}>{self.max_brier:.3f}")
        if residual_std <= 1e-8:
            rank_gate_failures.append("residual_std too small")
        rank_enabled = not rank_gate_failures
        classifier_failures = []
        if balanced_accuracy < self.min_balanced_accuracy:
            classifier_failures.append(f"balanced_acc {balanced_accuracy:.3f}<{self.min_balanced_accuracy:.3f}")
        if accuracy < self.min_directional_accuracy:
            classifier_failures.append(f"directional_acc {accuracy:.3f}<{self.min_directional_accuracy:.3f}")
        if brier > self.max_brier:
            classifier_failures.append(f"brier {brier:.3f}>{self.max_brier:.3f}")
        directional_enabled = not classifier_failures

        # Keep a validated regression ranker even when the 3-class classifier fails.
        # Bearish/up probabilities remain gated independently and are not treated as trusted
        # directional evidence unless the classifier clears its own validation checks.
        live_model = self._fit_model(x, y, classes) if (rank_enabled or directional_enabled) else validation_model
        live_model["residual_std"] = residual_std
        self._model = live_model
        self._classifier_enabled = directional_enabled
        fit_seconds = time.perf_counter() - fit_started

        predictions_started = time.perf_counter()
        predictions = self._current_predictions(histories)
        prediction_seconds = time.perf_counter() - predictions_started
        result = PredictorResult(
            enabled=rank_enabled,
            samples=samples,
            validation_directional_accuracy=accuracy,
            validation_correlation=correlation,
            validation_balanced_accuracy=balanced_accuracy,
            validation_top_decile_edge=top_edge,
            validation_top_decile_mean_return=top_mean_return,
            validation_top_decile_net_return=top_net_return,
            validation_brier=brier,
            residual_std=residual_std,
            validation_rank_ic=rank_ic,
            predictions=predictions,
            gate_reason=(
                "rank+directional gates passed" if rank_enabled and directional_enabled
                else "rank gate passed; classifier gated: " + "; ".join(classifier_failures) if rank_enabled
                else "rank gated: " + "; ".join(rank_gate_failures) + ("; classifier passed" if directional_enabled else "; classifier gated: " + "; ".join(classifier_failures))
            ),
            fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
            last_fit_ms=fit_timestamp_ms,
            directional_enabled=directional_enabled,
        )
        self._last_validation = result
        return result
