from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


ATTN_CANDIDATES = (
    ("attn_sum_sq_m",),
)


@dataclass(frozen=True)
class FitRow:
    iteration: int
    microbatch_id: int
    forward_ms: float
    attn_ms: float | None
    mlp_ms: float | None
    seq_lengths: List[float]
    total_tokens: float
    sum_sq: float


def _load_profile(profile_path: str | Path) -> Mapping[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _iteration_records(profile: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    iterations = profile.get("iterations", [])
    if not isinstance(iterations, list):
        return []
    return [iteration for iteration in iterations if isinstance(iteration, Mapping)]


def _time_series_from_profile(profile: Mapping[str, Any], key: str) -> List[List[float]]:
    values = profile.get(key)
    if isinstance(values, list):
        return [
            [float(value) for value in iteration_values]
            for iteration_values in values
            if isinstance(iteration_values, list)
        ]

    series: List[List[float]] = []
    for iteration in _iteration_records(profile):
        iteration_values = iteration.get(key)
        if isinstance(iteration_values, list):
            series.append([float(value) for value in iteration_values])
    return series


def _seq_info_series_from_profile(profile: Mapping[str, Any]) -> List[List[Any]]:
    values = profile.get("seq_info")
    if isinstance(values, list):
        return [
            list(iteration_values)
            for iteration_values in values
            if isinstance(iteration_values, list)
        ]

    series: List[List[Any]] = []
    for iteration in _iteration_records(profile):
        iteration_values = iteration.get("seq_info")
        if isinstance(iteration_values, list):
            series.append(list(iteration_values))
    return series


def _series_at(series: Sequence[Sequence[Any]], iteration: int) -> List[Any]:
    if iteration >= len(series):
        return []
    return list(series[iteration])


def _flatten_seq_lengths(value: Any) -> List[float]:
    lengths: List[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            lengths.append(float(item))

    visit(value)
    if not lengths:
        raise ValueError("seq_lengths is empty.")
    return lengths


def _rows_from_profile(profile: Mapping[str, Any], drop_warmup: int) -> List[FitRow]:
    fwd_times = _time_series_from_profile(profile, "fwd_times")
    seq_info = _seq_info_series_from_profile(profile)
    if not fwd_times:
        raise ValueError("Profile does not contain fwd_times at top level or in iterations[*].")
    if not seq_info:
        raise ValueError("Profile does not contain seq_info at top level or in iterations[*].")
    if len(fwd_times) != len(seq_info):
        raise ValueError(
            f"fwd_times and seq_info have different iteration counts: "
            f"{len(fwd_times)} vs {len(seq_info)}."
        )

    rows: List[FitRow] = []
    for iteration, (iteration_fwd, iteration_seq_info) in enumerate(zip(fwd_times, seq_info)):
        if iteration < drop_warmup:
            continue
        if len(iteration_fwd) != len(iteration_seq_info):
            raise ValueError(
                f"Iteration {iteration} has {len(iteration_fwd)} fwd times but "
                f"{len(iteration_seq_info)} seq_info entries."
            )
        for microbatch_id, (forward_sec, microbatch_info) in enumerate(
            zip(iteration_fwd, iteration_seq_info)
        ):
            seq_lengths = _flatten_seq_lengths(microbatch_info["seq_lengths"])
            total_tokens = float(microbatch_info.get("total_tokens", sum(seq_lengths)))
            sum_sq = float(sum(length * length for length in seq_lengths))
            rows.append(
                FitRow(
                    iteration=iteration,
                    microbatch_id=microbatch_id,
                    forward_ms=float(forward_sec) * 1000.0,
                    attn_ms=(
                        float(microbatch_info["attn_ms"])
                        if "attn_ms" in microbatch_info
                        else None
                    ),
                    mlp_ms=(
                        float(microbatch_info["linear_ms"])
                        if "linear_ms" in microbatch_info
                        else None
                    ),
                    seq_lengths=seq_lengths,
                    total_tokens=total_tokens,
                    sum_sq=sum_sq,
                )
            )
    if not rows:
        raise ValueError("No rows left after applying drop_warmup.")
    return rows


def _load_rows(profile_path: str | Path, drop_warmup: int) -> List[FitRow]:
    return _rows_from_profile(_load_profile(profile_path), drop_warmup=drop_warmup)


def _row_features(row: FitRow) -> Dict[str, float]:
    return {
        "attn_sum_sq_m": row.sum_sq / 1_000_000.0,
        "mlp_total_tokens_k": row.total_tokens / 1024.0,
    }


def _target(rows: Sequence[FitRow], name: str) -> np.ndarray:
    if name == "forward_ms":
        return np.array([row.forward_ms for row in rows], dtype=float)
    if name == "attn_ms":
        values = [row.attn_ms for row in rows]
    elif name == "mlp_ms":
        values = [row.mlp_ms for row in rows]
    else:
        raise ValueError(f"Unknown target: {name}")
    if any(value is None for value in values):
        raise ValueError(f"Target {name} is not available in every row.")
    return np.array(values, dtype=float)


def _design_matrix(
    rows: Sequence[FitRow],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.ones((len(rows), len(feature_names) + 1), dtype=float)
    for row_idx, row in enumerate(rows):
        features = _row_features(row)
        for feature_idx, feature_name in enumerate(feature_names, start=1):
            matrix[row_idx, feature_idx] = features[feature_name]
    return matrix


def _active_features(
    rows: Sequence[FitRow],
    feature_names: Sequence[str],
    min_std: float = 1e-12,
) -> tuple[List[str], List[str]]:
    active: List[str] = []
    dropped: List[str] = []
    for feature_name in feature_names:
        values = np.array([_row_features(row)[feature_name] for row in rows], dtype=float)
        if float(np.std(values)) <= min_std:
            dropped.append(feature_name)
        else:
            active.append(feature_name)
    return active, dropped


def _weighted_lstsq(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_w = np.sqrt(weights)
    return np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]


def _huber_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    delta: float = 1.0,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> np.ndarray:
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(max_iter):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            residual = x @ coef - y
        if not np.all(np.isfinite(residual)):
            break
        centered = residual - float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(centered)))
        if not math.isfinite(scale) or scale <= 1e-12:
            scale = float(np.std(residual))
        if not math.isfinite(scale) or scale <= 1e-12:
            break
        normalized = np.abs(residual) / (delta * scale)
        weights = np.ones_like(normalized)
        large_residual = normalized > 1.0
        weights[large_residual] = 1.0 / normalized[large_residual]
        next_coef = _weighted_lstsq(x, y, weights)
        if float(np.max(np.abs(next_coef - coef))) <= tol:
            coef = next_coef
            break
        coef = next_coef
    return coef


def _fit_linear_model(
    rows: Sequence[FitRow],
    target_name: str,
    feature_names: Sequence[str],
    *,
    robust: bool,
    huber_delta: float,
) -> Dict[str, Any]:
    active, dropped = _active_features(rows, feature_names)
    x = _design_matrix(rows, active)
    y = _target(rows, target_name)
    coef = _huber_fit(x, y, delta=huber_delta) if robust else np.linalg.lstsq(x, y, rcond=None)[0]
    return {
        "target": target_name,
        "features": ["bias", *active],
        "coefficients": [float(value) for value in coef],
        "dropped_constant_features": dropped,
    }


def _predict_model(model: Mapping[str, Any], rows: Sequence[FitRow]) -> np.ndarray:
    feature_names = [name for name in model["features"] if name != "bias"]
    x = _design_matrix(rows, feature_names)
    coef = np.array(model["coefficients"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return x @ coef


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residual = y_pred - y_true
    abs_percentage_error = np.abs(residual) / np.maximum(np.abs(y_true), 1e-12) * 100.0
    percentage_error = residual / np.maximum(np.abs(y_true), 1e-12) * 100.0
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return {
        "count": int(len(y_true)),
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(np.sqrt(np.mean(residual * residual))),
        "p95_abs_error_ms": float(np.percentile(np.abs(residual), 95)),
        "max_abs_error_ms": float(np.max(np.abs(residual))),
        "mae_pct": float(np.mean(abs_percentage_error)),
        "rmse_pct": float(np.sqrt(np.mean(percentage_error * percentage_error))),
        "p95_abs_error_pct": float(np.percentile(abs_percentage_error, 95)),
        "max_abs_error_pct": float(np.max(abs_percentage_error)),
        "r2": float(1.0 - np.sum(residual * residual) / denom) if denom > 0 else 0.0,
    }


def _empty_forward_metrics(original_row_count: int, excluded_row_count: int) -> Dict[str, Any]:
    return {
        "count": 0,
        "original_row_count": original_row_count,
        "excluded_row_count": excluded_row_count,
        "mae_ms": None,
        "rmse_ms": None,
        "p95_abs_error_ms": None,
        "max_abs_error_ms": None,
        "mae_pct": None,
        "rmse_pct": None,
        "p95_abs_error_pct": None,
        "max_abs_error_pct": None,
        "r2": None,
    }


def _forward_metrics_for_rows(
    rows: Sequence[FitRow],
    predictions_ms: Sequence[float],
    *,
    exclude_metric_iterations: set[int],
) -> Dict[str, Any]:
    if len(rows) != len(predictions_ms):
        raise ValueError(
            f"rows and predictions_ms have different lengths: {len(rows)} vs {len(predictions_ms)}."
        )
    filtered_rows: List[FitRow] = []
    filtered_predictions: List[float] = []
    excluded_iterations_present: set[int] = set()
    excluded_row_count = 0
    for row, prediction_ms in zip(rows, predictions_ms):
        if row.iteration in exclude_metric_iterations:
            excluded_iterations_present.add(row.iteration)
            excluded_row_count += 1
            continue
        filtered_rows.append(row)
        filtered_predictions.append(float(prediction_ms))

    if not filtered_rows:
        metric = _empty_forward_metrics(
            original_row_count=len(rows),
            excluded_row_count=excluded_row_count,
        )
    else:
        metric = _metrics(
            _target(filtered_rows, "forward_ms"),
            np.array(filtered_predictions, dtype=float),
        )
        metric["original_row_count"] = len(rows)
        metric["excluded_row_count"] = excluded_row_count
    metric["excluded_iterations_present"] = sorted(excluded_iterations_present)
    return metric


def _parse_iteration_list(value: str | None) -> List[int] | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        parsed = json.loads(stripped)
        return [int(item) for item in parsed]
    return [int(item.strip()) for item in stripped.split(",") if item.strip()]


def _empty_time_metrics(original_count: int, predicted_count: int) -> Dict[str, Any]:
    return {
        "matched_count": 0,
        "original_count": original_count,
        "predicted_count": predicted_count,
        "mae_ms": None,
        "rmse_ms": None,
        "p95_abs_error_ms": None,
        "max_abs_error_ms": None,
        "mae_pct": None,
        "rmse_pct": None,
        "p95_abs_error_pct": None,
        "max_abs_error_pct": None,
        "r2": None,
    }


def _time_list_metrics(
    original_sec: Sequence[float],
    predicted_sec: Sequence[float],
) -> Dict[str, Any]:
    matched_count = min(len(original_sec), len(predicted_sec))
    if matched_count <= 0:
        return _empty_time_metrics(
            original_count=len(original_sec),
            predicted_count=len(predicted_sec),
        )
    y_true = np.array(original_sec[:matched_count], dtype=float) * 1000.0
    y_pred = np.array(predicted_sec[:matched_count], dtype=float) * 1000.0
    metric: Dict[str, Any] = _metrics(y_true, y_pred)
    return {
        "matched_count": matched_count,
        "original_count": len(original_sec),
        "predicted_count": len(predicted_sec),
        **metric,
    }


def _series_errors(
    original_sec: Sequence[float],
    predicted_sec: Sequence[float],
) -> Dict[str, List[float]]:
    matched_count = min(len(original_sec), len(predicted_sec))
    errors = [
        float(predicted_sec[index]) - float(original_sec[index])
        for index in range(matched_count)
    ]
    signed_pct = [
        error / max(abs(float(original_sec[index])), 1e-12) * 100.0
        for index, error in enumerate(errors)
    ]
    return {
        "error_sec": errors,
        "signed_error_pct": signed_pct,
        "abs_error_pct": [abs(value) for value in signed_pct],
    }


def _ratio_factor(ratio: Sequence[float], index: int, field_name: str) -> float:
    if not ratio:
        raise ValueError(f"{field_name} must not be empty.")
    if index >= len(ratio):
        raise ValueError(f"{field_name} does not contain index {index}: {ratio}")
    fwd_ratio = float(ratio[0])
    if fwd_ratio == 0:
        raise ValueError(f"{field_name}[0] must not be zero.")
    return float(ratio[index]) / fwd_ratio


def _resolve_ratio_config(profile: Mapping[str, Any]) -> Dict[str, Any] | None:
    if "f_b_w_ratio" in profile:
        ratio = [float(value) for value in profile["f_b_w_ratio"]]
        return {
            "kind": "f_b_w_ratio",
            "schedule_method": "zbh",
            "ratio": ratio,
            "bwd_factor": _ratio_factor(ratio, 1, "f_b_w_ratio"),
            "wwd_factor": _ratio_factor(ratio, 2, "f_b_w_ratio"),
        }
    if "f_b_ratio" in profile:
        ratio = [float(value) for value in profile["f_b_ratio"]]
        return {
            "kind": "f_b_ratio",
            "schedule_method": "1f1b",
            "ratio": ratio,
            "bwd_factor": _ratio_factor(ratio, 1, "f_b_ratio"),
            "wwd_factor": 0.0,
        }
    return None


def _float_list_at(profile: Mapping[str, Any], key: str, iteration: int) -> List[float]:
    return [float(value) for value in _series_at(_time_series_from_profile(profile, key), iteration)]


def _derive_bwd_wwd_times(
    fwd_times_sec: Sequence[float],
    ratio_config: Mapping[str, Any] | None,
) -> tuple[List[float], List[float]]:
    if ratio_config is None:
        return [], []
    bwd_factor = float(ratio_config["bwd_factor"])
    wwd_factor = float(ratio_config["wwd_factor"])
    return (
        [float(fwd_time) * bwd_factor for fwd_time in fwd_times_sec],
        [float(fwd_time) * wwd_factor for fwd_time in fwd_times_sec],
    )


def _mean_flat(nested_values: Sequence[Sequence[float]]) -> float:
    total = 0.0
    count = 0
    for values in nested_values:
        for value in values:
            total += float(value)
            count += 1
    return total / count if count else 0.0


def _build_profile_shaped_predictions(
    profile: Mapping[str, Any],
    rows: Sequence[FitRow],
    predicted_forward_ms: Sequence[float],
    *,
    prediction_start_iteration: int,
) -> Dict[str, Any]:
    if len(rows) != len(predicted_forward_ms):
        raise ValueError(
            f"rows and predicted_forward_ms have different lengths: "
            f"{len(rows)} vs {len(predicted_forward_ms)}."
        )

    ratio_config = _resolve_ratio_config(profile)
    rows_by_iteration: Dict[int, List[tuple[FitRow, float]]] = {}
    for row, predicted_ms in zip(rows, predicted_forward_ms):
        rows_by_iteration.setdefault(row.iteration, []).append((row, float(predicted_ms)))

    source_fwd_times = _time_series_from_profile(profile, "fwd_times")
    source_bwd_times = _time_series_from_profile(profile, "bwd_times")
    source_wwd_times = _time_series_from_profile(profile, "wwd_times")
    source_seq_info = _seq_info_series_from_profile(profile)
    num_source_iterations = len(source_fwd_times)
    fwd_times: List[List[float]] = []
    bwd_times: List[List[float]] = []
    wwd_times: List[List[float]] = []
    seq_info: List[Any] = []
    iterations: List[Dict[str, Any]] = []
    original_iterations = profile.get("iterations", [])
    include_wwd = "f_b_w_ratio" in profile or bool(source_wwd_times)

    for iteration in range(num_source_iterations):
        pairs = sorted(
            rows_by_iteration.get(iteration, []),
            key=lambda item: item[0].microbatch_id,
        )
        predicted_fwd_times_sec = [predicted_ms / 1000.0 for _, predicted_ms in pairs]
        predicted_bwd_times_sec, predicted_wwd_times_sec = _derive_bwd_wwd_times(
            predicted_fwd_times_sec,
            ratio_config,
        )
        if iteration < prediction_start_iteration:
            iteration_fwd_times = [
                float(value) for value in _series_at(source_fwd_times, iteration)
            ]
            iteration_bwd_times = [
                float(value) for value in _series_at(source_bwd_times, iteration)
            ]
            iteration_wwd_times = [
                float(value) for value in _series_at(source_wwd_times, iteration)
            ]
            if not iteration_bwd_times:
                iteration_bwd_times, _ = _derive_bwd_wwd_times(
                    iteration_fwd_times,
                    ratio_config,
                )
            if include_wwd and not iteration_wwd_times:
                _, iteration_wwd_times = _derive_bwd_wwd_times(
                    iteration_fwd_times,
                    ratio_config,
                )
        else:
            iteration_fwd_times = predicted_fwd_times_sec
            iteration_bwd_times = predicted_bwd_times_sec
            iteration_wwd_times = predicted_wwd_times_sec

        fwd_times.append(iteration_fwd_times)
        bwd_times.append(iteration_bwd_times)
        if include_wwd:
            wwd_times.append(iteration_wwd_times)

        iteration_seq_info = _series_at(source_seq_info, iteration)
        seq_info.append(iteration_seq_info)

        source_iteration = (
            dict(original_iterations[iteration])
            if isinstance(original_iterations, list)
            and iteration < len(original_iterations)
            and isinstance(original_iterations[iteration], Mapping)
            else {}
        )
        source_iteration["fwd_times"] = iteration_fwd_times
        source_iteration["bwd_times"] = iteration_bwd_times
        if include_wwd:
            source_iteration["wwd_times"] = iteration_wwd_times
        source_iteration["seq_info"] = iteration_seq_info
        iterations.append(source_iteration)

    profile_shaped: Dict[str, Any] = {
        "configs": profile.get("configs", ""),
        "num_iterations": num_source_iterations,
        "iterations": iterations,
        "latest_iteration": iterations[-1] if iterations else {},
        "fwd_times": fwd_times,
        "bwd_times": bwd_times,
        "seq_info": seq_info,
        "all_avg_fwd": _mean_flat(fwd_times),
        "all_avg_bwd": _mean_flat(bwd_times),
        "time_source": "fit_forward_time_model",
        "time_source_note": (
            "Top-level time lists before prediction_start_iteration are copied from "
            "the source profile for fitting overhead/communication. Later fwd_times "
            "are predicted from seq_lengths; later bwd_times and wwd_times are "
            "derived from f_b_ratio/f_b_w_ratio."
        ),
        "copied_original_until_iteration": prediction_start_iteration - 1,
    }
    if include_wwd:
        profile_shaped["wwd_times"] = wwd_times
        profile_shaped["all_avg_wwd"] = _mean_flat(wwd_times)
    for key in (
        "f_b_ratio",
        "f_b_w_ratio",
        "avg_weight_update",
        "weight_update_times",
        "all_avg_wwd",
    ):
        if key in profile and key not in profile_shaped:
            profile_shaped[key] = profile[key]
    return profile_shaped


def _collect_time_metrics(
    iteration_predictions: Sequence[Mapping[str, Any]],
    allowed_iterations: set[int] | None,
    exclude_metric_iterations: set[int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    series_specs = {
        "fwd": ("original_fwd_times_sec", "predicted_fwd_times_sec"),
    }
    metrics: Dict[str, Dict[str, Any]] = {}
    for series_name, (original_key, predicted_key) in series_specs.items():
        original_values: List[float] = []
        predicted_values: List[float] = []
        original_count = 0
        predicted_count = 0
        for iteration_prediction in iteration_predictions:
            iteration = int(iteration_prediction["iteration"])
            if allowed_iterations is not None and iteration not in allowed_iterations:
                continue
            if exclude_metric_iterations is not None and iteration in exclude_metric_iterations:
                continue
            original = [float(value) for value in iteration_prediction[original_key]]
            predicted = [float(value) for value in iteration_prediction[predicted_key]]
            original_count += len(original)
            predicted_count += len(predicted)
            matched_count = min(len(original), len(predicted))
            original_values.extend(original[:matched_count])
            predicted_values.extend(predicted[:matched_count])

        if original_values:
            metric = _time_list_metrics(original_values, predicted_values)
            metric["original_count"] = original_count
            metric["predicted_count"] = predicted_count
            metric["matched_count"] = len(original_values)
        else:
            metric = _empty_time_metrics(
                original_count=original_count,
                predicted_count=predicted_count,
            )
        metrics[series_name] = metric
    return metrics


def _microbatch_detail(
    row: FitRow,
    *,
    original_fwd_times_sec: Sequence[float],
    predicted_fwd_times_sec: Sequence[float],
    predicted_bwd_times_sec: Sequence[float],
    predicted_wwd_times_sec: Sequence[float],
) -> Dict[str, Any]:
    index = row.microbatch_id

    def at(values: Sequence[float]) -> float | None:
        return float(values[index]) if index < len(values) else None

    original_fwd = at(original_fwd_times_sec)
    predicted_fwd = at(predicted_fwd_times_sec)
    fwd_error = (
        predicted_fwd - original_fwd
        if original_fwd is not None and predicted_fwd is not None
        else None
    )
    return {
        "microbatch_id": row.microbatch_id,
        "seq_lengths": row.seq_lengths,
        "total_tokens": row.total_tokens,
        "sum_seq_len_sq": row.sum_sq,
        "original_fwd_time_sec": original_fwd,
        "predicted_fwd_time_sec": predicted_fwd,
        "fwd_error_sec": fwd_error,
        "fwd_signed_error_pct": (
            fwd_error / max(abs(original_fwd), 1e-12) * 100.0
            if fwd_error is not None and original_fwd is not None
            else None
        ),
        "predicted_bwd_time_sec": at(predicted_bwd_times_sec),
        "predicted_wwd_time_sec": at(predicted_wwd_times_sec),
    }


def _build_microbatch_time_predictions(
    profile: Mapping[str, Any],
    rows: Sequence[FitRow],
    predicted_forward_ms: Sequence[float],
    *,
    train_iterations: set[int],
    validation_iterations: set[int],
    exclude_metric_iterations: set[int],
) -> Dict[str, Any]:
    if len(rows) != len(predicted_forward_ms):
        raise ValueError(
            f"rows and predicted_forward_ms have different lengths: "
            f"{len(rows)} vs {len(predicted_forward_ms)}."
        )

    ratio_config = _resolve_ratio_config(profile)
    warnings: List[str] = []
    if ratio_config is None:
        warnings.append(
            "No f_b_ratio or f_b_w_ratio found; only forward-time predictions are emitted."
        )

    rows_by_iteration: Dict[int, List[tuple[FitRow, float]]] = {}
    for row, predicted_ms in zip(rows, predicted_forward_ms):
        rows_by_iteration.setdefault(row.iteration, []).append((row, float(predicted_ms)))

    iteration_predictions: List[Dict[str, Any]] = []
    for iteration in sorted(rows_by_iteration):
        pairs = sorted(rows_by_iteration[iteration], key=lambda item: item[0].microbatch_id)
        iteration_rows = [row for row, _ in pairs]
        predicted_fwd_times_sec = [predicted_ms / 1000.0 for _, predicted_ms in pairs]
        predicted_bwd_times_sec, predicted_wwd_times_sec = _derive_bwd_wwd_times(
            predicted_fwd_times_sec,
            ratio_config,
        )

        original_fwd_times_sec = _float_list_at(profile, "fwd_times", iteration)

        iteration_prediction = {
            "iteration": iteration,
            "microbatch_count": len(iteration_rows),
            "original_fwd_times_sec": original_fwd_times_sec,
            "predicted_fwd_times_sec": predicted_fwd_times_sec,
            "fwd_errors": _series_errors(original_fwd_times_sec, predicted_fwd_times_sec),
            "predicted_bwd_times_sec": predicted_bwd_times_sec,
            "predicted_wwd_times_sec": predicted_wwd_times_sec,
            "metrics": {
                "fwd": _time_list_metrics(original_fwd_times_sec, predicted_fwd_times_sec),
            },
            "microbatches": [
                _microbatch_detail(
                    row,
                    original_fwd_times_sec=original_fwd_times_sec,
                    predicted_fwd_times_sec=predicted_fwd_times_sec,
                    predicted_bwd_times_sec=predicted_bwd_times_sec,
                    predicted_wwd_times_sec=predicted_wwd_times_sec,
                )
                for row in iteration_rows
            ],
        }
        iteration_predictions.append(iteration_prediction)

    result = {
        "ratio_config": ratio_config,
        "num_iterations": len(iteration_predictions),
        "iterations": iteration_predictions,
        "metrics": {
            "train": _collect_time_metrics(
                iteration_predictions,
                train_iterations,
                exclude_metric_iterations,
            ),
            "validation": _collect_time_metrics(
                iteration_predictions,
                validation_iterations,
                exclude_metric_iterations,
            ),
            "all": _collect_time_metrics(
                iteration_predictions,
                None,
                exclude_metric_iterations,
            ),
        },
        "warnings": warnings,
    }
    if exclude_metric_iterations:
        result["metrics_including_excluded_iterations"] = {
            "train": _collect_time_metrics(iteration_predictions, train_iterations),
            "validation": _collect_time_metrics(iteration_predictions, validation_iterations),
            "all": _collect_time_metrics(iteration_predictions, None),
        }
        present_iterations = sorted(
            {
                int(iteration_prediction["iteration"])
                for iteration_prediction in iteration_predictions
                if int(iteration_prediction["iteration"]) in exclude_metric_iterations
            }
        )
        result["metric_exclusion"] = {
            "requested_iterations": sorted(exclude_metric_iterations),
            "excluded_iterations_present": present_iterations,
            "excluded_iteration_count": len(present_iterations),
        }
    return result


def _split_train_validation(
    rows: Sequence[FitRow],
    validation_ratio: float,
) -> tuple[List[FitRow], List[FitRow]]:
    iterations = sorted({row.iteration for row in rows})
    if len(iterations) <= 1 or validation_ratio <= 0:
        rows_list = list(rows)
        return rows_list, rows_list
    validation_count = max(1, int(round(len(iterations) * validation_ratio)))
    validation_iterations = set(iterations[-validation_count:])
    train_rows = [row for row in rows if row.iteration not in validation_iterations]
    validation_rows = [row for row in rows if row.iteration in validation_iterations]
    if not train_rows:
        rows_list = list(rows)
        return rows_list, rows_list
    return train_rows, validation_rows


def _split_by_train_iterations(
    rows: Sequence[FitRow],
    train_iterations: int | None,
) -> tuple[List[FitRow], List[FitRow]]:
    if train_iterations is None or train_iterations <= 0:
        rows_list = list(rows)
        return rows_list, rows_list

    iterations = sorted({row.iteration for row in rows})
    if train_iterations >= len(iterations):
        rows_list = list(rows)
        return rows_list, rows_list

    train_iteration_set = set(iterations[:train_iterations])
    train_rows = [row for row in rows if row.iteration in train_iteration_set]
    validation_rows = [row for row in rows if row.iteration not in train_iteration_set]
    return train_rows, validation_rows


def _select_attention_features(
    train_rows: Sequence[FitRow],
    validation_rows: Sequence[FitRow],
    *,
    robust: bool,
    huber_delta: float,
    use_component_targets: bool,
) -> tuple[List[str], Dict[str, Any]]:
    target_name = "attn_ms" if use_component_targets else "forward_ms"
    results = []
    for candidate in ATTN_CANDIDATES:
        model = _fit_linear_model(
            train_rows,
            target_name=target_name,
            feature_names=candidate,
            robust=robust,
            huber_delta=huber_delta,
        )
        pred = _predict_model(model, validation_rows)
        y_true = _target(validation_rows, target_name)
        metric = _metrics(y_true, pred)
        results.append(
            {
                "candidate": list(candidate),
                "active_features": model["features"],
                "validation": metric,
            }
        )
    best = min(results, key=lambda item: item["validation"]["mae_ms"])
    return [name for name in best["candidate"]], {"target": target_name, "candidates": results, "best": best}


def _fit_component_models(
    rows: Sequence[FitRow],
    attention_features: Sequence[str],
    *,
    robust: bool,
    huber_delta: float,
) -> Dict[str, Any]:
    component_models: Dict[str, Any] = {}
    if all(row.attn_ms is not None for row in rows):
        component_models["attention"] = _fit_linear_model(
            rows,
            target_name="attn_ms",
            feature_names=attention_features,
            robust=robust,
            huber_delta=huber_delta,
        )
    if all(row.mlp_ms is not None for row in rows):
        component_models["mlp"] = _fit_linear_model(
            rows,
            target_name="mlp_ms",
            feature_names=("mlp_total_tokens_k",),
            robust=robust,
            huber_delta=huber_delta,
        )
    return component_models


def fit_forward_time_model(
    profile_path: str | Path,
    *,
    drop_warmup: int = 2,
    train_iterations: int | None = 50,
    validation_ratio: float = 0.2,
    robust: bool = True,
    huber_delta: float = 1.0,
    exclude_metric_iterations: Sequence[int] | None = None,
) -> Dict[str, Any]:
    profile = _load_profile(profile_path)
    rows = _rows_from_profile(profile, drop_warmup=drop_warmup)
    exclude_metric_iteration_set = (
        {int(iteration) for iteration in exclude_metric_iterations}
        if exclude_metric_iterations is not None
        else set()
    )
    if train_iterations is None or train_iterations <= 0:
        train_rows, validation_rows = _split_train_validation(
            rows,
            validation_ratio=validation_ratio,
        )
        selection_train_rows = train_rows
        selection_validation_rows = validation_rows
        split_strategy = "trailing_validation_ratio"
    else:
        train_rows, validation_rows = _split_by_train_iterations(
            rows,
            train_iterations=train_iterations,
        )
        selection_train_rows, selection_validation_rows = _split_train_validation(
            train_rows,
            validation_ratio=validation_ratio,
        )
        split_strategy = "first_train_iterations_after_warmup"

    can_fit_component_targets = all(
        row.attn_ms is not None and row.mlp_ms is not None for row in train_rows
    )
    attention_features, selection_report = _select_attention_features(
        selection_train_rows,
        selection_validation_rows,
        robust=robust,
        huber_delta=huber_delta,
        use_component_targets=can_fit_component_targets,
    )

    mlp_active, mlp_dropped = _active_features(train_rows, ("mlp_total_tokens_k",))
    final_features = [*attention_features, *mlp_active]
    forward_model = _fit_linear_model(
        train_rows,
        target_name="forward_ms",
        feature_names=final_features,
        robust=robust,
        huber_delta=huber_delta,
    )
    component_models = _fit_component_models(
        train_rows,
        attention_features=attention_features,
        robust=robust,
        huber_delta=huber_delta,
    )

    train_pred = _predict_model(forward_model, train_rows)
    validation_pred = _predict_model(forward_model, validation_rows)
    all_pred = _predict_model(forward_model, rows)
    train_iteration_set = set(row.iteration for row in train_rows)
    validation_iteration_set = set(row.iteration for row in validation_rows)
    prediction_start_iteration = (
        min(validation_iteration_set)
        if validation_iteration_set
        else min((row.iteration for row in rows), default=drop_warmup)
    )
    all_profile_rows = _rows_from_profile(profile, drop_warmup=0)
    all_profile_pred = _predict_model(forward_model, all_profile_rows)
    profile_shaped_predictions = _build_profile_shaped_predictions(
        profile,
        all_profile_rows,
        all_profile_pred,
        prediction_start_iteration=prediction_start_iteration,
    )
    microbatch_time_predictions = _build_microbatch_time_predictions(
        profile,
        rows,
        all_pred,
        train_iterations=train_iteration_set,
        validation_iterations=validation_iteration_set,
        exclude_metric_iterations=exclude_metric_iteration_set,
    )
    metrics = {
        "train_forward": _forward_metrics_for_rows(
            train_rows,
            train_pred,
            exclude_metric_iterations=exclude_metric_iteration_set,
        ),
        "validation_forward": _forward_metrics_for_rows(
            validation_rows,
            validation_pred,
            exclude_metric_iterations=exclude_metric_iteration_set,
        ),
        "all_forward": _forward_metrics_for_rows(
            rows,
            all_pred,
            exclude_metric_iterations=exclude_metric_iteration_set,
        ),
    }
    result = {
        **profile_shaped_predictions,
        "profile_path": str(profile_path),
        "drop_warmup": drop_warmup,
        "train_iterations": train_iterations,
        "split_strategy": split_strategy,
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_validation_rows": len(validation_rows),
        "train_iterations_used": sorted({row.iteration for row in train_rows}),
        "validation_iterations_used": sorted({row.iteration for row in validation_rows}),
        "exclude_metric_iterations": sorted(exclude_metric_iteration_set),
        "prediction_start_iteration": prediction_start_iteration,
        "prediction_start_note": (
            "Iterations before this value are copied from the source profile for "
            "warmup/model/comm fitting. Iteration-level prediction/evaluation rows "
            "should start at this value."
        ),
        "robust": robust,
        "huber_delta": huber_delta,
        "feature_units": {
            "attn_sum_sq_m": "sum(seq_len ** 2) / 1e6",
            "mlp_total_tokens_k": "sum(seq_lengths) / 1024",
        },
        "attention_feature_selection": selection_report,
        "forward_model": forward_model,
        "component_models": component_models,
        "metrics": metrics,
        "microbatch_time_predictions": microbatch_time_predictions,
        "warnings": [],
    }
    if exclude_metric_iteration_set:
        result["metrics_including_excluded_iterations"] = {
            "train_forward": _forward_metrics_for_rows(
                train_rows,
                train_pred,
                exclude_metric_iterations=set(),
            ),
            "validation_forward": _forward_metrics_for_rows(
                validation_rows,
                validation_pred,
                exclude_metric_iterations=set(),
            ),
            "all_forward": _forward_metrics_for_rows(
                rows,
                all_pred,
                exclude_metric_iterations=set(),
            ),
        }
        result["metric_exclusion"] = {
            "requested_iterations": sorted(exclude_metric_iteration_set),
            "excluded_iterations_present": sorted(
                {
                    row.iteration
                    for row in rows
                    if row.iteration in exclude_metric_iteration_set
                }
            ),
            "excluded_row_count": sum(
                1 for row in rows if row.iteration in exclude_metric_iteration_set
            ),
            "remaining_row_count": sum(
                1 for row in rows if row.iteration not in exclude_metric_iteration_set
            ),
        }
    result["warnings"].extend(microbatch_time_predictions["warnings"])
    if mlp_dropped:
        result["warnings"].append(
            "total_tokens is constant in this dataset, so the forward model cannot identify "
            "an MLP token-length slope; MLP cost is folded into the bias term. "
            "The component MLP model still reports the measured constant linear_ms if available."
        )
    return result


def _features_for_seq_lengths(seq_lengths: Sequence[float]) -> Dict[str, float]:
    lengths = [float(value) for value in seq_lengths]
    if not lengths:
        raise ValueError("seq_lengths must not be empty.")
    total_tokens = sum(lengths)
    return {
        "attn_sum_sq_m": sum(length * length for length in lengths) / 1_000_000.0,
        "mlp_total_tokens_k": total_tokens / 1024.0,
    }


def _predict_from_features(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    value = 0.0
    for coefficient, feature_name in zip(model["coefficients"], model["features"]):
        if feature_name == "bias":
            value += float(coefficient)
        else:
            value += float(coefficient) * float(features[feature_name])
    return value


def predict_forward_time_ms(seq_lengths: Sequence[float], fitted_model: Mapping[str, Any]) -> float:
    features = _features_for_seq_lengths(seq_lengths)
    return _predict_from_features(fitted_model["forward_model"], features)


def _format_formula(model: Mapping[str, Any], target_name: str) -> str:
    terms = []
    for coefficient, feature_name in zip(model["coefficients"], model["features"]):
        if feature_name == "bias":
            terms.append(f"{coefficient:.8g}")
        else:
            terms.append(f"{coefficient:.8g} * {feature_name}")
    return f"{target_name} = " + " + ".join(terms)


def _format_time_metric(metrics: Mapping[str, Any]) -> str:
    if not metrics.get("matched_count"):
        return (
            "no matched original values "
            f"(original={metrics.get('original_count', 0)}, "
            f"predicted={metrics.get('predicted_count', 0)})"
        )
    return (
        f"MAE={metrics['mae_pct']:.2f}%, "
        f"RMSE={metrics['rmse_pct']:.2f}%, "
        f"P95={metrics['p95_abs_error_pct']:.2f}%, "
        f"matched={metrics['matched_count']}, "
        f"original={metrics['original_count']}, "
        f"predicted={metrics['predicted_count']}"
    )


def _format_forward_metric(metrics: Mapping[str, Any]) -> str:
    if not metrics.get("count"):
        return (
            "no matched rows "
            f"(original={metrics.get('original_row_count', 0)}, "
            f"excluded={metrics.get('excluded_row_count', 0)})"
        )
    suffix = ""
    if metrics.get("excluded_row_count"):
        suffix = (
            f", rows={metrics['count']}/"
            f"{metrics.get('original_row_count', metrics['count'])}"
        )
    return (
        f"MAE={metrics['mae_pct']:.2f}%, "
        f"RMSE={metrics['rmse_pct']:.2f}%, "
        f"P95={metrics['p95_abs_error_pct']:.2f}%, "
        f"R2={metrics['r2']:.4f}"
        f"{suffix}"
    )


def _print_report(model: Mapping[str, Any]) -> None:
    print("Forward model:")
    print(_format_formula(model["forward_model"], "forward_ms"))
    train_iterations = model["train_iterations_used"]
    validation_iterations = model["validation_iterations_used"]
    if train_iterations and validation_iterations:
        print(
            "\nSplit: "
            f"train iterations {train_iterations[0]}-{train_iterations[-1]} "
            f"({len(train_iterations)} iterations, {model['num_train_rows']} rows), "
            f"validation iterations {validation_iterations[0]}-{validation_iterations[-1]} "
            f"({len(validation_iterations)} iterations, {model['num_validation_rows']} rows)"
        )
    print("\nMetrics:")
    if model.get("metric_exclusion"):
        exclusion = model["metric_exclusion"]
        print(
            "  excluding known noisy iterations from metrics: "
            f"{exclusion['excluded_iterations_present']} "
            f"({exclusion['excluded_row_count']} rows excluded)"
        )
    for split_name, metrics in model["metrics"].items():
        print(f"  {split_name}: {_format_forward_metric(metrics)}")
    time_predictions = model.get("microbatch_time_predictions")
    if time_predictions:
        ratio_config = time_predictions.get("ratio_config")
        if ratio_config:
            print(
                "\nDerived micro-batch time predictions: "
                f"{ratio_config['kind']} -> {ratio_config['schedule_method']}"
            )
        else:
            print("\nDerived micro-batch time predictions: forward only")
        print("  metrics compare forward time only; bwd/wwd are derived from the ratio.")
        for split_name, split_metrics in time_predictions["metrics"].items():
            print(f"  {split_name}:")
            for series_name, series_metrics in split_metrics.items():
                print(f"    {series_name}: {_format_time_metric(series_metrics)}")
    if model["component_models"]:
        print("\nComponent models:")
        for component_name, component_model in model["component_models"].items():
            print(f"  {component_name}: {_format_formula(component_model, component_model['target'])}")
    if model["warnings"]:
        print("\nWarnings:")
        for warning in model["warnings"]:
            print(f"  - {warning}")


def _parse_predict_lengths(value: str) -> List[float]:
    parsed = json.loads(value)
    return _flatten_seq_lengths(parsed)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an interpretable forward-time model from micro-batch seq_lengths. "
            "The fitted formula is forward = overhead + attention(seq_lengths) + mlp(total_tokens)."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/ma/Downloads/for_fit_model.json",
        help="Path to the profile JSON.",
    )
    parser.add_argument(
        "--output",
        default="forward_time_model.json",
        help="Where to write the fitted model JSON.",
    )
    parser.add_argument(
        "--drop-warmup",
        type=int,
        default=2,
        help="Number of leading iterations to discard before fitting.",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=50,
        help=(
            "After dropping warmup iterations, fit with the first N iterations and "
            "validate on the remaining iterations. Set to 0 to use --validation-ratio "
            "as the outer split instead."
        ),
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of trailing training iterations used only for attention-feature "
            "selection. Also used as the outer split when --train-iterations is 0."
        ),
    )
    parser.add_argument(
        "--no-robust",
        action="store_true",
        help="Use ordinary least squares instead of Huber robust regression.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help="Huber delta used by robust regression.",
    )
    parser.add_argument(
        "--exclude-metric-iterations",
        help=(
            "Comma-separated or JSON list of known noisy iterations to exclude only "
            "from aggregate accuracy metrics. Predictions and per-iteration details "
            "are still emitted for every iteration."
        ),
    )
    parser.add_argument(
        "--predict",
        help="Optional seq_lengths JSON, e.g. '[4096]' or '[832, 3264]', to predict after fitting.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    exclude_metric_iterations = _parse_iteration_list(args.exclude_metric_iterations)
    model = fit_forward_time_model(
        args.input,
        drop_warmup=args.drop_warmup,
        train_iterations=args.train_iterations,
        validation_ratio=args.validation_ratio,
        robust=not args.no_robust,
        huber_delta=args.huber_delta,
        exclude_metric_iterations=exclude_metric_iterations,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2, ensure_ascii=False)
        file.write("\n")
    _print_report(model)
    print(f"\nSaved model to {output_path}")
    if args.predict:
        seq_lengths = _parse_predict_lengths(args.predict)
        prediction = predict_forward_time_ms(seq_lengths, model)
        print(f"Prediction for {seq_lengths}: {prediction:.4f} ms")


if __name__ == "__main__":
    main()
