from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence


def _load_json(path: str | Path) -> Mapping[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * q
        low = int(pos)
        high = min(low + 1, len(sorted_values) - 1)
        weight = pos - low
        return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight

    return {
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": sorted_values[-1],
    }


def _metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    errors = [row["error_sec"] for row in rows]
    abs_errors = [abs(value) for value in errors]
    pct_errors = [row["error_pct"] for row in rows]
    abs_pct_errors = [abs(value) for value in pct_errors]
    squared_error = [value * value for value in errors]
    squared_pct_error = [value * value for value in pct_errors]
    return {
        "count": len(rows),
        "mean_predicted_sec": mean(row["predicted_sec"] for row in rows) if rows else 0.0,
        "mean_actual_sec": mean(row["actual_sec"] for row in rows) if rows else 0.0,
        "mean_error_sec": mean(errors) if errors else 0.0,
        "mae_sec": mean(abs_errors) if abs_errors else 0.0,
        "rmse_sec": (mean(squared_error) ** 0.5) if squared_error else 0.0,
        "mae_pct": mean(abs_pct_errors) if abs_pct_errors else 0.0,
        "rmse_pct": (mean(squared_pct_error) ** 0.5) if squared_pct_error else 0.0,
        "p95_abs_error_sec": _percentiles(abs_errors).get("p95", 0.0),
        "p95_abs_error_pct": _percentiles(abs_pct_errors).get("p95", 0.0),
        "error_sec_percentiles": _percentiles(errors),
        "abs_error_sec_percentiles": _percentiles(abs_errors),
        "abs_error_pct_percentiles": _percentiles(abs_pct_errors),
    }


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


def _metric_rows_with_exclusion(
    rows: Sequence[Mapping[str, float]],
    exclude_iterations: Sequence[int] | None,
) -> tuple[List[Mapping[str, float]], Dict[str, Any] | None]:
    if exclude_iterations is None:
        return list(rows), None

    exclude_set = {int(iteration) for iteration in exclude_iterations}
    metric_rows = [row for row in rows if int(row["iteration"]) not in exclude_set]
    excluded_rows = [dict(row) for row in rows if int(row["iteration"]) in exclude_set]
    present_excluded = sorted({int(row["iteration"]) for row in excluded_rows})
    return metric_rows, {
        "requested_iterations": sorted(exclude_set),
        "excluded_iterations_present": present_excluded,
        "excluded_count": len(excluded_rows),
        "remaining_count": len(metric_rows),
        "excluded_rows": excluded_rows,
    }


def _abs_error_pct_filter(
    rows: Sequence[Mapping[str, float]],
    threshold: float,
    *,
    ground_truth_iterations: Sequence[int] | None,
    ground_truth_window: int,
) -> Dict[str, Any]:
    detected = [dict(row) for row in rows if float(row["abs_error_pct"]) >= threshold]
    detected_ids = [int(row["iteration"]) for row in detected]
    payload: Dict[str, Any] = {
        "abs_error_pct_threshold": threshold,
        "count": len(detected),
        "iteration_ids": detected_ids,
        "iterations": detected,
    }
    if ground_truth_iterations is None:
        return payload

    ground_truth_ids = [int(iteration) for iteration in ground_truth_iterations]
    used_detected: set[int] = set()
    matched = []
    missed = []
    for truth_iteration in ground_truth_ids:
        candidates = [
            detected_iteration
            for detected_iteration in detected_ids
            if detected_iteration not in used_detected
            and abs(detected_iteration - truth_iteration) <= ground_truth_window
        ]
        if not candidates:
            missed.append(truth_iteration)
            continue
        detected_iteration = min(candidates, key=lambda item: abs(item - truth_iteration))
        used_detected.add(detected_iteration)
        matched.append(
            {
                "ground_truth_iteration": truth_iteration,
                "detected_iteration": detected_iteration,
                "offset": detected_iteration - truth_iteration,
            }
        )

    false_positive_ids = [
        iteration for iteration in detected_ids if iteration not in used_detected
    ]
    payload["ground_truth"] = {
        "iterations": ground_truth_ids,
        "window": ground_truth_window,
        "matched": matched,
        "missed_iterations": missed,
        "false_positive_iterations": false_positive_ids,
        "true_positive_count": len(matched),
        "false_positive_count": len(false_positive_ids),
        "false_negative_count": len(missed),
        "precision": len(matched) / len(detected_ids) if detected_ids else 0.0,
        "recall": len(matched) / len(ground_truth_ids) if ground_truth_ids else 0.0,
    }
    return payload


def _compare_target(
    predictions: Sequence[Mapping[str, Any]],
    actual_values: Sequence[float],
    *,
    align_by_iteration: bool,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for position, prediction in enumerate(predictions):
        actual_index = int(prediction["iteration"]) if align_by_iteration else position
        if actual_index >= len(actual_values):
            continue
        predicted_sec = float(prediction["predicted_iteration_time_sec"])
        actual_sec = float(actual_values[actual_index])
        if actual_sec == 0:
            continue
        error_sec = predicted_sec - actual_sec
        error_pct = error_sec / actual_sec * 100.0
        rows.append(
            {
                "position": position,
                "iteration": actual_index,
                "predicted_sec": predicted_sec,
                "actual_sec": actual_sec,
                "error_sec": error_sec,
                "abs_error_sec": abs(error_sec),
                "error_pct": error_pct,
                "abs_error_pct": abs(error_pct),
            }
        )
    return rows


def compare_iteration_predictions(
    prediction_path: str | Path,
    actual_path: str | Path,
    *,
    align_by_iteration: bool = True,
    abs_error_pct_threshold: float | None = None,
    ground_truth_iterations: Sequence[int] | None = None,
    ground_truth_window: int = 0,
    metric_exclude_iterations: Sequence[int] | None = None,
) -> Dict[str, Any]:
    prediction_data = _load_json(prediction_path)
    actual_data = _load_json(actual_path)
    predictions = prediction_data["iterations"]
    targets = {}
    for target_name in ("iter_time",):
        if target_name in actual_data:
            rows = _compare_target(
                predictions,
                actual_data[target_name],
                align_by_iteration=align_by_iteration,
            )
            top_abs_error_iterations = sorted(
                rows,
                key=lambda row: row["abs_error_sec"],
                reverse=True,
            )[:20]
            metric_rows, metric_exclusion = _metric_rows_with_exclusion(
                rows,
                metric_exclude_iterations,
            )
            payload = {
                "metrics": _metrics(metric_rows),
                "per_iteration": rows,
                "top_abs_error_iterations": top_abs_error_iterations,
            }
            if metric_exclusion is not None:
                payload["metric_exclusion"] = metric_exclusion
                payload["metrics_including_excluded_iterations"] = _metrics(rows)
            if abs_error_pct_threshold is not None:
                payload["abs_error_pct_filter"] = _abs_error_pct_filter(
                    rows,
                    abs_error_pct_threshold,
                    ground_truth_iterations=ground_truth_iterations,
                    ground_truth_window=ground_truth_window,
                )
            targets[target_name] = payload

    return {
        "prediction_path": str(prediction_path),
        "actual_path": str(actual_path),
        "prediction_configs": prediction_data.get("configs", ""),
        "actual_configs": actual_data.get("configs", ""),
        "align_by_iteration": align_by_iteration,
        "abs_error_pct_threshold": abs_error_pct_threshold,
        "ground_truth_iterations": (
            [int(iteration) for iteration in ground_truth_iterations]
            if ground_truth_iterations is not None
            else None
        ),
        "ground_truth_window": ground_truth_window,
        "metric_exclude_iterations": (
            [int(iteration) for iteration in metric_exclude_iterations]
            if metric_exclude_iterations is not None
            else None
        ),
        "prediction_count": len(predictions),
        "actual_iter_time_count": len(actual_data.get("iter_time", [])),
        "targets": targets,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predicted_iteration_time_sec from prediction JSON with measured "
            "iter_time in a real iteration JSON."
        )
    )
    parser.add_argument(
        "--prediction",
        default="/Users/ma/Downloads/llama_zbh1_l40_hid5120_seq16384_voc32000_mb8_pre_iter.json",
        help="Prediction JSON containing iterations[*].predicted_iteration_time_sec.",
    )
    parser.add_argument(
        "--actual",
        default="/Users/ma/Downloads/llama_zbh1_l40_hid5120_seq16384_voc32000_mb8_iter.json",
        help="Measured JSON containing iter_time.",
    )
    parser.add_argument(
        "--output",
        default="iteration_prediction_comparison.json",
        help="Where to write detailed comparison JSON.",
    )
    parser.add_argument(
        "--no-align-by-iteration",
        action="store_true",
        help="Compare by list position instead of prediction iteration id.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of per-iteration comparison rows to print before summary metrics.",
    )
    parser.add_argument(
        "--abs-error-pct-threshold",
        type=float,
        help="Filter iterations whose absolute prediction error percentage is at least this value.",
    )
    parser.add_argument(
        "--ground-truth-iterations",
        help=(
            "Optional comma-separated or JSON list of ground-truth change-point iterations "
            "used to evaluate the abs-error filter."
        ),
    )
    parser.add_argument(
        "--ground-truth-window",
        type=int,
        default=0,
        help="Allowed +/- iteration offset when matching filtered rows to ground truth.",
    )
    parser.add_argument(
        "--exclude-metric-iterations",
        help=(
            "Comma-separated or JSON list of known noisy iterations to exclude from "
            "aggregate accuracy metrics. Per-iteration rows and abs-error filtering "
            "still use the full data."
        ),
    )
    return parser


def _print_metric_line(target_name: str, metrics: Mapping[str, float]) -> None:
    print(
        f"{target_name}: count={metrics['count']}, "
        f"mean_pred={metrics['mean_predicted_sec']:.4f}s, "
        f"mean_actual={metrics['mean_actual_sec']:.4f}s, "
        f"mean_error={metrics['mean_error_sec']:.4f}s, "
        f"MAE={metrics['mae_sec']:.4f}s ({metrics['mae_pct']:.2f}%), "
        f"RMSE={metrics['rmse_sec']:.4f}s ({metrics['rmse_pct']:.2f}%), "
        f"P95_abs={metrics['p95_abs_error_sec']:.4f}s ({metrics['p95_abs_error_pct']:.2f}%)"
    )


def _print_iteration_preview(target_name: str, payload: Mapping[str, Any], preview_rows: int) -> None:
    print(f"{target_name} per-iteration differences (first {preview_rows}):")
    for row in payload["per_iteration"][:preview_rows]:
        print(
            f"  iter={row['iteration']}: "
            f"pred={row['predicted_sec']:.4f}s, "
            f"actual={row['actual_sec']:.4f}s, "
            f"error={row['error_sec']:+.4f}s "
            f"({row['error_pct']:+.2f}%)"
        )
    print(f"{target_name} largest absolute differences:")
    for row in payload["top_abs_error_iterations"][:preview_rows]:
        print(
            f"  iter={row['iteration']}: "
            f"pred={row['predicted_sec']:.4f}s, "
            f"actual={row['actual_sec']:.4f}s, "
            f"abs_error={row['abs_error_sec']:.4f}s "
            f"({row['abs_error_pct']:.2f}%)"
        )


def _print_abs_error_filter(target_name: str, payload: Mapping[str, Any], preview_rows: int) -> None:
    filter_payload = payload.get("abs_error_pct_filter")
    if not filter_payload:
        return
    threshold = filter_payload["abs_error_pct_threshold"]
    print(
        f"{target_name} abs_error_pct >= {threshold:.2f}%: "
        f"{filter_payload['count']} iterations"
    )
    print(f"  iteration_ids: {filter_payload['iteration_ids']}")
    ground_truth = filter_payload.get("ground_truth")
    if ground_truth:
        print(
            "  ground_truth: "
            f"TP={ground_truth['true_positive_count']}, "
            f"FP={ground_truth['false_positive_count']}, "
            f"FN={ground_truth['false_negative_count']}, "
            f"precision={ground_truth['precision']:.3f}, "
            f"recall={ground_truth['recall']:.3f}"
        )
        if ground_truth["missed_iterations"]:
            print(f"  missed: {ground_truth['missed_iterations']}")
        false_positive_preview = ground_truth["false_positive_iterations"][:preview_rows]
        if false_positive_preview:
            print(f"  false positives first {len(false_positive_preview)}: {false_positive_preview}")


def _print_metric_exclusion(target_name: str, payload: Mapping[str, Any]) -> None:
    exclusion = payload.get("metric_exclusion")
    if not exclusion:
        return
    print(
        f"{target_name} metrics excluded {exclusion['excluded_count']} known noisy rows; "
        f"remaining={exclusion['remaining_count']}."
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    ground_truth_iterations = _parse_iteration_list(args.ground_truth_iterations)
    metric_exclude_iterations = _parse_iteration_list(args.exclude_metric_iterations)
    result = compare_iteration_predictions(
        args.prediction,
        args.actual,
        align_by_iteration=not args.no_align_by_iteration,
        abs_error_pct_threshold=args.abs_error_pct_threshold,
        ground_truth_iterations=ground_truth_iterations,
        ground_truth_window=args.ground_truth_window,
        metric_exclude_iterations=metric_exclude_iterations,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(
        f"Compared {result['prediction_count']} predictions with "
        f"{result['actual_iter_time_count']} measured iter_time values "
        f"(align_by_iteration={result['align_by_iteration']})."
    )
    for target_name, payload in result["targets"].items():
        _print_iteration_preview(target_name, payload, args.preview_rows)
        _print_metric_exclusion(target_name, payload)
        _print_metric_line(target_name, payload["metrics"])
        _print_abs_error_filter(target_name, payload, args.preview_rows)
    print(f"Saved comparison to {output_path}")


if __name__ == "__main__":
    main()
