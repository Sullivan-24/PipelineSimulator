from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

from scipy.optimize import minimize_scalar

from compare_iteration_predictions import compare_iteration_predictions
from compare_iteration_predictions import _parse_iteration_list as parse_iteration_list
from compare_iteration_predictions import _print_abs_error_filter as print_abs_error_filter
from compare_iteration_predictions import _print_metric_line as print_comparison_metric_line
from compare_iteration_predictions import _print_iteration_preview as print_iteration_preview
from microbatch_schedule_generator import generate_microbatch_schedule


def _normalize_method(method: str) -> str:
    return method.lower().replace("-", "").replace("_", "")


def _infer_method_from_profile(profile: Mapping[str, Any]) -> str:
    if "f_b_w_ratio" in profile:
        return "zbh"
    if "f_b_ratio" in profile:
        return "1f1b"
    configs = str(profile.get("configs", ""))
    normalized = configs.lower().replace(":", "").replace("_", "").replace("-", "")
    if "zbh" in normalized:
        return "zbh"
    if "1f1b" in normalized or "onefoneb" in normalized:
        return "1f1b"
    return "1f1b"


def _parse_int_from_configs(configs: str, key: str) -> int | None:
    match = re.search(rf"{re.escape(key)}(\d+)", configs)
    return int(match.group(1)) if match else None


def _load_profile(profile_path: str | Path) -> Mapping[str, Any]:
    with open(profile_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_prediction_start_iteration(
    profile: Mapping[str, Any],
    requested_drop_warmup: int,
) -> int:
    prediction_start = profile.get("prediction_start_iteration")
    if prediction_start is None:
        return requested_drop_warmup
    return max(requested_drop_warmup, int(prediction_start))


def _validate_iteration_times(
    iteration: int,
    fwd_times: Sequence[float],
) -> None:
    if not fwd_times:
        raise ValueError(f"Iteration {iteration} has no micro-batches.")


def _ratio_factor(ratio: Sequence[float], index: int, field_name: str) -> float:
    if not ratio:
        raise ValueError(f"{field_name} must not be empty.")
    if index >= len(ratio):
        raise ValueError(f"{field_name} does not contain index {index}: {ratio}")
    fwd_ratio = float(ratio[0])
    if fwd_ratio == 0:
        raise ValueError(f"{field_name}[0] must not be zero.")
    return float(ratio[index]) / fwd_ratio


def _resolve_ratio_config(
    profile: Mapping[str, Any],
    f_b_ratio: Sequence[float] | None,
    f_b_w_ratio: Sequence[float] | None,
) -> Dict[str, Any]:
    if f_b_w_ratio is None and "f_b_w_ratio" in profile:
        f_b_w_ratio = profile["f_b_w_ratio"]
    if f_b_ratio is None and "f_b_ratio" in profile:
        f_b_ratio = profile["f_b_ratio"]

    if f_b_w_ratio is not None:
        return {
            "kind": "f_b_w_ratio",
            "ratio": [float(value) for value in f_b_w_ratio],
            "b_factor": _ratio_factor(f_b_w_ratio, 1, "f_b_w_ratio"),
            "w_factor": _ratio_factor(f_b_w_ratio, 2, "f_b_w_ratio"),
        }
    if f_b_ratio is not None:
        return {
            "kind": "f_b_ratio",
            "ratio": [float(value) for value in f_b_ratio],
            "bwd_factor": _ratio_factor(f_b_ratio, 1, "f_b_ratio"),
        }
    raise ValueError("No ratio found. Provide --f-b-ratio or --f-b-w-ratio.")


def _build_microbatch_times(
    fwd_times: Sequence[float],
    *,
    ratio_config: Mapping[str, Any],
    split_backward: bool,
    backward_split_ratio: float,
) -> tuple[List[Dict[str, float]], List[float], List[float]]:
    if not 0.0 <= backward_split_ratio <= 1.0:
        raise ValueError("--backward-split-ratio must be in [0, 1].")

    microbatch_times: List[Dict[str, float]] = []
    derived_b_times: List[float] = []
    derived_w_times: List[float] = []
    for fwd_time in fwd_times:
        fwd_time = float(fwd_time)
        if ratio_config["kind"] == "f_b_w_ratio":
            b_time = fwd_time * float(ratio_config["b_factor"])
            w_time = fwd_time * float(ratio_config["w_factor"])
        else:
            aggregate_backward = fwd_time * float(ratio_config["bwd_factor"])
            b_time = aggregate_backward * backward_split_ratio
            w_time = aggregate_backward * (1.0 - backward_split_ratio)

        derived_b_times.append(b_time)
        derived_w_times.append(w_time)
        if split_backward:
            microbatch_times.append({"f": fwd_time, "b": b_time, "w": w_time})
        else:
            microbatch_times.append({"f": fwd_time, "b": b_time + w_time})
    return microbatch_times, derived_b_times, derived_w_times


def _build_microbatch_times_from_actual(
    fwd_times: Sequence[float],
    bwd_times: Sequence[float],
    wwd_times: Sequence[float] | None = None,
    *,
    ratio_config: Mapping[str, Any],
    split_backward: bool,
    backward_split_ratio: float,
) -> tuple[List[Dict[str, float]], List[float], List[float]]:
    if len(bwd_times) > len(fwd_times):
        raise ValueError(
            f"bwd_times ({len(bwd_times)}) cannot be longer than fwd_times ({len(fwd_times)})."
        )
    if wwd_times is not None and len(wwd_times) > len(fwd_times):
        raise ValueError(
            f"wwd_times ({len(wwd_times)}) cannot be longer than fwd_times ({len(fwd_times)})."
        )
    microbatch_times: List[Dict[str, float]] = []
    derived_b_times: List[float] = []
    derived_w_times: List[float] = []
    wwd_values = list(wwd_times) if wwd_times is not None else None
    for microbatch_id, fwd_time in enumerate(fwd_times):
        fwd_time = float(fwd_time)
        has_actual_bwd = microbatch_id < len(bwd_times)
        has_actual_wwd = wwd_values is not None and microbatch_id < len(wwd_values)

        if has_actual_bwd and wwd_values is None:
            bwd_value = float(bwd_times[microbatch_id])
            bwd_total = bwd_value
            b_time = bwd_total * backward_split_ratio
            w_time = bwd_total * (1.0 - backward_split_ratio)
        elif has_actual_bwd and has_actual_wwd:
            b_time = float(bwd_times[microbatch_id])
            w_time = float(wwd_values[microbatch_id])
            bwd_total = b_time + w_time
        elif ratio_config["kind"] == "f_b_w_ratio":
            b_time = fwd_time * float(ratio_config["b_factor"])
            w_time = fwd_time * float(ratio_config["w_factor"])
            bwd_total = b_time + w_time
        else:
            bwd_total = fwd_time * float(ratio_config["bwd_factor"])
            b_time = bwd_total * backward_split_ratio
            w_time = bwd_total * (1.0 - backward_split_ratio)
        derived_b_times.append(b_time)
        derived_w_times.append(w_time)
        if split_backward:
            microbatch_times.append({"f": fwd_time, "b": b_time, "w": w_time})
        else:
            microbatch_times.append({"f": fwd_time, "b": bwd_total})
    return microbatch_times, derived_b_times, derived_w_times


def _uses_copied_original_iteration(profile: Mapping[str, Any], iteration: int) -> bool:
    copied_until = profile.get("copied_original_until_iteration")
    return copied_until is not None and iteration <= int(copied_until)


def _fit_comm_time(
    profile: Mapping[str, Any],
    actual_iter_times: Sequence[float],
    *,
    method: str,
    num_stages: int,
    split_backward: bool,
    ratio_config: Mapping[str, Any],
    backward_split_ratio: float,
    max_act: int,
    drop_warmup: int,
    include_weight_update: bool,
    fit_iters: int = 50,
    use_actual_bwd: bool = False,
) -> Dict[str, float]:
    fwd_times = profile["fwd_times"]
    bwd_times_all = profile.get("bwd_times", [])
    wwd_times_all = profile.get("wwd_times", [])
    weight_update_time = (
        float(profile.get("avg_weight_update", 0.0) or 0.0) if include_weight_update else 0.0
    )

    fit_data: List[tuple[List[Dict[str, float]], float]] = []
    for iteration, iteration_fwd_times in enumerate(fwd_times):
        if iteration < drop_warmup:
            continue
        if len(fit_data) >= fit_iters:
            break
        if iteration >= len(actual_iter_times):
            break
        _validate_iteration_times(iteration, iteration_fwd_times)
        use_profile_backward = (
            (use_actual_bwd or _uses_copied_original_iteration(profile, iteration))
            and iteration < len(bwd_times_all)
        )
        if use_profile_backward:
            iteration_wwd_times = (
                wwd_times_all[iteration]
                if iteration < len(wwd_times_all)
                and isinstance(wwd_times_all[iteration], list)
                else None
            )
            microbatch_times, _, _ = _build_microbatch_times_from_actual(
                iteration_fwd_times,
                bwd_times_all[iteration],
                iteration_wwd_times,
                ratio_config=ratio_config,
                split_backward=split_backward,
                backward_split_ratio=backward_split_ratio,
            )
        else:
            microbatch_times, _, _ = _build_microbatch_times(
                iteration_fwd_times,
                ratio_config=ratio_config,
                split_backward=split_backward,
                backward_split_ratio=backward_split_ratio,
            )
        fit_data.append((microbatch_times, float(actual_iter_times[iteration])))

    if not fit_data:
        return {"comm_time": 0.0, "overhead": 0.0}

    def _makespans_for(comm_time_candidate: float) -> List[float]:
        result = []
        for microbatch_times, _ in fit_data:
            schedule = generate_microbatch_schedule(
                num_stages=num_stages,
                microbatch_times=microbatch_times,
                method=method,
                split_backward=split_backward,
                max_act=max_act,
                comm_time=comm_time_candidate,
            )
            result.append(float(schedule["makespan"]) + weight_update_time)
        return result

    def objective(comm_time_candidate: float) -> float:
        makespans = _makespans_for(comm_time_candidate)
        residuals = [actual - ms for (_, actual), ms in zip(fit_data, makespans)]
        best_overhead = max(mean(residuals), 0.0)
        return mean((r - best_overhead) ** 2 for r in residuals)

    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    best_comm_time = float(res.x)
    makespans = _makespans_for(best_comm_time)
    residuals = [actual - ms for (_, actual), ms in zip(fit_data, makespans)]
    best_overhead = max(mean(residuals), 0.0)
    return {"comm_time": best_comm_time, "overhead": best_overhead}


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


def predict_iteration_times_from_profile(
    profile_path: str | Path,
    *,
    method: str | None = None,
    num_stages: int | None = None,
    split_backward: bool | None = None,
    f_b_ratio: Sequence[float] | None = None,
    f_b_w_ratio: Sequence[float] | None = None,
    backward_split_ratio: float = 1.0,
    comm_time: float = 0.0,
    max_act: int = 1,
    drop_warmup: int = 0,
    include_weight_update: bool = False,
    include_full_schedules: bool = False,
    fit_comm_time: bool = False,
    fit_iters: int = 50,
    actual_iter_times: Sequence[float] | None = None,
    use_actual_bwd: bool = False,
) -> Dict[str, Any]:
    profile = _load_profile(profile_path)
    fwd_times = profile["fwd_times"]
    requested_drop_warmup = drop_warmup
    prediction_start_iteration = _resolve_prediction_start_iteration(
        profile,
        requested_drop_warmup,
    )
    ratio_config = _resolve_ratio_config(profile, f_b_ratio=f_b_ratio, f_b_w_ratio=f_b_w_ratio)

    configs = str(profile.get("configs", ""))
    resolved_num_stages = num_stages or _parse_int_from_configs(configs, "PPSIZE")
    if resolved_num_stages is None:
        raise ValueError("Cannot infer num_stages from configs; pass --num-stages explicitly.")

    method_normalized = _normalize_method(method or _infer_method_from_profile(profile))
    resolved_split_backward = split_backward
    if resolved_split_backward is None:
        resolved_split_backward = method_normalized == "zbh" or ratio_config["kind"] == "f_b_w_ratio"

    if method_normalized == "zbh" and not resolved_split_backward:
        raise ValueError("ZBH requires split backward. Use --split-backward.")

    fitted_comm_time: float | None = None
    fitted_overhead: float | None = None
    if fit_comm_time:
        if actual_iter_times is None:
            raise ValueError("--fit-comm-time requires --actual to provide measured iteration times.")
        fit_result = _fit_comm_time(
            profile,
            actual_iter_times,
            method=method_normalized,
            num_stages=resolved_num_stages,
            split_backward=resolved_split_backward,
            ratio_config=ratio_config,
            backward_split_ratio=backward_split_ratio,
            max_act=max_act,
            drop_warmup=requested_drop_warmup,
            include_weight_update=include_weight_update,
            fit_iters=fit_iters,
            use_actual_bwd=use_actual_bwd,
        )
        fitted_comm_time = fit_result["comm_time"]
        fitted_overhead = fit_result["overhead"]
        comm_time = fitted_comm_time

    overhead = fitted_overhead if fitted_overhead is not None else 0.0
    bwd_times_all = profile.get("bwd_times", []) if use_actual_bwd else []
    weight_update_time = float(profile.get("avg_weight_update", 0.0) or 0.0)
    iteration_results: List[Dict[str, Any]] = []
    for iteration, iteration_fwd_times in enumerate(fwd_times):
        if iteration < prediction_start_iteration:
            continue
        _validate_iteration_times(iteration, iteration_fwd_times)
        if use_actual_bwd and iteration < len(bwd_times_all):
            microbatch_times, derived_b_times, derived_w_times = _build_microbatch_times_from_actual(
                iteration_fwd_times,
                bwd_times_all[iteration],
                ratio_config=ratio_config,
                split_backward=resolved_split_backward,
                backward_split_ratio=backward_split_ratio,
            )
        else:
            microbatch_times, derived_b_times, derived_w_times = _build_microbatch_times(
                iteration_fwd_times,
                ratio_config=ratio_config,
                split_backward=resolved_split_backward,
                backward_split_ratio=backward_split_ratio,
            )
        schedule = generate_microbatch_schedule(
            num_stages=resolved_num_stages,
            microbatch_times=microbatch_times,
            method=method_normalized,
            split_backward=resolved_split_backward,
            max_act=max_act,
            comm_time=comm_time,
        )
        schedule_makespan = float(schedule["makespan"])
        predicted_iteration_time = schedule_makespan + overhead
        if include_weight_update:
            predicted_iteration_time += weight_update_time
        item = {
            "iteration": iteration,
            "num_microbatches": len(microbatch_times),
            "schedule_makespan_sec": schedule_makespan,
            "schedule_makespan_ms": schedule_makespan * 1000.0,
            "predicted_iteration_time_sec": predicted_iteration_time,
            "predicted_iteration_time_ms": predicted_iteration_time * 1000.0,
            "mean_fwd_time_ms": mean(float(value) for value in iteration_fwd_times) * 1000.0,
            "mean_b_time_ms": mean(derived_b_times) * 1000.0,
            "mean_w_time_ms": mean(derived_w_times) * 1000.0,
            "mean_backward_total_time_ms": mean(
                b_time + w_time for b_time, w_time in zip(derived_b_times, derived_w_times)
            )
            * 1000.0,
        }
        if include_full_schedules:
            item["schedule"] = schedule
        iteration_results.append(item)

    predicted_seconds = [
        item["predicted_iteration_time_sec"] for item in iteration_results
    ]
    summary = {
        "count": len(predicted_seconds),
        "mean_sec": mean(predicted_seconds) if predicted_seconds else 0.0,
        "mean_ms": (mean(predicted_seconds) * 1000.0) if predicted_seconds else 0.0,
        "sec_percentiles": _percentiles(predicted_seconds),
        "ms_percentiles": _percentiles([value * 1000.0 for value in predicted_seconds]),
    }
    return {
        "profile_path": str(profile_path),
        "configs": configs,
        "method": method_normalized,
        "num_stages": resolved_num_stages,
        "split_backward": resolved_split_backward,
        "ratio_source": ratio_config["kind"],
        "ratio": ratio_config["ratio"],
        "backward_split_ratio": backward_split_ratio if resolved_split_backward else None,
        "comm_time": comm_time,
        "fitted_comm_time": fitted_comm_time,
        "fitted_overhead": fitted_overhead,
        "fit_iters": fit_iters if fit_comm_time else None,
        "use_actual_bwd": use_actual_bwd,
        "fit_comm_uses_copied_original_times": (
            bool(fit_comm_time) and profile.get("copied_original_until_iteration") is not None
        ),
        "copied_original_until_iteration": profile.get("copied_original_until_iteration"),
        "max_act": max_act,
        "drop_warmup": requested_drop_warmup,
        "requested_drop_warmup": requested_drop_warmup,
        "prediction_start_iteration": prediction_start_iteration,
        "fit_comm_start_iteration": requested_drop_warmup if fit_comm_time else None,
        "fit_comm_end_iteration": (
            requested_drop_warmup + fit_iters - 1 if fit_comm_time else None
        ),
        "profile_prediction_start_iteration": profile.get("prediction_start_iteration"),
        "include_weight_update": include_weight_update,
        "weight_update_time_sec": weight_update_time if include_weight_update else 0.0,
        "summary": summary,
        "iterations": iteration_results,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Feed every iteration's fwd_times from a profile JSON into "
            "microbatch_schedule_generator.py and predict each iteration's pipeline time."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/ma/Downloads/for_fit_model.json",
        help="Path to profile JSON containing fwd_times and f_b_ratio or f_b_w_ratio.",
    )
    parser.add_argument(
        "--output",
        default="iteration_time_predictions.json",
        help="Where to write iteration-level prediction JSON.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        help=(
            "Scheduling method: auto, 1f1b, or zbh. auto uses f_b_w_ratio -> zbh, "
            "f_b_ratio -> 1f1b, then falls back to configs."
        ),
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        help="Pipeline stage count. Defaults to PPSIZE parsed from profile configs.",
    )
    parser.add_argument(
        "--split-backward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to pass B/W separately to the scheduler. Defaults to true for "
            "ZBH or f_b_w_ratio, false for 1F1B with f_b_ratio."
        ),
    )
    parser.add_argument(
        "--f-b-ratio",
        help="Override profile f_b_ratio as JSON, e.g. '[1.0, 1.871]'.",
    )
    parser.add_argument(
        "--f-b-w-ratio",
        help="Override profile f_b_w_ratio as JSON, e.g. '[1.0, 1.2, 0.6]'.",
    )
    parser.add_argument(
        "--backward-split-ratio",
        type=float,
        default=1.0,
        help=(
            "Only used with f_b_ratio and split backward. "
            "B = ratio * aggregate_backward and W = (1-ratio) * aggregate_backward."
        ),
    )
    parser.add_argument("--comm-time", type=float, default=0.0, help="Uniform comm time in seconds.")
    parser.add_argument("--max-act", type=int, default=1, help="ZBH warmup MAX_ACT parameter.")
    parser.add_argument(
        "--drop-warmup",
        type=int,
        default=0,
        help="Skip the first N profile iterations before predicting.",
    )
    parser.add_argument(
        "--include-weight-update",
        action="store_true",
        help="Add avg_weight_update from the profile to each predicted iteration time.",
    )
    parser.add_argument(
        "--include-full-schedules",
        action="store_true",
        help="Store the full per-operation schedule for every iteration. This can create a large JSON.",
    )
    parser.add_argument(
        "--actual",
        help=(
            "Optional measured iteration JSON containing iter_time. When provided, "
            "the script also compares predictions with measured iter_time."
        ),
    )
    parser.add_argument(
        "--comparison-output",
        help=(
            "Where to write comparison JSON. Defaults to '<output stem>_comparison.json' "
            "when --actual is provided."
        ),
    )
    parser.add_argument(
        "--no-align-by-iteration",
        action="store_true",
        help="Compare by list position instead of prediction iteration id when --actual is provided.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of per-iteration comparison rows to print when --actual is provided.",
    )
    parser.add_argument(
        "--abs-error-pct-threshold",
        type=float,
        help=(
            "When --actual is provided, filter iterations whose absolute prediction "
            "error percentage is at least this threshold."
        ),
    )
    parser.add_argument(
        "--exclude-metric-iterations",
        help=(
            "Comma-separated or JSON list of known noisy iterations to exclude from "
            "aggregate accuracy metrics. Per-iteration rows and abs-error filtering "
            "still use the full data."
        ),
    )
    parser.add_argument(
        "--fit-comm-time",
        action="store_true",
        help=(
            "Fit comm_time from the first N iterations (see --fit-iters) by minimizing "
            "MAE against measured iter_time. Requires --actual."
        ),
    )
    parser.add_argument(
        "--fit-iters",
        type=int,
        default=50,
        help="Number of post-warmup iterations used to fit comm_time (default: 50).",
    )
    parser.add_argument(
        "--use-actual-bwd",
        action="store_true",
        help="Use actual bwd_times from the profile instead of deriving from f_b_ratio.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    method = None if args.method == "auto" else args.method
    f_b_ratio = json.loads(args.f_b_ratio) if args.f_b_ratio else None
    f_b_w_ratio = json.loads(args.f_b_w_ratio) if args.f_b_w_ratio else None
    metric_exclude_iterations = parse_iteration_list(args.exclude_metric_iterations)

    actual_iter_times = None
    if args.fit_comm_time:
        if not args.actual:
            parser.error("--fit-comm-time requires --actual.")
        with open(args.actual, "r", encoding="utf-8") as fh:
            actual_iter_times = json.load(fh)["iter_time"]

    result = predict_iteration_times_from_profile(
        args.input,
        method=method,
        num_stages=args.num_stages,
        split_backward=args.split_backward,
        f_b_ratio=f_b_ratio,
        f_b_w_ratio=f_b_w_ratio,
        backward_split_ratio=args.backward_split_ratio,
        comm_time=args.comm_time,
        max_act=args.max_act,
        drop_warmup=args.drop_warmup,
        include_weight_update=args.include_weight_update,
        include_full_schedules=args.include_full_schedules,
        fit_comm_time=args.fit_comm_time,
        fit_iters=args.fit_iters,
        actual_iter_times=actual_iter_times,
        use_actual_bwd=args.use_actual_bwd,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    summary = result["summary"]
    print(
        f"Predicted {summary['count']} iterations with method={result['method']}, "
        f"num_stages={result['num_stages']}, split_backward={result['split_backward']}."
    )
    if result["prediction_start_iteration"] != result["requested_drop_warmup"]:
        print(
            f"Prediction starts at iteration {result['prediction_start_iteration']} "
            f"(requested --drop-warmup {result['requested_drop_warmup']}, "
            f"profile prediction_start_iteration={result['profile_prediction_start_iteration']})."
        )
    if result.get("fitted_comm_time") is not None:
        print(
            f"Fitted comm_time={result['fitted_comm_time']:.6f}s, "
            f"overhead={result['fitted_overhead']:.6f}s "
            f"from iterations {result['fit_comm_start_iteration']}-"
            f"{result['fit_comm_end_iteration']}."
        )
        if result.get("fit_comm_uses_copied_original_times"):
            print(
                "Comm fitting uses copied original micro-batch times through "
                f"iteration {result['copied_original_until_iteration']}."
            )
    print(f"Mean predicted iteration time: {summary['mean_ms']:.3f} ms")
    percentiles = summary["ms_percentiles"]
    print(
        "Predicted iteration time percentiles: "
        f"min={percentiles['min']:.3f} ms, "
        f"p50={percentiles['p50']:.3f} ms, "
        f"p90={percentiles['p90']:.3f} ms, "
        f"p95={percentiles['p95']:.3f} ms, "
        f"max={percentiles['max']:.3f} ms"
    )
    print(f"Saved predictions to {output_path}")

    if args.actual:
        comparison_output_path = (
            Path(args.comparison_output)
            if args.comparison_output
            else output_path.with_name(f"{output_path.stem}_comparison{output_path.suffix}")
        )
        comparison = compare_iteration_predictions(
            output_path,
            args.actual,
            align_by_iteration=not args.no_align_by_iteration,
            abs_error_pct_threshold=args.abs_error_pct_threshold,
            metric_exclude_iterations=metric_exclude_iterations,
        )
        comparison_output_path.parent.mkdir(parents=True, exist_ok=True)
        with comparison_output_path.open("w", encoding="utf-8") as file:
            json.dump(comparison, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(
            f"Compared {comparison['prediction_count']} predictions with "
            f"{comparison['actual_iter_time_count']} measured iter_time values "
            f"(align_by_iteration={comparison['align_by_iteration']})."
        )
        for target_name, payload in comparison["targets"].items():
            print_iteration_preview(target_name, payload, args.preview_rows)
            if payload.get("metric_exclusion"):
                exclusion = payload["metric_exclusion"]
                print(
                    f"{target_name} metrics excluded {exclusion['excluded_count']} "
                    f"known noisy rows; remaining={exclusion['remaining_count']}."
                )
            print_comparison_metric_line(target_name, payload["metrics"])
            print_abs_error_filter(target_name, payload, args.preview_rows)
        print(f"Saved comparison to {comparison_output_path}")


if __name__ == "__main__":
    main()
