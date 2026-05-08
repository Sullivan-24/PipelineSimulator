from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from fit_forward_time_model import _flatten_seq_lengths
from fit_forward_time_model import predict_forward_time_ms
from microbatch_schedule_generator import generate_microbatch_schedule
from predict_iteration_time_from_profile import _build_microbatch_times
from predict_iteration_time_from_profile import _infer_method_from_profile
from predict_iteration_time_from_profile import _normalize_method
from predict_iteration_time_from_profile import _parse_int_from_configs
from predict_iteration_time_from_profile import _resolve_ratio_config


DEFAULT_NAME = "13bllama_1f1b_l40_hid5120_seq16384_voc32000_mb8"
DEFAULT_BASE = Path("/Users/ma/Downloads") / DEFAULT_NAME / "generate"
DEFAULT_ACTUAL = Path("/Users/ma/Downloads") / f"{DEFAULT_NAME}_iter.json"
DEFAULT_FORWARD_MODEL = DEFAULT_BASE / f"{DEFAULT_NAME}_forward_time_model.json"
DEFAULT_OUTPUT = DEFAULT_BASE / f"{DEFAULT_NAME}_overhead_benchmark.json"
DEFAULT_BOCD_DIR = Path("/Users/ma/code/Greyhound/detector/control_plane")


def _load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _summarize_durations(
    durations_sec: Sequence[float],
    *,
    denominator: int,
) -> Dict[str, float]:
    per_item = [duration / denominator for duration in durations_sec]
    return {
        "runs": len(durations_sec),
        "denominator": denominator,
        "mean_total_sec": statistics.mean(durations_sec),
        "median_total_sec": statistics.median(durations_sec),
        "mean_per_item_sec": statistics.mean(per_item),
        "median_per_item_sec": statistics.median(per_item),
        "mean_per_item_ms": statistics.mean(per_item) * 1000.0,
        "median_per_item_ms": statistics.median(per_item) * 1000.0,
        "mean_per_item_us": statistics.mean(per_item) * 1_000_000.0,
        "median_per_item_us": statistics.median(per_item) * 1_000_000.0,
    }


def _benchmark(
    fn: Callable[[], Any],
    *,
    repeat: int,
    warmup_runs: int,
    denominator: int,
) -> Dict[str, Any]:
    for _ in range(warmup_runs):
        fn()
    durations = []
    last_result = None
    for _ in range(repeat):
        started = time.perf_counter()
        last_result = fn()
        durations.append(time.perf_counter() - started)
    summary = _summarize_durations(durations, denominator=denominator)
    summary["durations_sec"] = durations
    return {"summary": summary, "last_result": last_result}


def _resolve_ratio(profile: Dict[str, Any]) -> Dict[str, float]:
    if "f_b_w_ratio" in profile:
        ratio = [float(value) for value in profile["f_b_w_ratio"]]
        return {
            "kind": "f_b_w_ratio",
            "b_factor": ratio[1] / ratio[0],
            "w_factor": ratio[2] / ratio[0],
        }
    ratio = [float(value) for value in profile["f_b_ratio"]]
    return {
        "kind": "f_b_ratio",
        "b_factor": ratio[1] / ratio[0],
        "w_factor": 0.0,
    }


def _benchmark_microbatch_prediction(
    forward_model: Dict[str, Any],
    iteration_ids: Sequence[int],
) -> Dict[str, Any]:
    ratio = _resolve_ratio(forward_model)
    predicted_iterations = []
    microbatch_count = 0
    for iteration in iteration_ids:
        fwd_times = []
        bwd_times = []
        wwd_times = []
        for microbatch in forward_model["seq_info"][iteration]:
            seq_lengths = _flatten_seq_lengths(microbatch["seq_lengths"])
            fwd_time = predict_forward_time_ms(seq_lengths, forward_model) / 1000.0
            fwd_times.append(fwd_time)
            bwd_times.append(fwd_time * ratio["b_factor"])
            wwd_times.append(fwd_time * ratio["w_factor"])
            microbatch_count += 1
        predicted_iterations.append(
            {
                "iteration": iteration,
                "fwd_times": fwd_times,
                "bwd_times": bwd_times,
                "wwd_times": wwd_times,
            }
        )
    return {
        "iteration_count": len(predicted_iterations),
        "microbatch_count": microbatch_count,
        "last_iteration": predicted_iterations[-1] if predicted_iterations else None,
    }


def _benchmark_iteration_time_prediction(
    forward_model: Dict[str, Any],
    iteration_ids: Sequence[int],
    *,
    comm_time: float,
    max_act: int,
) -> Dict[str, Any]:
    ratio_config = _resolve_ratio_config(forward_model, f_b_ratio=None, f_b_w_ratio=None)
    method = _normalize_method(_infer_method_from_profile(forward_model))
    split_backward = method == "zbh" or ratio_config["kind"] == "f_b_w_ratio"
    num_stages = _parse_int_from_configs(str(forward_model.get("configs", "")), "PPSIZE")
    if num_stages is None:
        raise ValueError("Cannot infer num_stages from forward_model configs.")

    predicted_times = []
    for iteration in iteration_ids:
        microbatch_times, _, _ = _build_microbatch_times(
            forward_model["fwd_times"][iteration],
            ratio_config=ratio_config,
            split_backward=split_backward,
            backward_split_ratio=1.0,
        )
        schedule = generate_microbatch_schedule(
            num_stages=num_stages,
            microbatch_times=microbatch_times,
            method=method,
            split_backward=split_backward,
            max_act=max_act,
            comm_time=comm_time,
        )
        predicted_times.append(float(schedule["makespan"]))
    return {
        "iteration_count": len(predicted_times),
        "method": method,
        "split_backward": split_backward,
        "num_stages": num_stages,
        "last_predicted_iteration_time_sec": predicted_times[-1] if predicted_times else None,
    }


def _load_bocd_module(bocd_dir: Path) -> Any:
    sys.path.insert(0, str(bocd_dir))
    import bocd_iteration_detector  # type: ignore

    return bocd_iteration_detector


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark overhead of BOCD+V, micro-batch time prediction, and iteration-time scheduling."
    )
    parser.add_argument("--actual-json", default=str(DEFAULT_ACTUAL))
    parser.add_argument("--forward-model-json", default=str(DEFAULT_FORWARD_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--bocd-dir", default=str(DEFAULT_BOCD_DIR))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Benchmark warm-up runs that are not included in timing.",
    )
    parser.add_argument("--verification-mode", default="window")
    parser.add_argument("--verification-window", type=int, default=3)
    parser.add_argument("--hazard-lambda", type=float, default=250.0)
    parser.add_argument("--reset-threshold", type=int, default=5)
    parser.add_argument("--length-thresh", type=int, default=10)
    parser.add_argument("--degradation-thresh", type=float, default=0.1)
    parser.add_argument("--bocd-warmup", type=int, default=50)
    parser.add_argument("--max-act", type=int, default=1)
    parser.add_argument("--comm-time", type=float, default=0.0)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    forward_model = _load_json(args.forward_model_json)
    actual_payload = _load_json(args.actual_json)
    values = [float(value) for value in actual_payload["iter_time"]]
    prediction_start = int(forward_model.get("prediction_start_iteration", 0))
    iteration_ids = list(range(prediction_start, len(forward_model["fwd_times"])))
    if not iteration_ids:
        raise ValueError("No prediction iterations found.")

    bocd_module = _load_bocd_module(Path(args.bocd_dir))

    def run_bocd() -> Any:
        return bocd_module.run_detection(
            values=values,
            hazard_lambda=args.hazard_lambda,
            reset_threshold=args.reset_threshold,
            length_thresh=args.length_thresh,
            verification_mode=args.verification_mode,
            verification_window=args.verification_window,
            degradation_thresh=args.degradation_thresh,
            warmup=args.bocd_warmup,
        )

    bocd_result = _benchmark(
        run_bocd,
        repeat=args.repeat,
        warmup_runs=args.warmup_runs,
        denominator=max(len(values) - args.bocd_warmup, 1),
    )
    microbatch_result = _benchmark(
        lambda: _benchmark_microbatch_prediction(forward_model, iteration_ids),
        repeat=args.repeat,
        warmup_runs=args.warmup_runs,
        denominator=len(iteration_ids),
    )
    scheduler_result = _benchmark(
        lambda: _benchmark_iteration_time_prediction(
            forward_model,
            iteration_ids,
            comm_time=args.comm_time,
            max_act=args.max_act,
        ),
        repeat=args.repeat,
        warmup_runs=args.warmup_runs,
        denominator=len(iteration_ids),
    )

    result = {
        "inputs": {
            "actual_json": str(args.actual_json),
            "forward_model_json": str(args.forward_model_json),
            "repeat": args.repeat,
            "warmup_runs": args.warmup_runs,
            "prediction_start_iteration": prediction_start,
            "prediction_iteration_count": len(iteration_ids),
            "bocd_warmup": args.bocd_warmup,
            "bocd_denominator_excludes_warmup": True,
        },
        "bocd_plus_verification": {
            "summary": bocd_result["summary"],
            "event_count": len(bocd_result["last_result"]),
        },
        "microbatch_time_prediction": {
            "summary": microbatch_result["summary"],
            "last_result": microbatch_result["last_result"],
        },
        "iteration_time_prediction": {
            "summary": scheduler_result["summary"],
            "last_result": scheduler_result["last_result"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Saved overhead benchmark to {output_path}")
    print(
        "BOCD+V: "
        f"{result['bocd_plus_verification']['summary']['mean_per_item_us']:.3f} us/step"
    )
    print(
        "Micro-batch time prediction: "
        f"{result['microbatch_time_prediction']['summary']['mean_per_item_us']:.3f} "
        "us/iteration"
    )
    print(
        "Iteration time prediction: "
        f"{result['iteration_time_prediction']['summary']['mean_per_item_us']:.3f} "
        "us/iteration"
    )


if __name__ == "__main__":
    main()
