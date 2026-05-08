from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_NAME = "1f1b_l28_hid3584_seq8192_voc152064_mb4"
DEFAULT_DOWNLOADS = Path("/Users/ma/code/PipelineSimulator/predict_tmp/ss")
DEFAULT_REPO = Path("/Users/ma/code/PipelineSimulator")
DEFAULT_OUTPUT_DIR = DEFAULT_DOWNLOADS / DEFAULT_NAME / "generate"
DEFAULT_BOCD_SCRIPT = Path(
    "/Users/ma/code/Greyhound/detector/control_plane/bocd_iteration_detector.py"
)
DEFAULT_BENCHMARK_SCRIPT = DEFAULT_REPO / "benchmark_prediction_overhead.py"


def _comma_join(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _parse_value(raw_value: str) -> Any:
    value = raw_value.rstrip("%")
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return raw_value


def _parse_bocd_stdout(stdout: str) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if "index" not in fields or "verdict" not in fields:
            continue
        event = {key: _parse_value(value) for key, value in fields.items()}
        if "degradation" in event:
            event["degradation_pct"] = event.pop("degradation")
        events.append(event)

    change_points = [
        int(event["index"]) for event in events if event.get("verdict") == "change_point"
    ]
    jitters = [int(event["index"]) for event in events if event.get("verdict") == "jitter"]
    noisy_iterations = sorted(set(change_points + jitters))
    return {
        "events": events,
        "change_point_iterations": change_points,
        "jitter_iterations": jitters,
        "exclude_metric_iterations": noisy_iterations,
    }


def _run_step(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
) -> Dict[str, Any]:
    started = time.time()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    ended = time.time()
    return {
        "name": name,
        "command": list(command),
        "cwd": str(cwd),
        "returncode": completed.returncode,
        "duration_sec": ended - started,
        "stdout_lines": completed.stdout.splitlines(),
        "stderr_lines": completed.stderr.splitlines(),
    }


def _write_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _summarize_benchmark(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    return {
        "output": str(path),
        "inputs": payload.get("inputs", {}),
        "bocd_plus_verification": payload.get("bocd_plus_verification", {}).get(
            "summary", {}
        ),
        "microbatch_time_prediction": payload.get("microbatch_time_prediction", {}).get(
            "summary", {}
        ),
        "iteration_time_prediction": payload.get("iteration_time_prediction", {}).get(
            "summary", {}
        ),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BOCD detection, forward-time fitting, iteration-time prediction, "
            "and overhead benchmarking as one end-to-end pipeline. Subprocess "
            "stdout/stderr lines are captured into the result JSON."
        )
    )
    parser.add_argument(
        "--profile-json",
        default=str(DEFAULT_DOWNLOADS / f"{DEFAULT_NAME}.json"),
        help="Original profile JSON used by fit_forward_time_model.py.",
    )
    parser.add_argument(
        "--actual-json",
        default=str(DEFAULT_DOWNLOADS / f"{DEFAULT_NAME}_iter.json"),
        help="Measured iteration JSON used by BOCD and comparison.",
    )
    parser.add_argument(
        "--forward-model-output",
        default=str(DEFAULT_OUTPUT_DIR / f"{DEFAULT_NAME}_forward_time_model.json"),
        help="Output path for fit_forward_time_model.py.",
    )
    parser.add_argument(
        "--prediction-output",
        default=str(DEFAULT_OUTPUT_DIR / f"{DEFAULT_NAME}_pre_iter.json"),
        help="Output path for predict_iteration_time_from_profile.py.",
    )
    parser.add_argument(
        "--comparison-output",
        default=str(DEFAULT_OUTPUT_DIR / f"{DEFAULT_NAME}_prediction_comparison.json"),
        help="Output path for comparison JSON.",
    )
    parser.add_argument(
        "--benchmark-output",
        default=str(DEFAULT_OUTPUT_DIR / f"{DEFAULT_NAME}_overhead_benchmark.json"),
        help="Output path for benchmark_prediction_overhead.py.",
    )
    parser.add_argument(
        "--result-output",
        default=str(DEFAULT_OUTPUT_DIR / f"{DEFAULT_NAME}_end2end_result.json"),
        help="Final JSON containing all command prints and summarized results.",
    )
    parser.add_argument("--bocd-script", default=str(DEFAULT_BOCD_SCRIPT))
    parser.add_argument(
        "--fit-script",
        default=str(DEFAULT_REPO / "fit_forward_time_model.py"),
    )
    parser.add_argument(
        "--predict-script",
        default=str(DEFAULT_REPO / "predict_iteration_time_from_profile.py"),
    )
    parser.add_argument(
        "--benchmark-script",
        default=str(DEFAULT_BENCHMARK_SCRIPT),
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable.")
    parser.add_argument("--cwd", default=str(DEFAULT_REPO), help="Working directory for commands.")
    parser.add_argument("--drop-warmup", type=int, default=2)
    parser.add_argument("--train-iterations", type=int, default=50)
    parser.add_argument("--fit-iters", type=int, default=50)
    parser.add_argument("--benchmark-repeat", type=int, default=5)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=2)
    parser.add_argument("--verification-mode", default="window")
    parser.add_argument("--verification-window", type=int, default=3)
    parser.add_argument("--abs-error-pct-threshold", type=float, default=25.0)
    parser.add_argument(
        "--extra-exclude-metric-iterations",
        default="",
        help=(
            "Optional comma-separated iterations to add to BOCD change_point+jitter "
            "when excluding aggregate accuracy metrics."
        ),
    )
    return parser


def _parse_extra_iterations(value: str) -> List[int]:
    if not value.strip():
        return []
    if value.strip().startswith("["):
        return [int(item) for item in json.loads(value)]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cwd = Path(args.cwd)
    profile_json = Path(args.profile_json)
    actual_json = Path(args.actual_json)
    forward_model_output = Path(args.forward_model_output)
    prediction_output = Path(args.prediction_output)
    comparison_output = Path(args.comparison_output)
    benchmark_output = Path(args.benchmark_output)
    result_output = Path(args.result_output)

    result: Dict[str, Any] = {
        "status": "running",
        "inputs": {
            "profile_json": str(profile_json),
            "actual_json": str(actual_json),
            "drop_warmup": args.drop_warmup,
            "train_iterations": args.train_iterations,
            "fit_iters": args.fit_iters,
            "benchmark_repeat": args.benchmark_repeat,
            "benchmark_warmup_runs": args.benchmark_warmup_runs,
            "verification_mode": args.verification_mode,
            "verification_window": args.verification_window,
            "abs_error_pct_threshold": args.abs_error_pct_threshold,
            "outputs": {
                "forward_model_output": str(forward_model_output),
                "prediction_output": str(prediction_output),
                "comparison_output": str(comparison_output),
                "benchmark_output": str(benchmark_output),
                "result_output": str(result_output),
            },
        },
        "steps": [],
    }

    bocd_command = [
        args.python,
        args.bocd_script,
        "--input-json",
        str(actual_json),
        "--verification-mode",
        args.verification_mode,
        "--verification-window",
        str(args.verification_window),
    ]
    bocd_step = _run_step("bocd_iteration_detector", bocd_command, cwd=cwd)
    result["steps"].append(bocd_step)
    bocd_result = _parse_bocd_stdout("\n".join(bocd_step["stdout_lines"]))
    if bocd_step["returncode"] != 0:
        result["status"] = "failed"
        result["failed_step"] = "bocd_iteration_detector"
        _write_result(result_output, result)
        return bocd_step["returncode"]

    exclude_metric_iterations = sorted(
        set(bocd_result["exclude_metric_iterations"])
        | set(_parse_extra_iterations(args.extra_exclude_metric_iterations))
    )
    exclude_metric_arg = _comma_join(exclude_metric_iterations)

    fit_command = [
        args.python,
        args.fit_script,
        "--input",
        str(profile_json),
        "--output",
        str(forward_model_output),
        "--drop-warmup",
        str(args.drop_warmup),
        "--train-iterations",
        str(args.train_iterations),
    ]
    if exclude_metric_arg:
        fit_command.extend(["--exclude-metric-iterations", exclude_metric_arg])
    fit_step = _run_step("fit_forward_time_model", fit_command, cwd=cwd)
    result["steps"].append(fit_step)
    if fit_step["returncode"] != 0:
        result["status"] = "failed"
        result["failed_step"] = "fit_forward_time_model"
        _write_result(result_output, result)
        return fit_step["returncode"]

    predict_command = [
        args.python,
        args.predict_script,
        "--input",
        str(forward_model_output),
        "--output",
        str(prediction_output),
        "--drop-warmup",
        str(args.drop_warmup),
        "--actual",
        str(actual_json),
        "--comparison-output",
        str(comparison_output),
        "--fit-comm-time",
        "--fit-iters",
        str(args.fit_iters),
        "--abs-error-pct-threshold",
        str(args.abs_error_pct_threshold),
    ]
    if exclude_metric_arg:
        predict_command.extend(["--exclude-metric-iterations", exclude_metric_arg])
    predict_step = _run_step("predict_iteration_time_from_profile", predict_command, cwd=cwd)
    result["steps"].append(predict_step)
    if predict_step["returncode"] != 0:
        result["status"] = "failed"
        result["failed_step"] = "predict_iteration_time_from_profile"
        _write_result(result_output, result)
        return predict_step["returncode"]

    benchmark_command = [
        args.python,
        args.benchmark_script,
        "--actual-json",
        str(actual_json),
        "--forward-model-json",
        str(forward_model_output),
        "--output",
        str(benchmark_output),
        "--repeat",
        str(args.benchmark_repeat),
        "--warmup-runs",
        str(args.benchmark_warmup_runs),
        "--verification-mode",
        args.verification_mode,
        "--verification-window",
        str(args.verification_window),
    ]
    benchmark_step = _run_step("benchmark_prediction_overhead", benchmark_command, cwd=cwd)
    result["steps"].append(benchmark_step)
    if benchmark_step["returncode"] != 0:
        result["status"] = "failed"
        result["failed_step"] = "benchmark_prediction_overhead"
        _write_result(result_output, result)
        return benchmark_step["returncode"]

    result["benchmark_overhead"] = _summarize_benchmark(benchmark_output)
    result["status"] = "succeeded"
    _write_result(result_output, result)
    print(f"Saved end-to-end result to {result_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
