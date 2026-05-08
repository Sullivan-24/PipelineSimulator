from __future__ import annotations

"""
Input JSON example:
[
  {"f": 4, "b": 6, "w": 2},
  {"f": [4, 4, 5, 5], "b": [6, 6, 7, 7], "w": [2, 2, 3, 3]}
]
"""

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


VALID_METHODS = {"1f1b", "zbh"}


@dataclass(frozen=True)
class OperationKey:
    workload_type: str
    microbatch_id: int
    stage_id: int


@dataclass(frozen=True)
class ScheduledOperation:
    workload_type: str
    microbatch_id: int
    stage_id: int
    device_id: int
    duration: float
    start_time: float
    end_time: float
    order_index: int


def _normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "").replace("_", "")
    if normalized not in VALID_METHODS:
        raise ValueError(f"Unsupported method: {method}. Expected one of {sorted(VALID_METHODS)}.")
    return normalized


def _resolve_stage_value(
    value: Any,
    stage_id: int,
    num_stages: int,
    field_name: str,
) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != num_stages:
            raise ValueError(
                f"{field_name} expects {num_stages} stage values, got {len(value)}."
            )
        return float(value[stage_id])
    if isinstance(value, Mapping):
        if stage_id in value:
            return float(value[stage_id])
        stage_key = str(stage_id)
        if stage_key in value:
            return float(value[stage_key])
        if "default" in value:
            return float(value["default"])
    raise ValueError(
        f"{field_name} must be a scalar, a stage-length list, or a stage-id mapping."
    )


def _normalize_microbatch_times(
    microbatch_times: Sequence[Mapping[str, Any]],
    num_stages: int,
    split_backward: bool,
) -> List[Dict[str, List[float]]]:
    normalized: List[Dict[str, List[float]]] = []
    required_fields = ("f", "b", "w") if split_backward else ("f", "b")
    for mid, microbatch_time in enumerate(microbatch_times):
        if not isinstance(microbatch_time, Mapping):
            raise ValueError(
                f"microbatch_times[{mid}] must be a mapping like {{'f': 4, 'b': 6, 'w': 3}}."
            )
        normalized_item: Dict[str, List[float]] = {}
        for workload_type in required_fields:
            if workload_type not in microbatch_time:
                raise ValueError(
                    f"microbatch_times[{mid}] is missing required field '{workload_type}'."
                )
            normalized_item[workload_type] = [
                _resolve_stage_value(
                    microbatch_time[workload_type],
                    stage_id=stage_id,
                    num_stages=num_stages,
                    field_name=f"microbatch_times[{mid}]['{workload_type}']",
                )
                for stage_id in range(num_stages)
            ]
        normalized.append(normalized_item)
    return normalized


def _comm_delay(src_stage: int, dst_stage: int, comm_time: Any) -> float:
    if src_stage == dst_stage:
        return 0.0
    if isinstance(comm_time, (int, float)):
        return float(comm_time)
    if isinstance(comm_time, Sequence) and not isinstance(comm_time, (str, bytes)):
        return float(comm_time[src_stage][dst_stage])
    raise ValueError("comm_time must be a scalar or a stage-to-stage matrix.")


def build_static_schedule(
    num_stages: int,
    num_microbatches: int,
    method: str,
    *,
    split_backward: bool = True,
    max_act: int = 1,
) -> List[List[Tuple[str, int, int]]]:
    method = _normalize_method(method)
    if method == "zbh" and not split_backward:
        raise ValueError("ZBH requires split_backward=True.")

    schedule: List[List[Tuple[str, int, int]]] = [[] for _ in range(num_stages)]
    for stage_id in range(num_stages):
        if method == "1f1b":
            workload_order = ["b", "w", "f"] if split_backward else ["b", "f"]
            workload_index = {workload_type: idx for idx, workload_type in enumerate(("f", "b", "w"))}
            active_slots = len(workload_order)
            mids = [0] * 3

            warmup_forward = min(num_stages - stage_id, num_microbatches)
            while mids[workload_index["f"]] < warmup_forward:
                schedule[stage_id].append(("f", mids[workload_index["f"]], stage_id))
                mids[workload_index["f"]] += 1

            finished = [False] * active_slots
            cursor = 0
            while not all(finished):
                next_workload = workload_order[cursor % active_slots]
                idx = workload_index[next_workload]
                next_mid = mids[idx]
                if next_mid < num_microbatches:
                    schedule[stage_id].append((next_workload, next_mid, stage_id))
                    mids[idx] += 1
                else:
                    finished[cursor % active_slots] = True
                cursor += 1
        else:
            workload_order = ["b", "w", "f"]
            workload_index = {"f": 0, "b": 1, "w": 2}
            mids = [0, 0, 0]

            warmup_forward = min(num_microbatches, (num_stages - stage_id - 1) * max_act + 1)
            while mids[workload_index["f"]] < warmup_forward:
                schedule[stage_id].append(("f", mids[workload_index["f"]], stage_id))
                mids[workload_index["f"]] += 1

            finished = [False, False, False]
            cursor = 0
            while not all(finished):
                next_workload = workload_order[cursor % len(workload_order)]
                next_mid = mids[workload_index[next_workload]]
                if mids[workload_index["f"]] < min(num_microbatches, num_stages * max_act):
                    if next_workload == "w":
                        cursor += 1
                        continue
                if next_mid < num_microbatches:
                    schedule[stage_id].append((next_workload, next_mid, stage_id))
                    mids[workload_index[next_workload]] += 1
                else:
                    finished[workload_index[next_workload]] = True
                cursor += 1
    return schedule


def _duration_for(
    normalized_microbatch_times: Sequence[Dict[str, List[float]]],
    workload_type: str,
    microbatch_id: int,
    stage_id: int,
) -> float:
    return normalized_microbatch_times[microbatch_id][workload_type][stage_id]


def _build_predecessor_edges(
    static_schedule: Sequence[Sequence[Tuple[str, int, int]]],
    normalized_microbatch_times: Sequence[Dict[str, List[float]]],
    method: str,
    split_backward: bool,
    comm_time: Any,
) -> Tuple[
    Dict[OperationKey, Dict[OperationKey, float]],
    Dict[OperationKey, float],
    Dict[OperationKey, int],
]:
    method = _normalize_method(method)
    node_duration: Dict[OperationKey, float] = {}
    node_order: Dict[OperationKey, int] = {}
    predecessors: Dict[OperationKey, Dict[OperationKey, float]] = {}
    for stage_id, stage_schedule in enumerate(static_schedule):
        for order_index, (workload_type, microbatch_id, _) in enumerate(stage_schedule):
            key = OperationKey(workload_type, microbatch_id, stage_id)
            node_duration[key] = _duration_for(
                normalized_microbatch_times,
                workload_type=workload_type,
                microbatch_id=microbatch_id,
                stage_id=stage_id,
            )
            node_order[key] = order_index
            predecessors[key] = {}

    def add_edge(
        before: OperationKey,
        after: OperationKey,
        lag: float,
    ) -> None:
        if before not in node_duration or after not in node_duration:
            raise KeyError(f"Missing node when adding dependency {before} -> {after}.")
        predecessors[after][before] = max(predecessors[after].get(before, 0.0), lag)

    for stage_id, stage_schedule in enumerate(static_schedule):
        previous_key: OperationKey | None = None
        for workload_type, microbatch_id, _ in stage_schedule:
            key = OperationKey(workload_type, microbatch_id, stage_id)
            if previous_key is not None:
                add_edge(previous_key, key, node_duration[previous_key])
            previous_key = key

    last_stage_id = len(static_schedule) - 1
    backward_dependency_type = "w" if method == "1f1b" and split_backward else "b"
    for key in node_duration:
        if key.workload_type == "f":
            if key.stage_id == 0:
                continue
            prev_key = OperationKey("f", key.microbatch_id, key.stage_id - 1)
            add_edge(
                prev_key,
                key,
                node_duration[prev_key] + _comm_delay(key.stage_id - 1, key.stage_id, comm_time),
            )
        elif key.workload_type == "b":
            if key.stage_id == last_stage_id:
                prev_key = OperationKey("f", key.microbatch_id, key.stage_id)
                add_edge(prev_key, key, node_duration[prev_key])
            else:
                prev_key = OperationKey(backward_dependency_type, key.microbatch_id, key.stage_id + 1)
                add_edge(
                    prev_key,
                    key,
                    node_duration[prev_key] + _comm_delay(key.stage_id + 1, key.stage_id, comm_time),
                )
        elif key.workload_type == "w":
            prev_key = OperationKey("b", key.microbatch_id, key.stage_id)
            add_edge(prev_key, key, node_duration[prev_key])

    return predecessors, node_duration, node_order


def _topological_order(
    predecessors: Mapping[OperationKey, Mapping[OperationKey, float]],
    node_order: Mapping[OperationKey, int],
) -> List[OperationKey]:
    outgoing: Dict[OperationKey, List[OperationKey]] = {node: [] for node in predecessors}
    indegree: Dict[OperationKey, int] = {}
    for node, incoming in predecessors.items():
        indegree[node] = len(incoming)
        for parent in incoming:
            outgoing[parent].append(node)

    queue = deque(
        sorted(
            (node for node, degree in indegree.items() if degree == 0),
            key=lambda item: (item.stage_id, node_order[item], item.microbatch_id, item.workload_type),
        )
    )
    order: List[OperationKey] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        released: List[OperationKey] = []
        for child in outgoing[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                released.append(child)
        for child in sorted(
            released,
            key=lambda item: (item.stage_id, node_order[item], item.microbatch_id, item.workload_type),
        ):
            queue.append(child)

    if len(order) != len(predecessors):
        raise RuntimeError("Dependency graph contains a cycle.")
    return order


def generate_microbatch_schedule(
    num_stages: int,
    microbatch_times: Sequence[Mapping[str, Any]],
    method: str,
    *,
    dp_size: int = 1,
    split_backward: bool = True,
    max_act: int = 1,
    comm_time: Any = 0,
) -> Dict[str, Any]:
    if dp_size != 1:
        raise ValueError("This generator only supports dp_size=1.")
    if num_stages <= 0:
        raise ValueError("num_stages must be positive.")
    if not microbatch_times:
        raise ValueError("microbatch_times must not be empty.")

    method = _normalize_method(method)
    normalized_microbatch_times = _normalize_microbatch_times(
        microbatch_times=microbatch_times,
        num_stages=num_stages,
        split_backward=split_backward,
    )
    static_schedule = build_static_schedule(
        num_stages=num_stages,
        num_microbatches=len(normalized_microbatch_times),
        method=method,
        split_backward=split_backward,
        max_act=max_act,
    )
    predecessors, node_duration, node_order = _build_predecessor_edges(
        static_schedule=static_schedule,
        normalized_microbatch_times=normalized_microbatch_times,
        method=method,
        split_backward=split_backward,
        comm_time=comm_time,
    )
    topo_order = _topological_order(predecessors=predecessors, node_order=node_order)

    start_time: Dict[OperationKey, float] = {}
    scheduled_ops: List[ScheduledOperation] = []
    for key in topo_order:
        if predecessors[key]:
            start = max(start_time[parent] + lag for parent, lag in predecessors[key].items())
        else:
            start = 0.0
        end = start + node_duration[key]
        start_time[key] = start
        scheduled_ops.append(
            ScheduledOperation(
                workload_type=key.workload_type,
                microbatch_id=key.microbatch_id,
                stage_id=key.stage_id,
                device_id=key.stage_id,
                duration=node_duration[key],
                start_time=start,
                end_time=end,
                order_index=node_order[key],
            )
        )

    scheduled_ops.sort(key=lambda op: (op.start_time, op.stage_id, op.order_index))
    per_stage: List[List[Dict[str, Any]]] = [[] for _ in range(num_stages)]
    results: Dict[str, float] = {}
    for scheduled_op in scheduled_ops:
        payload = asdict(scheduled_op)
        per_stage[scheduled_op.stage_id].append(payload)
        results[
            f"{scheduled_op.workload_type}_{scheduled_op.microbatch_id}_{scheduled_op.stage_id}_{scheduled_op.device_id}"
        ] = scheduled_op.start_time

    makespan = max((op.end_time for op in scheduled_ops), default=0.0)
    return {
        "method": method,
        "dp_size": dp_size,
        "num_stages": num_stages,
        "num_microbatches": len(normalized_microbatch_times),
        "split_backward": split_backward,
        "max_act": max_act,
        "comm_time": comm_time,
        "static_schedule": static_schedule,
        "operations": [asdict(op) for op in scheduled_ops],
        "per_stage": per_stage,
        "results": results,
        "makespan": makespan,
    }


def _load_microbatch_times(input_path: str | Path) -> Any:
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DP=1 pipeline schedule timeline from per-microbatch durations "
            "using fixed 1F1B or ZBH ordering."
        )
    )
    parser.add_argument("--method", required=True, help="Scheduling method: 1f1b or zbh.")
    parser.add_argument("--num-stages", required=True, type=int, help="Pipeline stage count.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file containing microbatch_times.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "--split-backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether backward is split into B and W. ZBH requires this to stay enabled.",
    )
    parser.add_argument(
        "--max-act",
        type=int,
        default=1,
        help="ZBH warmup parameter. Ignored by 1F1B.",
    )
    parser.add_argument(
        "--comm-time",
        type=float,
        default=0.0,
        help="Uniform communication delay between adjacent stages.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    microbatch_times = _load_microbatch_times(args.input)
    result = generate_microbatch_schedule(
        num_stages=args.num_stages,
        microbatch_times=microbatch_times,
        method=args.method,
        split_backward=args.split_backward,
        max_act=args.max_act,
        comm_time=args.comm_time,
    )
    content = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            file.write(content)
            file.write("\n")
    else:
        print(content)


if __name__ == "__main__":
    main()
