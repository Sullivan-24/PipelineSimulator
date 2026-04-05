import random
from typing import List, Dict

def compute_variance(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    mean = sum(values) / n
    return sum((v - mean) ** 2 for v in values) / n

def greedy_ordered_partition(times: List[float],
                             mems: List[float],
                             D: int,
                             mem_limits: List[float]) -> Dict:
    """
    顺序约束下的初始划分
    """
    L = len(times)
    T_sum = sum(times)
    target = T_sum / D

    assignments = []
    loads = []
    mem_used = []

    start = 0
    for d in range(D):
        remaining = D - d
        if d == D - 1:
            end = L
        else:
            acc_t, acc_m = 0.0, 0.0
            end = start
            while end < L and acc_t + times[end] <= target * 1.05:
                if acc_m + mems[end] > mem_limits[d]:
                    break
                acc_t += times[end]
                acc_m += mems[end]
                end += 1
            if end == start:
                raise ValueError(f"machine {d} cannot fit even one layer due to memory limit")

        seg = list(range(start, end))
        t_load = sum(times[i] for i in seg)
        m_load = sum(mems[i] for i in seg)
        if m_load > mem_limits[d]:
            raise ValueError(f"Segment {d} exceeds memory limit")
        assignments.append(seg)
        loads.append(t_load)
        mem_used.append(m_load)
        start = end

    var = compute_variance(loads)
    assignments = sort_lists(assignments)
    return {
        "assignments": assignments,
        "loads": loads,
        "mem_used": mem_used,
        "variance": var
    }

def local_adjust(assignments: List[List[int]],
                 times: List[float],
                 mems: List[float],
                 mem_limits: List[float],
                 max_iters: int = 200) -> Dict:
    """
    相邻机器之间的局部调整
    - 只能移动边界层
    """
    D = len(assignments)
    loads = [sum(times[i] for i in seg) for seg in assignments]
    mem_used = [sum(mems[i] for i in seg) for seg in assignments]

    best_var = compute_variance(loads)
    improved = True
    it = 0

    while improved and it < max_iters:
        improved = False
        it += 1

        for d in range(D - 1):
            left, right = assignments[d], assignments[d+1]
            if not left or not right:
                continue

            # 尝试把左边最后一个移到右边
            li = left[-1]
            new_mem_left = mem_used[d] - mems[li]
            new_mem_right = mem_used[d+1] + mems[li]
            if new_mem_left <= mem_limits[d] and new_mem_right <= mem_limits[d+1]:
                new_loads = loads[:]
                new_loads[d] -= times[li]
                new_loads[d+1] += times[li]
                new_var = compute_variance(new_loads)
                if new_var + 1e-12 < best_var:
                    # 接受移动
                    left.pop()
                    right.insert(0, li)
                    loads = new_loads
                    mem_used[d] = new_mem_left
                    mem_used[d+1] = new_mem_right
                    best_var = new_var
                    improved = True
                    continue

            # 尝试把右边第一个移到左边
            ri = right[0]
            new_mem_left = mem_used[d] + mems[ri]
            new_mem_right = mem_used[d+1] - mems[ri]
            if new_mem_left <= mem_limits[d] and new_mem_right <= mem_limits[d+1]:
                new_loads = loads[:]
                new_loads[d] += times[ri]
                new_loads[d+1] -= times[ri]
                new_var = compute_variance(new_loads)
                if new_var + 1e-12 < best_var:
                    right.pop(0)
                    left.append(ri)
                    loads = new_loads
                    mem_used[d] = new_mem_left
                    mem_used[d+1] = new_mem_right
                    best_var = new_var
                    improved = True
                    continue

    assignments = sort_lists(assignments)
    return {
        "assignments": assignments,
        "loads": loads,
        "mem_used": mem_used,
        "variance": best_var
    }

def solve_ordered(times: List[float],
                  mems: List[float],
                  D: int,
                  mem_limits: List[float],
                  max_iters=800) -> Dict:
    init = greedy_ordered_partition(times, mems, D, mem_limits)
    improved = local_adjust(init["assignments"], times, mems, mem_limits, max_iters=max_iters)
    return improved

def sort_lists(lists:list):
    return sorted([sorted(arr) for arr in lists], key=lambda x : x[0])

if __name__ == "__main__":
    random.seed(1)
    L = 128
    D = 16
    times = [random.uniform(10, 20) for _ in range(L)]
    mems = [random.uniform(1, 5) for _ in range(L)]
    mem_limits = [144 for _ in range(D)]
    
    result = solve_ordered(times, mems, D, mem_limits, max_iters=800)

    for d in range(D):
        print(f"machine {d}: layers {result['assignments'][d]} load {result['loads'][d]:.2f} mem {result['mem_used'][d]:.2f}/{mem_limits[d]:.2f}")
    print("Final variance:", result["variance"])
