from enum import Enum
class RecomputeType(Enum):
    FULL = 1
    SELECTIVE = 2

class WorkloadType(Enum):
    F = "F"
    B = "B"
    W = "W"


class Schedule(Enum):
    GREEDY_v1 = 1
    INTERLEAVED = 2
    ONE_F_ONE_B = 3
    ZBV = 4
    ZBH1 = 5
    BFW = 6
    GREEDY_v2 = 7
    Layerwise = 8
    Chimera = 9

    STANDARD_1F1B = 10
    STANDARD_INTERLEAVED = 11
    STANDARD_ZBH1 = 12
    STANDARD_ZBV = 13

class StageSearchOrder(Enum):
    Random = "Random"
    IncDec = "IncDec"

class RunMode(Enum):
    SEARCH_SCHEDULE = "search_schedule"
    Z3_SOLVE = "z3"
    GUROBI_SOLVE = "gurobi"
    SIM_SOLVE = "sim"
    LAYERWISE_GUROBI_SOLVE = "layer"
    CHIMERA = "chimera"

class WorkloadConstraint:

    def __init__(self, 
                 device_id: int,
                 microbatch_id: int, 
                 stage_id: int, 
                 workload_type: WorkloadType,
                ) -> None:
        self.device_id = device_id
        self.microbatch_id: int = microbatch_id  # 微批次编号
        self.stage_id: int = stage_id              # 阶段编号
        self.workload_type: WorkloadType = workload_type  # 工作负载类型

    def __eq__(self, other):
        if not isinstance(other, WorkloadConstraint):
            return NotImplemented
        return (
            self.microbatch_id == other.microbatch_id 
            and self.stage_id == other.stage_id
            and self.workload_type == other.workload_type
        )
    
    def __hash__(self):
        return hash((self.microbatch_id, self.stage_id, self.workload_type))
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(device_id={self.device_id}, "
            f"microbatch_id={self.microbatch_id}, stage_id={self.stage_id}, "
            f"workload_type={self.workload_type})")