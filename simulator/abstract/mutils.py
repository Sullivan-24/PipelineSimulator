from enum import Enum

G = 1024 * 1024 * 1024
M = 1024 * 1024
K = 1024
B = G

GLOBAL_TIME = 0
CHUNK_NUM = 1
DEVICE_NUM = 8

FPW_TIME = 6
FPW_IGW_RATE = 1.25
FPW_PGW_RATE = 0.75 
IGW_TIME = FPW_TIME * FPW_IGW_RATE
PGW_TIME = FPW_TIME * FPW_PGW_RATE
IGW_TIME = 10
PGW_TIME = 4
COMM_TIME = 1

class SchedulePriority(Enum):
    GREEDY = 1
    INTERLEAVED = 2
    ONE_F_ONE_B = 3
    ZBV = 4
    ZBH1 = 5
    BFW = 6

SCHEDULE_METHOD = SchedulePriority.ZBH1
if SCHEDULE_METHOD == SchedulePriority.ONE_F_ONE_B:
    SPLIT_BACKPROP = False
    CHUNK_NUM = 1
    IGW_TIME += PGW_TIME
elif SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
    SPLIT_BACKPROP = False
elif SCHEDULE_METHOD == SchedulePriority.ZBH1:
    SPLIT_BACKPROP = True
    CHUNK_NUM = 1
elif SCHEDULE_METHOD in (SchedulePriority.GREEDY, SchedulePriority.ZBV):
    SPLIT_BACKPROP = True
    
WORKLOAD_TYPE_NUM = 3 if SPLIT_BACKPROP else 2
STAGE_NUM = DEVICE_NUM * CHUNK_NUM + 0
MICRO_BATCH_NUM = DEVICE_NUM * 2 + 0
MODEL_LAYER_NUM = STAGE_NUM + 0

MODEL_PARA_SIZE = DEVICE_NUM
PP_SIZE = DEVICE_NUM
TP_SIZE = MODEL_PARA_SIZE // PP_SIZE

GPU_MAX_MEM = 80 * G
MODEL_PARA_NUM = 7 * B
MIX_TRAINING = True

MAX_ACTIVATION_COUNTS = DEVICE_NUM * CHUNK_NUM * 2
OFFLOAD_TIME = 4

def UPDATE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME+=1

def DECREASE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME-=1

def GET_TIME():
    global GLOBAL_TIME
    return GLOBAL_TIME

def GET_MODEL_PARTITION_MEM():
    global MODEL_PARA_SIZE, MODEL_PARA_NUM, MIX_TRAINING
    # if MIX_TRAINING:
    #     return (MODEL_PARA_NUM * )


class WorkloadType(Enum):
    FORWARD_PASS_WORKLOAD = "F"
    INPUT_GRADIENT_WORKLOAD = "B"
    PARAMETER_GRADIENT_WORKLOAD = "W"


class StageSearchOrder(Enum):
    Random = "Random"
    IncDec = "IncDec"

class RunMode(Enum):
    SEARCH_SCHEDULE = "search_schedule"
    Z3_SOLVE = "z3"
    GUROBI_SOLVE = "gurobi"
    SIM_SOLVE = "sim"

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
    
RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.SIM_SOLVE
SOLVING_TIME_LIMIT = 180
STAGE_SEARCH_METHOD = StageSearchOrder.Random