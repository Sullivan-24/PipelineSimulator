from .variables import *
from ..config import *
PIXEL_BASE = 1
BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
# RUN_MODE = RunMode.GUROBI_SOLVE
# RUN_MODE = RunMode.SIM_SOLVE
# RUN_MODE = RunMode.Z3_SOLVE
SOLVING_TIME_LIMIT = 60 * 30
GLOBAL_TIME = 0
CHUNK_NUM = 2
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 3
SPLIT_BACKPROP = True
SCHEDULE_METHOD = SchedulePriority.Layerwise
if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
    SCHEDULE_METHOD = SchedulePriority.Layerwise
# SCHEDULE_METHOD = SchedulePriority.Layerwise
# SCHEDULE_METHOD = SchedulePriority.GREEDY_v1
if RUN_MODE == RunMode.SIM_SOLVE:
    if SCHEDULE_METHOD == SchedulePriority.ONE_F_ONE_B:
        SPLIT_BACKPROP = False
        CHUNK_NUM = 1
        B_TIME += W_TIME
    elif SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
        assert CHUNK_NUM > 1, "INTERLEAVED: CHUNK_NUM should be larger than 1"
        SPLIT_BACKPROP = False
        if not SPLIT_BACKPROP:
            B_TIME += W_TIME
    elif SCHEDULE_METHOD == SchedulePriority.ZBH1:
        SPLIT_BACKPROP = True
        CHUNK_NUM = 1
    elif SCHEDULE_METHOD in (SchedulePriority.GREEDY_v1, SchedulePriority.ZBV):
        SPLIT_BACKPROP = True
        CHUNK_NUM = 2

STAGE_NUM = int(DEVICE_NUM * CHUNK_NUM)
MAX_ACTIVATION_COUNTS = int(STAGE_NUM * MAX_ACTIVATION_TIMES_OF_STAGE_NUM)
WORKLOAD_TYPE_NUM = 3 if SPLIT_BACKPROP else 2
STAGE_SEARCH_METHOD = StageSearchOrder.Random

def UPDATE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME+=1

def DECREASE_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME-=1

def GET_TIME():
    global GLOBAL_TIME
    return GLOBAL_TIME



