from ..config import *

if RUN_MODE == RunMode.CHIMERA:
    SCHEDULE_METHOD = Schedule.ZBH1
    CHUNK_NUM=1
if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
    SCHEDULE_METHOD = Schedule.Layerwise
if RUN_MODE == RunMode.GUROBI_SOLVE:
    if SCHEDULE_METHOD == Schedule.ZBH1:
        print("Pre CHUNK_NUM={}, set CHUNK_NUM=1".format(CHUNK_NUM))
        CHUNK_NUM = 1
# SCHEDULE_METHOD = Schedule.Layerwise
# SCHEDULE_METHOD = Schedule.GREEDY_v1
if RUN_MODE == RunMode.SIM_SOLVE:
    if SCHEDULE_METHOD in (Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB, Schedule.ONE_F_ONE_B):
        SPLIT_BACKPROP = False
        CHUNK_NUM = 1
        B_TIME += W_TIME
    elif SCHEDULE_METHOD in (Schedule.STANDARD_INTERLEAVED, Schedule.INTERLEAVED):
        assert CHUNK_NUM > 1, "INTERLEAVED: CHUNK_NUM should be larger than 1"
        if not SPLIT_BACKPROP:
            B_TIME += W_TIME
    elif SCHEDULE_METHOD in (Schedule.STANDARD_ZBH1, Schedule.ZBH1):
        SPLIT_BACKPROP = True
        CHUNK_NUM = 1
    elif SCHEDULE_METHOD == Schedule.ZBV and RUN_STANDARD_ZBV:
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

def RESET_TIME():
    global GLOBAL_TIME
    GLOBAL_TIME = 0



