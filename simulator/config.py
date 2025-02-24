from dataclasses import dataclass
from .abstract.variables import *
from .model_config import *

# --------------------- Solver config ---------------------
BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.CHIMERA
RUN_MODE = RunMode.SIM_SOLVE

SOLVING_TIME_LIMIT = 60 * 30
SCHEDULE_METHOD = Schedule.Layerwise
# SCHEDULE_METHOD = Schedule.STANDARD_INTERLEAVED
# SCHEDULE_METHOD = Schedule.INTERLEAVED
# SCHEDULE_METHOD = Schedule.ZBV
STAGE_PLACEMENT = Placement.WAVELIKE
STAGE_PLACEMENT = Placement.CROSS
# STAGE_PLACEMENT = Placement.RECURRENT
STAGE_PLACEMENT = Placement.INTERLEAVED

# --------------------- Solver config ---------------------


# --------------------- Simulator config ---------------------
FIND_OPTIMAL_RECOMP = True
TEMP_TEST= True
TIME_LIMIT = 4500

EMB_TIME = 1
HEAD_F_TIME = 2
HEAD_B_TIME = 1
HEAD_W_TIME = 1
CE_F_TIME = 2
CE_B_TIME = 2
CE_W_TIME = 0
F_TIME = 4
B_TIME = 4
W_TIME = 4
COMM_TIME = 0

SPLIT_BACKPROP = True
LAYERWISE = True
RECOMP = False
RUN_SCHEDULE = False
RUN_STANDARD_ZBV = True
AUTO_RECOMP_SEARCH = False
SCHEDULE_UNIT = MICRO_BATCH_NUM // 1
REVERSE_LAST_STAGES = False

# Run standard ZBV ---------------------
# SCHEDULE_METHOD = Schedule.ZBV
# RUN_SCHEDULE = False
# RUN_STANDARD_ZBV = True
# Run standard ZBV ---------------------

# --------------------- Simulator config ---------------------

# Memory overhead calculation
GPU_MAX_MEM = 80 * G / G
FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes
MIX_TRAINING = True
DATA_TYPE: int = FP16 if MIX_TRAINING else FP32
b = MICRO_BATCH_SIZE
s = SEQ_LEN
h = HIDDEN_SIZE
a = NUM_ATTENTION_HEAD
l = LAYER_NUM
v = VOCAB_SIZE

LAYER_PARA_NUM = 12 * h * h + 13 * h
HEAD_PARA_NUM = v * h
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM

LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G

OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 3
@dataclass
class Activation:
    INPUT: int = (2*b*s*h)/G
    FULL: int = (34*b*s*h + 5*b*s*s*a)/G/TP_SIZE
    LOSS: int = (2*FP32*b*s*v)/G
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * LAYER_PARA_NUM / G
    PARAMETER: int = DATA_TYPE * LAYER_PARA_NUM / G
    HEAD: int = DATA_TYPE * HEAD_PARA_NUM / G



# --------------------- Painter Config ---------------------
PIXEL_BASE = 4
PP_HEIGHT = 35
PP_ALIGN = 5
SHOW_WORKLOAD_TEXT = True
# --------------------- Painter Config ---------------------


