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
SCHEDULE_METHOD = Schedule.STANDARD_INTERLEAVED
SCHEDULE_METHOD = Schedule.UnifiedPP
STAGE_PLACEMENT = Placement.INTERLEAVED
STAGE_PLACEMENT = Placement.WAVELIKE
if SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
    STAGE_PLACEMENT = Placement.INTERLEAVED

# --------------------- Solver config ---------------------
test_upp = True if SCHEDULE_METHOD == Schedule.UnifiedPP else False
OVERLAP_AWARE_SCHEDULE = test_upp
HOMO_DEVICE = True
# --------------------- Simulator config ---------------------
FIND_OPTIMAL_RECOMP = True
TIME_LIMIT = 11000

# [1, 2, 100, None]
OVERLAP_DEGREE = 2
MEMORY_CONSTRAIN = 1

EMB_TIME = 0
HEAD_F_TIME = 4
HEAD_B_TIME = 4
HEAD_W_TIME = 4
CE_F_TIME = 4
CE_B_TIME = 4
CE_W_TIME = 0
F_TIME = 4
B_TIME = 4
W_TIME = 4
COMM_TIME = 0

SPLIT_BACKPROP = test_upp
LAYERWISE = False
RECOMP = False
AUTO_RECOMP_SEARCH = RECOMP
RUN_SCHEDULE = False
RUN_STANDARD_ZBV = False
SCHEDULE_UNIT = MICRO_BATCH_NUM // 1
REVERSE_LAST_STAGES = False
# Run standard ZBV ---------------------
# SCHEDULE_METHOD = Schedule.ZBV
# RUN_SCHEDULE = False
# RUN_STANDARD_ZBV = True
# Run standard ZBV ---------------------
DENSITY_MAX = 1
DENSITY_MIN = 1
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
i = INTER_SIZE

LAYER_PARA_NUM = 12 * h * h + 13 * h
HEAD_PARA_NUM = v * h
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM

LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G

OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 3

@dataclass
class Parameter:
    EMB: int = v * h
    HEAD: int = v * h
    LAYER: int = 4 * h * h + 3 * h * i + 2 * h if MODEL_TYPE == "LLAMA" else 12 * h * h + 13 * h 

@dataclass
class StateMemory:
    EMB: int = DATA_TYPE * Parameter.EMB / G / TP_SIZE
    HEAD: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    LAYER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    # Optimizer M + V, gradients * 1, model * 1
    OPTIMIZER: int = FP32 * 4 * (Parameter.LAYER * l + Parameter.EMB + Parameter.HEAD) / G / (TP_SIZE * PP_SIZE) / ZERO_SIZE

ACT_OPT_COE = 0.15 # adjust by profiling results
@dataclass
class Activation:
    INPUT: int = (2*b*s*h) / G / TP_SIZE
    FULL: int = (34*b*s*h + 5*b*s*s*a) * ACT_OPT_COE / G / TP_SIZE
    LOSS: int = (2*FP32*b*s*v) / G / TP_SIZE
    
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    PARAMETER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    HEAD_INPUT: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    HEAD_PARA: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    # HEAD_INPUT: int = 0
    # HEAD_PARA: int = 0



# --------------------- Painter Config ---------------------
PIXEL_BASE = 2
PP_HEIGHT = 35
PP_ALIGN = 5
SHOW_WORKLOAD_TEXT = True
# --------------------- Painter Config ---------------------

# --------------------- Save File Config ---------------------
RES_FILE_PATH = f"schedule_results/schedules/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}.txt"
PLA_FILE_PATH = f"schedule_results/placement.txt"
TEMP_RES_PATH = f"schedule_results/result.txt"
