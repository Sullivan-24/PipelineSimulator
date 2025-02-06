from dataclasses import dataclass
from .abstract.variables import *

def MEMORY(mem):
    #TODO not always work
    if mem > G:
        return round(mem / G, 2)
    return mem

G = 1024 * 1024 * 1024
M = 1024 * 1024
K = 1024
B = G

BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
# RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.SIM_SOLVE
# RUN_MODE = RunMode.CHIMERA
SOLVING_TIME_LIMIT = 60 * 30
GLOBAL_TIME = 0
CHUNK_NUM = 2
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 3
SPLIT_BACKPROP = True
SCHEDULE_METHOD = SchedulePriority.Layerwise
SCHEDULE_METHOD = SchedulePriority.ZBV
# SCHEDULE_METHOD = SchedulePriority.Chimera

FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes
TIME_LIMIT = 10000
SHOW_WORKLOAD_TEXT = True
PIXEL_BASE = 1
PP_HEIGHT = 35
PP_ALIGN = 5
# Known parameter settings
DEVICE_NUM = 4 * 1
GPU_MAX_MEM = 80 * G / G
WORLD_SIZE = DEVICE_NUM
PP_SIZE = DEVICE_NUM
TP_SIZE = WORLD_SIZE // PP_SIZE

VOCAB_SIZE = 92544
NUM_ATTENTION_HEAD = 32
SEQ_LEN = 6 * K
HIDDEN_SIZE = 4 * K
MICRO_BATCH_SIZE = 1
MICRO_BATCH_NUM = 4 * 4 * 1 - 0
LAYER_NUM = 4 * 4 * 1
SPLIT_EMB_HEAD_CE = True

# Memory overhead calculation
b = MICRO_BATCH_SIZE
s = SEQ_LEN
h = HIDDEN_SIZE
a = NUM_ATTENTION_HEAD
l = LAYER_NUM
v = VOCAB_SIZE

MIX_TRAINING = True
DATA_TYPE: int = FP16 if MIX_TRAINING else FP32

@dataclass
class Activation:
    INPUT: int = (2*b*s*h)/G
    FULL: int = (34*b*s*h + 5*b*s*s*a)/G
    LOSS: int = (2*FP32*b*s*v)/G

LAYER_PARA_NUM = 12 * h * h + 13 * h
HEAD_PARA_NUM = v * h
LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * LAYER_PARA_NUM / G
    PARAMETER: int = DATA_TYPE * LAYER_PARA_NUM / G
    HEAD: int = DATA_TYPE * HEAD_PARA_NUM / G

# Memory Overhead
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM
OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1

# Profiled time overhead
EMBEDDING_TIME = 1
HEAD_F_TIME = 6
HEAD_B_TIME = 3
HEAD_W_TIME = 3

CE_F_TIME = 5
CE_B_TIME = 5
CE_W_TIME = 0

F_TIME = 12
B_TIME = 12
W_TIME = 12
COMM_TIME = 0
RUN_SCHEDULE = True

RUN_STANDARD_ZBV = True
# EMBEDDING_TIME = 0
# HEAD_F_TIME = 0
# HEAD_B_TIME = 0
# HEAD_W_TIME = 0
# CE_F_TIME = 0
# CE_B_TIME = 0
# CE_W_TIME = 0