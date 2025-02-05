from dataclasses import dataclass
def MEMORY(mem):
    #TODO not always work
    if mem > G:
        return round(mem / G, 2)
    return mem

G = 1024 * 1024 * 1024
M = 1024 * 1024
K = 1024
B = G

FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes

PIXEL_BASE = 0.5
# Known parameter settings
DEVICE_NUM = 4 * 1
GPU_MAX_MEM = 80 * G / G
WORLD_SIZE = DEVICE_NUM
PP_SIZE = DEVICE_NUM
TP_SIZE = WORLD_SIZE // PP_SIZE

VOCAB_SIZE = 92544
NUM_ATTENTION_HEAD = 32
SEQ_LEN = 4 * K
HIDDEN_SIZE = 4 * K
MICRO_BATCH_SIZE = 1
MICRO_BATCH_NUM = 4 * 2 * 2
LAYER_NUM = 8
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
HEAD_B_TIME = 9
HEAD_W_TIME = 3

CE_F_TIME = 3
CE_B_TIME = 6
CE_W_TIME = 0

F_TIME = 12
B_TIME = 20
W_TIME = 8
COMM_TIME = 0

# EMBEDDING_TIME = 0
# HEAD_F_TIME = 0
# HEAD_B_TIME = 0
# HEAD_W_TIME = 0

# CE_F_TIME = 0
# CE_B_TIME = 0
# CE_W_TIME = 0

# F_TIME = 12
# B_TIME = 12
# W_TIME = 12