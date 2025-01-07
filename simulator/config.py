from dataclasses import dataclass

G = 1024 * 1024 * 1024
M = 1024 * 1024
K = 1024
B = G

FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes

# Known parameter settings
DEVICE_NUM = 4
GPU_MAX_MEM = 80 * G
WORLD_SIZE = DEVICE_NUM
PP_SIZE = DEVICE_NUM
TP_SIZE = WORLD_SIZE // PP_SIZE

VOCAB_SIZE = 92544
NUM_ATTENTION_HEAD = 32
SEQ_LEN = 4096
HIDDEN_SIZE = 4096
MICRO_BATCH_SIZE = 1
MICRO_BATCH_NUM = 12
LAYER_NUM = 8


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
    INPUT: int = DATA_TYPE * b * s * h
    LAYER_NORM: int = DATA_TYPE * b * s * h
    ATTN_DROPOUT: int = b * s * s * a
    MLP_DROPOUT: int = b * s * h
    ATTENTION_PART: int = (5 * DATA_TYPE * b * s * h + 2 * DATA_TYPE * b * s * s * a) + MLP_DROPOUT + ATTN_DROPOUT + LAYER_NORM
    MLP_PART: int = 9 * DATA_TYPE * b * s * h + MLP_DROPOUT + LAYER_NORM
    FULL_LAYER: int = ATTENTION_PART + MLP_PART
    LOSS: int = 2 * FP32 * b * s * v

LAYER_PARA_NUM = 12 * h * h + 13 * h
HEAD_PARA_NUM = v * h
LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * LAYER_PARA_NUM
    PARAMETER: int = DATA_TYPE * LAYER_PARA_NUM
    HEAD: int = DATA_TYPE * HEAD_PARA_NUM

@dataclass
class Loss:
    OUTPUT: int = FP32 * 2 * b * s * v

# Memory Overhead
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM
MODEL_MEMORY = PARAMETER_NUM * FP16 if MIX_TRAINING else PARAMETER_NUM * FP32
GRADIENT_MEMORY = PARAMETER_NUM * FP16 if MIX_TRAINING else PARAMETER_NUM * FP32
OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 # Optimizer status * 2, gradients * 1, model parameters * 1

BASIC_MEMORY = (MODEL_MEMORY + OPTIMIZER_MEMORY) / (PP_SIZE * TP_SIZE)

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
COMM_TIME = 1