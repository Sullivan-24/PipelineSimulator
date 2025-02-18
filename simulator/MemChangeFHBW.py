FP32 = 4 # 4 Bytes
FP16 = 2 # 2 Bytes
TIME_LIMIT = 10000
G = 1024 * 1024 * 1024
M = 1024 * 1024
K = 1024
B = G

SHOW_WORKLOAD_TEXT = False
PIXEL_BASE = 1
PP_HEIGHT = 35
PP_ALIGN = 5
# Known parameter settings
DEVICE_NUM = 4 * 4
GPU_MAX_MEM = 80 * G / G
WORLD_SIZE = DEVICE_NUM
PP_SIZE = DEVICE_NUM
TP_SIZE = WORLD_SIZE // PP_SIZE

RECOMP = False
RECOMP = True
VOCAB_SIZE = 92544
NUM_ATTENTION_HEAD = 32
SEQ_LEN = 4 * K
HIDDEN_SIZE = 4 * K
MICRO_BATCH_SIZE = 1
MICRO_BATCH_NUM = 4 * 2 * 4
LAYER_NUM = 4 * 1 * 9
SPLIT_EMB_HEAD_CE = True

b = MICRO_BATCH_SIZE
s = SEQ_LEN
h = HIDDEN_SIZE
a = NUM_ATTENTION_HEAD
l = LAYER_NUM
v = VOCAB_SIZE
G = 1024 * 1024 * 1024
from dataclasses import dataclass

MIX_TRAINING = True
DATA_TYPE: int = 2 if MIX_TRAINING else 4
@dataclass
class Activation:
    INPUT: int = (2*b*s*h)/G
    FULL: int = (34*b*s*h + 5*b*s*s*a)/G
    LOSS: int = (2*2*b*s*v)/G

LAYER_PARA_NUM = 12 * h * h + 13 * h
HEAD_PARA_NUM = v * h
LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * LAYER_PARA_NUM / G
    PARAMETER: int = DATA_TYPE * LAYER_PARA_NUM / G
    HEAD: int = DATA_TYPE * HEAD_PARA_NUM / G
import matplotlib.pyplot as plt
title_size = 24
xy_label_size = 20
xy_tick_size = 16
# 数据
ywo = [0, Activation.FULL, Activation.FULL + Activation.LOSS, Activation.FULL, Activation.FULL + Gradient.INPUT, Activation.FULL + Gradient.INPUT + Gradient.PARAMETER, 0]
yw = [0, Activation.INPUT, Activation.INPUT + Activation.LOSS, Activation.INPUT, Activation.FULL + Gradient.INPUT, Activation.FULL + Gradient.INPUT + Gradient.PARAMETER, 0]
x = range(len(ywo))
x = ["F0","F1-H0","H","H1-B0","B1-W0","W","W1"]
# x = ["F0","F1-H0","H0.5","H1-B0","B1-W0","W0.5","W1"]
awo = [0, 0, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, 0]
xawo = [0, 1, 1, 2, 3, 4, 5, 6, 6]
ao = [0, 0, 0, 0, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, 0]
xao = [0, 1, 2, 3, 3, 4, 5, 6, 6]

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
title = "Memory Usage Changes While Computing F-H-B-W"
fig.suptitle(title, fontsize=title_size, y=0.950)

# 第一张子图：Memory Usage
ax1.plot(x, ywo, marker='^', linestyle='dashdot', c='blue', label="Layerwise w/o Recomp.")
ax1.plot(x, yw, marker='*', linestyle='--', c='red', label="Layerwise w/ Recomp.")
ax1.plot(xawo, awo, marker='^', linestyle='dashdot', c='#006600', alpha=0.8, label="Act. counts w/o Recomp.")
ax1.plot(xao, ao, marker='*', linestyle='dashed', c='#FF8000', alpha=0.8, label="Act. counts w/ Recomp.")
ax1.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
ax1.set_xlabel("SEQ_LEN=4K, HID_SIZE=4k", fontsize=xy_label_size)
# ax1.legend(fontsize=12)
ax1.grid(True)
ax1.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
ax1.tick_params(axis='x', labelsize=xy_tick_size-3)
ax1.tick_params(axis='y', labelsize=xy_tick_size) 

# ax1.set_xticks([]) 

s = 8*K
h = 8*K

full = (34*b*s*h + 5*b*s*s*a)/G
loss = (2*2*b*s*v)/G
act_input = (2*b*s*h)/G
gradient_input = DATA_TYPE * (12 * h * h + 13 * h) / G
gradient_para = DATA_TYPE * (12 * h * h + 13 * h) / G
head_para = DATA_TYPE * v * h / G

ywo = [0, full, full + loss, full, full + gradient_input, full + gradient_input + gradient_para, 0]
yw = [0, act_input, act_input + loss, act_input, full + gradient_input, full + gradient_input + gradient_para, 0]
x = range(len(ywo))
x = ["F0","F1-H0","H","H1-B0","B1-W0","W","W1"]

awo = [0, 0, full, full, full, full, full, full, 0]
xawo = [0, 1, 1, 2, 3, 4, 5, 6, 6]
ao = [0, 0, 0, 0, full, full, full, full, 0]
xao = [0, 1, 2, 3, 3, 4, 5, 6, 6]
ax2.plot(x, ywo, marker='^', linestyle='dashdot', c='blue', label="Layerwise w/o Recomp.")
ax2.plot(x, yw, marker='*', linestyle='--', c='red', label="Layerwise w/ Recomp.")
ax2.plot(xawo, awo, marker='^', linestyle='dashdot', c='#006600', alpha=0.8, label="Act. counts w/o Recomp.")
ax2.plot(xao, ao, marker='*', linestyle='dashed', c='#FF8000', alpha=0.8, label="Act. counts w/ Recomp.")
ax2.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
ax2.set_xlabel("SEQ_LEN=8K, HID_SIZE=8K", fontsize=xy_label_size)
ax2.legend(fontsize=12)
# ax2.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7])
# ax2.set_yticks([0,1.0,2,3,4,5,6,7])
ax2.grid(True)

ax2.tick_params(axis='x', labelsize=xy_tick_size-3)
ax2.tick_params(axis='y', labelsize=xy_tick_size) 
# 调整布局
plt.tight_layout()

plt.savefig(f"/Users/hanayukino/{title}.svg",format='svg',dpi=200)
# 显示图形
plt.show()