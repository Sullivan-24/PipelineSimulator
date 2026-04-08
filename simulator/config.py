from dataclasses import dataclass
from simulator.abstract.variables import *
from simulator.model_config import *
import math
def allocate_tasks(machine_capacities, total_tasks, opti=False):
    """
    将任务分配给不同计算能力的机器，最小化最大执行时间
    
    Args:
        machine_capacities: 机器计算能力列表
        total_tasks: 总任务数
    
    Returns:
        每台机器分配的任务数
    """
    if opti :
        n = len(machine_capacities)
        allocation = [0] * n
        current_time = [0.0] * n
        
        # 初始化：给每个工人分配1个任务（如果总任务足够）
        initial_assign = min(total_tasks, n)
        for i in range(initial_assign):
            allocation[i] += 1
            current_time[i] = 1.0 / machine_capacities[i]
        total_tasks -= initial_assign
        
        # 贪心分配剩余任务：每次给能力最大的工人分配下一个任务（在相同能力中，选择当前时间最小的）
        while total_tasks > 0:
            # 找到能力最大的工人
            max_cap = max(machine_capacities)
            candidates = [i for i in range(n) if machine_capacities[i] == max_cap]
            # 在候选者中，选择当前时间最小的
            min_time_among_candidates = min(current_time[i] for i in candidates)
            min_time_idx = [i for i in candidates if current_time[i] == min_time_among_candidates][0]
            # 分配一个任务
            allocation[min_time_idx] += 1
            # 更新完成时间
            current_time[min_time_idx] += 1.0 / machine_capacities[min_time_idx]
            total_tasks -= 1
    else:
        total_capacity = sum(machine_capacities)
        # 按比例分配，然后四舍五入到整数
        allocation = [round(capacity / total_capacity * total_tasks) for capacity in machine_capacities]

        # 确保总数等于total_tasks
        diff = total_tasks - sum(allocation)
        while diff != 0:
            if diff > 0:
                # 找到当前负载最小的机器（allocation / capacity 最小），分配额外任务
                loads = [allocation[i] / machine_capacities[i] for i in range(len(machine_capacities))]
                min_load_idx = loads.index(min(loads))
                allocation[min_load_idx] += 1
                diff -= 1
            else:
                # 找到当前负载最大的机器，减少任务
                loads = [allocation[i] / machine_capacities[i] for i in range(len(machine_capacities))]
                max_load_idx = loads.index(max(loads))
                allocation[max_load_idx] -= 1
                diff += 1
    return allocation
# --------------------- Solver config ---------------------
BASE_SOLUTION = True
RUN_MODE = RunMode.LAYERWISE_GUROBI_SOLVE
RUN_MODE = RunMode.GUROBI_SOLVE
RUN_MODE = RunMode.CHIMERA
RUN_MODE = RunMode.SIM_SOLVE

SOLVING_TIME_LIMIT = 60 * 30
SCHEDULE_METHOD = Schedule.Layerwise
SCHEDULE_METHOD = Schedule.STANDARD_1F1B
# SCHEDULE_METHOD = Schedule.STANDARD_INTERLEAVED
# SCHEDULE_METHOD = Schedule.STANDARD_ZBH
# SCHEDULE_METHOD = Schedule.Mist
# SCHEDULE_METHOD = Schedule.OctoPipe
# SCHEDULE_METHOD = Schedule.ZBV
# SCHEDULE_METHOD = Schedule.STANDARD_AFAB
STAGE_PLACEMENT = Placement.INTERLEAVED
# STAGE_PLACEMENT = Placement.SEARCHED
# STAGE_PLACEMENT = Placement.WAVELIKE
SPLIT_BACKPROP = True
LAYER_ADAPT = False #TODO
if SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
    STAGE_PLACEMENT = Placement.INTERLEAVED
    CHUNK_NUM = LAYER_NUM // PP_SIZE
    SPLIT_BACKPROP = False
if SCHEDULE_METHOD in (Schedule.STANDARD_ZBH, Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB):
    CHUNK_NUM = 1
Hierarchical = True
test_upp = True if SCHEDULE_METHOD == Schedule.OctoPipe else False

# --------------------- Solver config ---------------------

CHUNK_NUM = 1
DP_SIZE = 2
HETER_RATIOS = [[1 for _ in range(PP_SIZE)]for _ in range(DP_SIZE)]
HETER_DP_ID = []
HETER_PP_ID = []
Failure_ranks_map = [[[] for _ in range(PP_SIZE)] for _ in range(DP_SIZE)]
FAILURE_DP_ID = []
FAILURE_PP_ID = []
ALL_TPfail_map = [[] for _ in range(DP_SIZE)]
FAILURE_GLOBAL_RANKS = []
Failure_ranks_info = []
Available_ranks_map = [[[i for i in range(TP_SIZE)] for _ in range(PP_SIZE) ] for _ in range(DP_SIZE)]

HETER_DEVICE = True
HETER_DEVICE_Transfer = False

# HETER_RATIOS[0][0] = 3
# HETER_RATIOS[0][1] = 1.5
# HETER_RATIOS[0][2] = 3
# HETER_RATIOS[0][3] = 1.5
# HETER_RATIOS[0][4] = 1.5
# HETER_RATIOS[0][5] = 1.5
# HETER_RATIOS[0][6] = 3
# HETER_RATIOS[0][7] = 1.5


# HETER_RATIOS[1][0] = 1.5
# HETER_RATIOS[1][1] = 3
# HETER_RATIOS[1][2] = 1.5
# HETER_RATIOS[1][3] = 3
# HETER_RATIOS[1][4] = 3
# HETER_RATIOS[1][5] = 3
# HETER_RATIOS[1][6] = 1.5
# HETER_RATIOS[1][7] = 3


FAILURE_DEVICE = False
# Failure_ranks_map[0][2]= [0,1,2,3]

Good = False
BEST = False
opti = False#for layer paratation
NMB_PER_DP = [MICRO_BATCH_NUM]*DP_SIZE
# NMB_PER_DP = [12,4]
if Good:
    SCHEDULE_METHOD = Schedule.OctoPipe
    LAYER_ADAPT = True
if BEST == True:
    SCHEDULE_METHOD = Schedule.OctoPipe
    LAYER_ADAPT = True 
    opti = False
    HETER_DEVICE_Transfer = True
if SCHEDULE_METHOD != Schedule.OctoPipe:
    HETER_DEVICE_Transfer = False
if SCHEDULE_METHOD == Schedule.OctoPipe:
    NMB_PER_DP = [MICRO_BATCH_NUM]*DP_SIZE


if FAILURE_DEVICE:
    for dp_index, pp_ranks in enumerate(Failure_ranks_map):
        for pp_index, failure_tp_ranks in enumerate(pp_ranks): 
            TPGroup = Available_ranks_map[dp_index][pp_index]
            for failure_tp_rank in failure_tp_ranks:
                TPGroup.remove(failure_tp_rank)
                Failure_ranks_info.append(f"dp:{dp_index}, pp:{pp_index},tp:{failure_tp_rank}")
            if len(TPGroup)>0:
                while(math.log2(len(TPGroup))%1 != 0):
                    Failure_ranks_map[dp_index][pp_index].append(TPGroup.pop(-1))
            for failure_local_tp_rank in Failure_ranks_map[dp_index][pp_index]:
                FAILURE_GLOBAL_RANKS.append(pp_index*(DP_SIZE*TP_SIZE)+dp_index*TP_SIZE+failure_local_tp_rank)
            Available_ranks_map[dp_index][pp_index] = TPGroup
            if len(TPGroup)== 0:
                FAILURE_DP_ID.append(dp_index)  
                FAILURE_PP_ID.append(pp_index)
                ALL_TPfail_map[dp_index].append(pp_index)
            else:
                slow_down_by_failTP = int(TP_SIZE/len(TPGroup))
                if slow_down_by_failTP == 2:
                    HETER_RATIOS[dp_index][pp_index] = 1.5
                elif slow_down_by_failTP == 4:
                    HETER_RATIOS[dp_index][pp_index] = 2
                elif slow_down_by_failTP == 8:
                    HETER_RATIOS[dp_index][pp_index] = 3.5

if HETER_DEVICE:
    for dp_index, comptime_pps in enumerate(HETER_RATIOS):
        for pp_index, comptime in enumerate(comptime_pps):
            if comptime > 1:
                HETER_DP_ID.append(dp_index)
                HETER_PP_ID.append(pp_index)
# print(f"HETER_DP_ID: {HETER_DP_ID}, HETER_PP_ID: {HETER_PP_ID}")
pipeline_comp_power = [0 for _ in range(PP_SIZE)]
for dp_index in range(DP_SIZE):
    for pp_index in range(PP_SIZE):
        if HETER_DEVICE:
            pipeline_comp_power[pp_index] += 1/HETER_RATIOS[dp_index][pp_index]
        else:
            pipeline_comp_power[pp_index] += 1
suggest_allocation = allocate_tasks(pipeline_comp_power, LAYER_NUM)
print(f"Suggest Layer Assignment: {suggest_allocation}, pipeline_comp_power: {pipeline_comp_power}")

OVERLAP_AWARE_SCHEDULE = True if not HETER_DEVICE else False
OVERLAP_AWARE_SCHEDULE = True
# --------------------- Simulator config ---------------------
FIND_OPTIMAL_RECOMP = False
TIME_LIMIT = 15000
HEAD_DP = False if test_upp else False
# [1, 2, 100, None]
OVERLAP_DEGREE = None
MEMORY_CONSTRAIN = 0.9
MEMORY_REDUCATION = 0.0
IDEAL_SITUATION = True

# Gemma
EMB_F_TIME = 0
EMB_B_TIME = 0
EMB_W_TIME = 0
HEAD_F_TIME = 0
HEAD_B_TIME = 0
HEAD_W_TIME = 0
CE_F_TIME = 0
CE_B_TIME = 0
CE_W_TIME = 0
F_TIME = 0
B_TIME = 0
W_TIME = 0
COMM_TIME = [[0 for _ in range(PP_SIZE)] for _ in range(PP_SIZE)]
# COMM_TIME[0][1] = 120

if SCHEDULE_METHOD in (Schedule.STANDARD_ZBH, Schedule.ZBV):
    SPLIT_BACKPROP = True
    if SCHEDULE_METHOD == Schedule.ZBV:
        STAGE_PLACEMENT = Placement.WAVELIKE

if SPLIT_BACKPROP:
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_W_TIME = HEAD_B_TIME // 2
    HEAD_B_TIME = HEAD_B_TIME // 2
    CE_B_TIME = 0
    CE_W_TIME = 0
    W_TIME = B_TIME // 2
    B_TIME = B_TIME // 2

if IDEAL_SITUATION:
    EMB_F_TIME = 0
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_F_TIME = 0
    HEAD_B_TIME = 0
    HEAD_W_TIME = 0
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0

LAYERWISE = False
RECOMP = False
AUTO_RECOMP_SEARCH = False
RUN_SCHEDULE = False
RUN_STANDARD_ZBV = False
if not RUN_SCHEDULE and RUN_STANDARD_ZBV:
    print("Overlooking non-transformer layers")
    EMB_F_TIME = 0
    EMB_B_TIME = 0
    EMB_W_TIME = 0
    HEAD_F_TIME = 0
    HEAD_B_TIME = 0
    HEAD_W_TIME = 0
    CE_F_TIME = 0
    CE_B_TIME = 0
    CE_W_TIME = 0
    F_TIME = 12
    B_TIME = 12
    W_TIME = 12

if DEEPSEEK + GEMMA + NEMOTRONH > 1:
    print(f"DeepSeek:{DEEPSEEK}, Gemma:{GEMMA}, NemotronH:{NEMOTRONH}")

SAVE_MEMORY = True
CONSTRAIN_WARMUP = False
SWITCH_WORKLOAD_TYPE = True

# f_b_w = [1,1.6,0.4] #llama2 40layers,32layers
#qwen H200:
# f_b_w = [1,1.6,0.4] #7b 
# f_b_w = [1,1.55,0.45] # 14b
f_b_w = [1,1.5,0.5]#32b [40:60:20]
# f_b_w = [1,1,1]
#H800
# if LAYER_NUM == 80 and PP_SIZE==16:
#     f_b_w = [1,1.8,0.5]
#     if ZERO_SIZE == 4 and TP_SIZE ==4:#A100
#         f_b_w = [1,1.5,0.6]
# elif LAYER_NUM == 64 and PP_SIZE==8:
#     f_b_w = [1,2,0.5]#llama2,64layers
# elif LAYER_NUM == 40 and PP_SIZE==4:
#     f_b_w = [1,2,2/7]
# elif LAYER_NUM == 32 and PP_SIZE==2:
#     f_b_w = [1,1.5,1/6]
# if not SPLIT_BACKPROP:
#     f_b_w = [f_b_w[0],f_b_w[1]+f_b_w[2],0]

# #A100
# if LAYER_NUM == 80 and PP_SIZE==16:
#     f_b_w = [1,1.5,0.6]
# elif LAYER_NUM == 64 and PP_SIZE==8:
#     f_b_w = [1,1.5,0.5]#llama2,64layers
# elif LAYER_NUM == 40 and PP_SIZE==4:
#     f_b_w = [1,1.6,0.4]
# elif LAYER_NUM == 32 and PP_SIZE==2:
#     f_b_w = [1,1.45,0.3]
# if not SPLIT_BACKPROP:
#     f_b_w = [f_b_w[0],f_b_w[1]+f_b_w[2],0]



F_TIME = 10
F_TIMES = [F_TIME] * LAYER_NUM
B_TIMES = [F_TIME*f_b_w[1]] * LAYER_NUM
W_TIMES = [F_TIME*f_b_w[2]] * LAYER_NUM

if not IDEAL_SITUATION:
    F_TIME = 10
    F_TIMES = [F_TIME] * LAYER_NUM
    B_TIMES = [F_TIME] * LAYER_NUM
    W_TIMES = [F_TIME] * LAYER_NUM
    if GEMMA:
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["GEMMA"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_tf, tb_tf, tw_tf, _, _, _, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
            B_TIMES = [t*(tb_tf+tw_tf) for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hf_tf
            HEAD_B_TIME = F_TIME * (hb_tf + hw_tf)
            if SPLIT_BACKPROP:
                B_TIMES = [t*tb_tf for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*tw_tf for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_tf
                HEAD_W_TIME = F_TIME * hw_tf
        except:
            print("----- No profiled data! Use predefined ratios. -----")

    if DEEPSEEK:
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["DEEPSEEK"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_tf, tb_tf, tw_tf, mf_tf, mb_tf, mw_tf, hf_tf, hb_tf, hw_tf] = [round(r, 1) for r in ratios]
            B_TIMES = [t*(mb_tf+mw_tf) if i >= LAYER_NUM//PP_SIZE - 1  else t * (tb_tf+tw_tf) for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hw_tf
            HEAD_B_TIME = F_TIME * (hb_tf+hw_tf)
            if SPLIT_BACKPROP:
                if tw_tf == 0:
                    tw_tf = 0.2
                    tb_tf -= tw_tf
                B_TIMES = [t*mb_tf if i >= LAYER_NUM//PP_SIZE - 1  else t * tb_tf for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*mw_tf if i >= LAYER_NUM//PP_SIZE - 1  else t * tw_tf for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_tf
                HEAD_W_TIME = F_TIME * hw_tf
            F_TIMES = [t*mf_tf if i >= LAYER_NUM//PP_SIZE - 1 else t for i,t in enumerate(F_TIMES)]
        except:
            print("----- No profiled data! Use predefined ratios. -----")

    if NEMOTRONH:
        diff = 3 * N_SCALE
        try:
            from data.profiled_data import profiled_data
            ratios = profiled_data["NEMOTRONH"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
            [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
            print(hf_mf,hb_mf,hw_mf)
            B_TIMES = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(F_TIMES)]
            HEAD_F_TIME = F_TIME * hf_mf
            HEAD_B_TIME = F_TIME * (hb_mf + hw_mf)
            if SPLIT_BACKPROP:
                B_TIMES = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(F_TIMES)]
                W_TIMES = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(F_TIMES)]
                HEAD_B_TIME = F_TIME * hb_mf
                HEAD_W_TIME = F_TIME * hw_mf
            F_TIMES = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(F_TIMES)]
        except:
            print("----- No profiled data! Use predefined ratios. -----")


    if VARYLEN:
        diff = 12
        from data.profiled_data import profiled_data
        ratios = profiled_data["NEMOTRONH"][HIDDEN_SIZE][SEQ_LEN][VOCAB_SIZE]
        [tf_mf, tb_mf, tw_mf, mf_mf, mb_mf, mw_mf, hf_mf, hb_mf, hw_mf] = [round(r, 1) for r in ratios]
        B_TIMES = [t*(tb_mf+tw_mf) if (i+1)%diff==0  else t * mb_mf for i,t in enumerate(F_TIMES)]
        HEAD_F_TIME = F_TIME * hf_mf
        HEAD_B_TIME = F_TIME * (hb_mf + hw_mf)
        if SPLIT_BACKPROP:
            B_TIMES = [t*tb_mf if (i+1)%diff==0  else t * (mb_mf-0.1) for i,t in enumerate(F_TIMES)]
            W_TIMES = [t*tw_mf if (i+1)%diff==0  else t * 0.1 for i,t in enumerate(F_TIMES)]
            HEAD_B_TIME = F_TIME * hb_mf
            HEAD_W_TIME = F_TIME * hw_mf
        F_TIMES = [t*tf_mf if (i+1)%diff==0 else t for i,t in enumerate(F_TIMES)]
        print("------ Test vary length sequence. -----")

SCHEDULE_UNIT = MICRO_BATCH_NUM // 1
REVERSE_LAST_STAGES = False
REVERSE_FIRST_STAGES = False
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

LAYER_PARA_NUM = 4 * h * h + 3 * h * i + 2 * h if MODEL_TYPE in ("LLAMA", "Qwen") else 12 * h * h + 13 * h 
HEAD_PARA_NUM = v * h
PARAMETER_NUM = LAYER_PARA_NUM * LAYER_NUM + HEAD_PARA_NUM

LAYER_MEMORY = DATA_TYPE * LAYER_PARA_NUM / G
HEAD_MEMORY = DATA_TYPE * HEAD_PARA_NUM / G

OPTIMIZER_MEMORY = PARAMETER_NUM * FP32 * 3 / G # Optimizer status * 2, gradients * 1, model parameters * 1
MAX_ACTIVATION_TIMES_OF_STAGE_NUM = 1

@dataclass
class Parameter:
    EMB: int = b * v * h
    HEAD: int = b * v * h
    LAYER: int = LAYER_PARA_NUM

@dataclass
class StateMemory:
    EMB: int = DATA_TYPE * Parameter.EMB / G / TP_SIZE
    HEAD: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    LAYER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    # Optimizer M + V, gradients * 1, model * 1
    OPTIMIZER: int = FP32 * 4 * (Parameter.LAYER * l + Parameter.EMB + Parameter.HEAD) / G / (TP_SIZE * PP_SIZE) / ZERO_SIZE

ACT_OPT_COE = 0.18298 # adjust by profiling results
ACT_B_RATIO = 0.5669
ACT_W_RATIO = 1 - ACT_B_RATIO
ACT_HEAD_B = 2/3
ACT_HEAD_W = 1 - ACT_HEAD_B
@dataclass
class Activation:
    INPUT: int = (2*b*s*h) / G / TP_SIZE
    FULL: int = (34*b*s*h + 5*b*s*s*a) * ACT_OPT_COE / G / TP_SIZE
    LOSS: int = (2*FP32*b*s*v) / G / TP_SIZE
    HEAD: int = (3*FP16*b*s*h) / G / TP_SIZE
    EMB: int = (FP16*b*s*h) / G / TP_SIZE

GRAD_COE = 0.225
# 1.5->0.6
@dataclass
class Gradient:
    INPUT: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE * GRAD_COE
    PARAMETER: int = DATA_TYPE * Parameter.LAYER / G / TP_SIZE
    HEAD_INPUT: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    HEAD_PARA: int = DATA_TYPE * Parameter.HEAD / G / TP_SIZE
    # HEAD_INPUT: int = 0
    # HEAD_PARA: int = 0



# --------------------- Painter Config ---------------------
PIXEL_BASE = 1
PP_HEIGHT = 25
PP_ALIGN = 5
SHOW_WORKLOAD_TEXT = True
if CHUNK_NUM > PP_SIZE:
    SHOW_WORKLOAD_TEXT = False
# --------------------- Painter Config ---------------------

# --------------------- Save File Config ---------------------
SAVE_RES_TO_FILE = True
SCH_FILE_PATH = f"schedule_results/{MODEL_NAME}/schedules/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
PLA_FILE_PATH = f"schedule_results/{MODEL_NAME}/placements/heter{HETER_DEVICE}/vs{VOCAB_SIZE}_l{LAYER_NUM}_s{SEQ_LEN}_h{HIDDEN_SIZE}/mb{MICRO_BATCH_NUM}_pp{PP_SIZE}_tp{TP_SIZE}_zr{ZERO_SIZE}_c{CHUNK_NUM}/{SCHEDULE_METHOD.name}_{STAGE_PLACEMENT.name}_w{SPLIT_BACKPROP}_l{LAYERWISE}_o{OVERLAP_DEGREE}.txt"
TEMP_PLA_PATH = f"schedule_results/{MODEL_NAME}/placement.txt"
TEMP_RES_PATH = f"schedule_results/{MODEL_NAME}/result.txt"

STAGE_NUM = int(PP_SIZE * CHUNK_NUM)
assert STAGE_NUM <= LAYER_NUM, f"Stage ({STAGE_NUM}) should be less than Layer ({LAYER_NUM}). "

WORKLOAD_TYPE_NUM = 3
if not SPLIT_BACKPROP:
    B_TIME += W_TIME
    WORKLOAD_TYPE_NUM = 2

MAX_ACTIVATION_COUNTS = int(STAGE_NUM * 2)
MAX_ACT = 1
PROFILE_GENERATION = False
