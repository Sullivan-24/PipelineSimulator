import os
import sys
import time
from datetime import datetime
from simulator.DSASimulator import DSASimulator
from simulator.LayerwiseSimulator import LayerwiseSimulator
from simulator.ChimeraSimulator import ChimeraSimulator
from simulator.GSimulator import GSimulator
from simulator.CompCommSimulator import CompCommSimulator
from simulator.SPSimulator import SPSimulator
from simulator.abstract import Pipeline
from simulator.abstract import ChimeraScheduler
from simulator.abstract.mutils import *

def set_workload_length(config):
    if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
        config["f_time"] = [EMB_TIME] + [F_TIME for _ in range(LAYER_NUM)] + [HEAD_F_TIME, CE_F_TIME]
        config["b_time"] = [0] + [B_TIME for _ in range(LAYER_NUM)] + [HEAD_B_TIME, CE_B_TIME]
        config["w_time"] = [0] + [W_TIME for _ in range(LAYER_NUM)] + [HEAD_W_TIME, CE_W_TIME]
        config["layer_num"] = config["layer_num"] + 3
    else:
        config["f_time"] =    [F_TIME for _ in range(LAYER_NUM)] 
        config["b_time"] = [B_TIME for _ in range(LAYER_NUM)] 
        config["w_time"] = [W_TIME for _ in range(LAYER_NUM)]
        
        fwd_time = config["f_time"]
        fwd_time[0] += EMB_TIME
        fwd_time[-1] += CE_F_TIME + HEAD_F_TIME
        config["f_time"] = fwd_time

        iwd_time = config["b_time"]
        iwd_time[-1] += CE_B_TIME + HEAD_B_TIME
        config["b_time"] = iwd_time

        gwd_time = config["w_time"]
        gwd_time[-1] += CE_W_TIME + HEAD_W_TIME
        config["w_time"] = gwd_time

def check_standard_zbv_conditions():
    if EMB_TIME!=0:
        print("Required to ignore EMB layers.")
        return False
    if HEAD_B_TIME!=0 or HEAD_F_TIME!=0 or HEAD_W_TIME!=0:
        print("Required to ignore HEAD layers.")
        return False
    if CE_B_TIME!=0 or CE_F_TIME!=0 or CE_W_TIME!=0:
        print("Required to ignore CE layers.")
    if F_TIME==B_TIME==W_TIME:
        return True
    else:
        print("Required F=B=W")
        return False
                    
def main():
    config = {
        "run_mode": RUN_MODE,
        "device_num": int(DEVICE_NUM),
        "time_limit": int(SOLVING_TIME_LIMIT),
        "stage_order_search": STAGE_SEARCH_METHOD,
        "stage_num": int(STAGE_NUM),
        "layer_num": int(LAYER_NUM),
        "nmb": int(MICRO_BATCH_NUM),
        "f_time": [F_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "b_time": [B_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "w_time": [W_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "device_mem": [GPU_MAX_MEM for _ in range(DEVICE_NUM)],
        "mix_training": MIX_TRAINING,
        "para_num": PARAMETER_NUM,
        "comm_time": [[COMM_TIME if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [MAX_ACTIVATION_COUNTS for _ in range(STAGE_NUM)],
        # "file_path": filename,
        "file_path": None,
        "base_solution" : BASE_SOLUTION,
        "schedule_method": SCHEDULE_METHOD,
        "emb_head_ce": LAYERWISE,
    }
    set_workload_length(config=config)

    if os.path.exists(f"schedule_results/{RUN_MODE}_{SCHEDULE_METHOD}_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}_l{LAYER_NUM}_{SPLIT_BACKPROP}_{LAYERWISE}.txt"):
        os.remove(f"schedule_results/{RUN_MODE}_{SCHEDULE_METHOD}_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}_l{LAYER_NUM}_{SPLIT_BACKPROP}_{LAYERWISE}.txt")
        print("delete file:{}".format(f"schedule_results/{RUN_MODE}_{SCHEDULE_METHOD}_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}_l{LAYER_NUM}_{SPLIT_BACKPROP}_{LAYERWISE}.txt"))
    
    for did in range(DEVICE_NUM):
        filepath = f"schedule_results/device{did}.txt"
        if os.path.exists(filepath):
            os.remove(filepath)
            print("delete file:{}".format(filepath))
     
    print("SEQ={},HID={}".format(SEQ_LEN,HIDDEN_SIZE))
    print("Activation Layer={},Activation Input={},Activation Loss={}".format(Activation.FULL, Activation.INPUT, Activation.LOSS))
    print("Gradient Input={},Gradient Parameters={},Gradient Head={}".format(Gradient.INPUT,Gradient.PARAMETER, Gradient.HEAD_INPUT))
    print("LOSS={},VOC={}".format(Activation.LOSS,VOCAB_SIZE))

    print("LAYER_MEM:{}".format(StateMemory.LAYER))
    print("MODEL MEM:{}".format(StateMemory.LAYER * LAYER_NUM // DEVICE_NUM))
    print("OPT MEM:{}".format(StateMemory.OPTIMIZER))
    # input()

    if config["run_mode"] == RunMode.SEARCH_SCHEDULE:
        config["file_path"] = os.path.join("results", filename)
        s_time = time.time()
        simulator = DSASimulator(config)
        e_time = simulator.traverse_run()
        print(f"Traverse Run Total time: {e_time - s_time}")
    elif config["run_mode"] == RunMode.LAYERWISE_GUROBI_SOLVE:
        simulator = LayerwiseSimulator(config=config)
        simulator.run(draw=True)
    elif config["run_mode"] == RunMode.CHIMERA:
        config["f_time"] =    [F_TIME for _ in range(LAYER_NUM)] 
        config["b_time"] = [B_TIME for _ in range(LAYER_NUM)] 
        config["w_time"] = [W_TIME for _ in range(LAYER_NUM)]
        simulator = ChimeraSimulator(config)
        simulator.run(draw=True)
    elif config["run_mode"] == RunMode.GUROBI_SOLVE:
        simulator = GSimulator(config)
        simulator.run(draw=True)
        simulator.show_solution_detail()
    elif config["run_mode"] == RunMode.Z3_SOLVE:
        simulator = SPSimulator(config,
        )
        simulator.run(base_solution=True, draw=True)
        simulator.show_solution_detail()
    elif config['run_mode'] == RunMode.SIM_SOLVE:
        if Schedule.Chimera == SCHEDULE_METHOD:
            simulator = ChimeraScheduler.ChimeraPipelineScheduler()
        else:
            simulator = Pipeline.PipelineScheduler(run_schedule=RUN_SCHEDULE)
            
            if RUN_STANDARD_ZBV and not RUN_SCHEDULE and SCHEDULE_METHOD == Schedule.ZBV:
                if check_standard_zbv_conditions():
                    simulator.run_pipeline_parallelism()
                    simulator.result2file()
                    print("ZBV F=B=W Schedule saved to data.txt.")
                    return
        simulator.run_pipeline_parallelism()
        # simulator.show_detail_info()
        simulator.show_mem_usage(show_all=True)
        simulator.draw()
        print("Time:{}.".format(simulator.last_workload.end_time))

    else:
        print("Unknown run mode.")

if __name__ == "__main__":
    # config = {
    #     "run_mode": RUN_MODE,
    #     "device_num": int(DEVICE_NUM),
    #     "time_limit": int(SOLVING_TIME_LIMIT),
    #     "stage_order_search": STAGE_SEARCH_METHOD,
    #     "stage_num": int(STAGE_NUM),
    #     "layer_num": int(LAYER_NUM),
    #     "nmb": int(MICRO_BATCH_NUM),
    #     "f_time": [F_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
    #     "b_time": [B_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
    #     "w_time": [W_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
    #     "device_mem": [GPU_MAX_MEM for _ in range(DEVICE_NUM)],
    #     "mix_training": MIX_TRAINING,
    #     "para_num": PARAMETER_NUM,
    #     "comm_time": [[COMM_TIME if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
    #     "sequential_order_constraint_strategy": "strict",
    #     "max_activation_counts": [MAX_ACTIVATION_COUNTS for _ in range(STAGE_NUM)],
    #     # "file_path": filename,
    #     "file_path": None,
    #     "base_solution" : BASE_SOLUTION,
    #     "schedule_method": SCHEDULE_METHOD,
    #     "emb_head_ce": LAYERWISE,
    #     "split_backprop": True,
    #     "pixel_base":1,
    # }
    # sim = CompCommSimulator(config=config)
    # sim.run(draw=True)
    # input("WAIT")

    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{DEVICE_NUM}-{MICRO_BATCH_NUM}-{int(F_TIME)}-{int(B_TIME)}-{int(W_TIME)}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
