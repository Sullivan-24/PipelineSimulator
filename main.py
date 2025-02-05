import os
import sys
import time
from datetime import datetime
from simulator.DSASimulator import DSASimulator
from simulator.LayerwiseSimulator import LayerwiseSimulator
from simulator.ChimeraSimulator import ChimeraSimulator
from simulator.GSimulator import GSimulator
from simulator.SPSimulator import SPSimulator
from simulator.abstract import Pipeline
from simulator.abstract import ChimeraScheduler
from simulator.abstract.mutils import *

def main():
    config = {
        "run_mode": RUN_MODE,
        "device_size": int(DEVICE_NUM),
        "time_limit": int(SOLVING_TIME_LIMIT),
        "stage_order_search": STAGE_SEARCH_METHOD,
        "pp_size": int(STAGE_NUM),
        "model_size": int(LAYER_NUM),
        "num_microbatches": int(MICRO_BATCH_NUM),
        "forward_execution_time": [F_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_i_time": [B_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_g_time": [W_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "device_mem": [GPU_MAX_MEM for _ in range(DEVICE_NUM)],
        "mix_training": MIX_TRAINING,
        "model_para_num": PARAMETER_NUM,
        "communication_time": [[COMM_TIME if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [MAX_ACTIVATION_COUNTS for _ in range(STAGE_NUM)],
        # "file_path": filename,
        "file_path": None,
        "base_solution" : BASE_SOLUTION,
        "schedule_method": SCHEDULE_METHOD,
        "emb_head_ce": SPLIT_EMB_HEAD_CE,
    }
    print(MEMORY(Activation.FULL))
    print(MEMORY(OPTIMIZER_MEMORY))
    print(MEMORY(GPU_MAX_MEM))

    if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
        config["forward_execution_time"] = [EMBEDDING_TIME] + [F_TIME for _ in range(LAYER_NUM)] + [HEAD_F_TIME, CE_F_TIME]
        config["backward_execution_i_time"] = [0] + [B_TIME for _ in range(LAYER_NUM)] + [HEAD_B_TIME, CE_B_TIME]
        config["backward_execution_g_time"] = [0] + [W_TIME for _ in range(LAYER_NUM)] + [HEAD_W_TIME, CE_W_TIME]
        config["model_size"] = config["model_size"] + 3
    else:
        config["forward_execution_time"] =    [F_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_i_time"] = [B_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_g_time"] = [W_TIME for _ in range(LAYER_NUM)]
        
        fwd_time = config["forward_execution_time"]
        fwd_time[0] += EMBEDDING_TIME
        fwd_time[-1] += CE_F_TIME + HEAD_F_TIME
        config["forward_execution_time"] = fwd_time

        iwd_time = config["backward_execution_i_time"]
        iwd_time[-1] += CE_B_TIME + HEAD_B_TIME
        config["backward_execution_i_time"] = iwd_time

        gwd_time = config["backward_execution_g_time"]
        gwd_time[-1] += CE_W_TIME + HEAD_W_TIME
        config["backward_execution_g_time"] = gwd_time
    
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
        config["forward_execution_time"] =    [F_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_i_time"] = [B_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_g_time"] = [W_TIME for _ in range(LAYER_NUM)]
        simulator = ChimeraSimulator(config)
        simulator.run(draw=True)
    elif config["run_mode"] == RunMode.GUROBI_SOLVE:
        config["forward_execution_time"] =    [F_TIME * LAYER_NUM // DEVICE_NUM // CHUNK_NUM for _ in range(STAGE_NUM)] 
        config["backward_execution_i_time"] = [B_TIME * LAYER_NUM // DEVICE_NUM // CHUNK_NUM for _ in range(STAGE_NUM)] 
        config["backward_execution_g_time"] = [W_TIME * LAYER_NUM // DEVICE_NUM // CHUNK_NUM for _ in range(STAGE_NUM)]
        
        fwd_time = config["forward_execution_time"]
        fwd_time[0] += EMBEDDING_TIME
        fwd_time[-1] += CE_F_TIME + HEAD_F_TIME
        config["forward_execution_time"] = fwd_time

        iwd_time = config["backward_execution_i_time"]
        iwd_time[-1] += CE_B_TIME + HEAD_B_TIME
        config["backward_execution_i_time"] = iwd_time

        gwd_time = config["backward_execution_g_time"]
        gwd_time[-1] += CE_W_TIME + HEAD_W_TIME
        config["backward_execution_g_time"] = gwd_time

        simulator = GSimulator(config)
        simulator.run(draw=True)
        simulator.show_solution_detail()
    elif config["run_mode"] == RunMode.Z3_SOLVE:
        simulator = SPSimulator(config,
        )
        simulator.run(base_solution=True, draw=True)
        simulator.show_solution_detail()
    elif config['run_mode'] == RunMode.SIM_SOLVE:
        if SchedulePriority.Chimera == SCHEDULE_METHOD:
            simulator = ChimeraScheduler.ChimeraPipelineScheduler()
        else:
            simulator = Pipeline.PipelineScheduler(run_schedule=RUN_SCHEDULE)
        simulator.run_pipeline_parallelism()
        if SCHEDULE_METHOD == SchedulePriority.ZBV:
            simulator.result2file()
        # simulator.result2schedule()
        # simulator.show_detail_info()
        simulator.show_mem_usage(show_all=True)
        simulator.draw()
    else:
        print("Unknown run mode.")

if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{DEVICE_NUM}-{MICRO_BATCH_NUM}-{int(F_TIME)}-{int(B_TIME)}-{int(W_TIME)}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
