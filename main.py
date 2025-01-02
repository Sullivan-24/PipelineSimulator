import os
import sys
import time
from datetime import datetime
from simulator.DSASimulator import DSASimulator
from simulator.LayerwiseSimulator import LayerwiseSimulator
from simulator.GSimulator import GSimulator
from simulator.SPSimulator import SPSimulator
from simulator.abstract import Pipeline
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
        "forward_execution_time": [FPW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_i_time": [IGW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_g_time": [PGW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
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
    }
    print(Activation.FULL_LAYER)
    print(Loss.OUTPUT)
    print(BASIC_MEMORY)
    print(GRADIENT_MEMORY / LAYER_NUM)
    print(GPU_MAX_MEM)

    if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
        config["forward_execution_time"] = [EMBEDDING_TIME] + [FPW_TIME for _ in range(LAYER_NUM)] + [LAST_FFN_F_TIME, LOSS_F_TIME]
        config["backward_execution_i_time"] = [0] + [IGW_TIME for _ in range(LAYER_NUM)] + [LAST_FFN_B_TIME, LOSS_B_TIME]
        config["backward_execution_g_time"] = [0] + [PGW_TIME for _ in range(LAYER_NUM)] + [LAST_FFN_W_TIME, LOSS_W_TIME]
        config["model_size"] = config["model_size"] + 3
    else:
        config["forward_execution_time"] =    [FPW_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_i_time"] = [IGW_TIME for _ in range(LAYER_NUM)] 
        config["backward_execution_g_time"] = [PGW_TIME for _ in range(LAYER_NUM)]
        
        fwd_time = config["forward_execution_time"]
        fwd_time[0] += EMBEDDING_TIME
        fwd_time[-1] += LOSS_F_TIME + LAST_FFN_F_TIME
        config["forward_execution_time"] = fwd_time

        iwd_time = config["backward_execution_i_time"]
        iwd_time[-1] += LOSS_B_TIME + LAST_FFN_B_TIME
        config["backward_execution_i_time"] = iwd_time

        gwd_time = config["backward_execution_g_time"]
        gwd_time[-1] += LOSS_W_TIME + LAST_FFN_W_TIME
        config["backward_execution_g_time"] = gwd_time
        
    if config["run_mode"] == RunMode.SEARCH_SCHEDULE:
        config["file_path"] = os.path.join("results", filename)
        s_time = time.time()
        simulator = DSASimulator(config)
        e_time = simulator.traverse_run()
        print(f"Traverse Run Total time: {e_time - s_time}")
    elif config["run_mode"] == RunMode.LAYERWISE_GUROBI_SOLVE:
        simulator = LayerwiseSimulator(config, 
            # device_stage_alignments=[
            #     [0, 9],
            #     [1, 8],
            #     [2, 7],
            #     [3, 6],
            #     [4, 5],
            #     ]
        )
        simulator.run(draw=True)
        simulator.show_solution_detail()
    elif config["run_mode"] == RunMode.GUROBI_SOLVE:
        simulator = GSimulator(config, 
            # device_stage_alignments=[
            #     [0, 9],
            #     [1, 8],
            #     [2, 7],
            #     [3, 6],
            #     [4, 5],
            #     ]
        )
        simulator.run(draw=True)
        simulator.show_solution_detail()
    elif config["run_mode"] == RunMode.Z3_SOLVE:
        simulator = SPSimulator(config,
        )
        simulator.run(base_solution=True, draw=True)
        simulator.show_solution_detail()
    elif config['run_mode'] == RunMode.SIM_SOLVE:
        simulator = Pipeline.PipelineScheduler()
        # simulator.show_detail_info()
        simulator.run_pipeline_parallelism()
        # simulator.show_detail_info()
        simulator.draw()
    else:
        print("Unknown run mode.")

if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{DEVICE_NUM}-{MICRO_BATCH_NUM}-{int(FPW_TIME)}-{int(IGW_TIME)}-{int(PGW_TIME)}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
