import os
import sys
import time
from datetime import datetime
from simulator.DSASimulator import DSASimulator
from simulator.GSimulator import GSimulator
from simulator.SPSimulator import SPSimulator
from simulator.abstract.mutils import *

def main():
    config = {
        "run_mode": RUN_MODE,
        "device_size": int(DEVICE_NUM),
        "time_limit": int(SOLVING_TIME_LIMIT),
        "stage_order_search": STAGE_SEARCH_METHOD,
        "pp_size": int(STAGE_NUM),
        "model_size": int(MODEL_LAYER_NUM),
        "num_microbatches": int(MICRO_BATCH_NUM),
        "forward_execution_time": [FPW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_i_time": [IGW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_g_time": [PGW_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "communication_time": [[COMM_TIME if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [MAX_ACTIVATION_COUNTS * CHUNK_NUM for _ in range(STAGE_NUM)],
        "file_path": None,
    }

    if config["run_mode"] == RunMode.SEARCH_SCHEDULE:
        config["file_path"] = os.path.join("results", filename)
        s_time = time.time()
        simulator = DSASimulator(config)
        e_time = simulator.traverse_run()
        print(f"Traverse Run Total time: {e_time - s_time}")
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
        simulator.run(base_solution=True, draw=True)
        simulator.show_solution_detail()
    elif config["run_mode"] == RunMode.Z3_SOLVE:
        simulator = SPSimulator(config,
            # device_stage_alignments=[
            #     [0, 9],
            #     [1, 8],
            #     [2, 7],
            #     [3, 6],
            #     [4, 5],
            #     ]
        )
        simulator.run(base_solution=True, draw=True)
    else:
        print("Unknown run mode.")

if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{DEVICE_NUM}-{MICRO_BATCH_NUM}-{int(FPW_TIME)}-{int(IGW_TIME)}-{int(PGW_TIME)}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
