"""
main package
"""
import os
import sys
from simulator.config import *
from simulator.DSASimulator import DSASimulator
from simulator.GSimulator import GSimulator
from simulator.SPSimulator import SPSimulator
from datetime import datetime
import time

def main():
    config = {
        "device_size": int(device_size),
        "time_limit": int(time_limit),
        "stage_order_search": stage_order_search,
        "pp_size": int(pp_size),
        "model_size": int(model_size),
        "virtual_stage": nvs,               # times of the number of devices
        "num_microbatches": int(nmb),
        "forward_execution_time": [ft / (int(pp_size) // int(device_size)) for _ in range(pp_size)],
        "backward_execution_i_time": [bt / (int(pp_size) // int(device_size)) for _ in range(pp_size)],
        "backward_execution_g_time": [wt / (int(pp_size) // int(device_size)) for _ in range(pp_size)],
        "communication_time": [[comm if i != j else 0 for j in range(pp_size)] for i in range(pp_size)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [8 for _ in range(pp_size)],
        "file_path": None,
    }

    # simulator = Simulator(config)
    # simulator.run()
    if len(sys.argv) == 1:
        config["file_path"] = os.path.join("results", filename)
        s_time = time.time()
        simulator = DSASimulator(config)
        e_time = simulator.traverse_run()
        print(f"Traverse Run Total time: {e_time - s_time}")
    else:
        # simulator = SPSimulator(config)
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

if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{device_size}-{nmb}-{int(ft)}-{int(bt)}-{int(wt)}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
