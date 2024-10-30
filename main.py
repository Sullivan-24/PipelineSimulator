"""
main package
"""
import os
import sys
from simulator.config import *
from simulator.simulator import SPSimulator, GSPSimulator, DSASimulator
import gurobipy as grb
from datetime import datetime
import contextlib

def main():
    config = {
        "device_size": int(device_size),
        "pp_size": int(pp_size),
        "model_size": int(model_size),
        "virtual_stage": nvs,               # times of the number of devices
        "num_microbatches": int(nmb),
        "forward_execution_time": [ft for _ in range(pp_size)],
        "backward_execution_i_time": [bt for _ in range(pp_size)],
        "backward_execution_g_time": [wt for _ in range(pp_size)],
        "communication_time": [[comm if i != j else 0 for j in range(pp_size)] for i in range(pp_size)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [8 for _ in range(pp_size)],
    }

    # simulator = Simulator(config)
    # simulator.run()
    if len(sys.argv) == 1:
        with open(os.path.join("results", filename), "w", encoding="utf-8") as file:
            with contextlib.redirect_stdout(file):
                simulator = DSASimulator(config)
                simulator.traverse_run()
    else:
        simulator = SPSimulator(config)
        simulator.run(draw=True)

if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}-{device_size}-{nmb}.txt"

    print(f"Begin solving procedure...")
    main()
    print(f"Finish.")
