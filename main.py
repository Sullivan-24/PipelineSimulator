"""
main package
"""
import sys
from simulator.config import *
from simulator.simulator import Simulator, SimulatorGurobipy

def main():
    """main function"""
    print("forward_execution_time:{}\nbackward_execution_i_time:{}\nbackward_execution_g_time:{}.".format(ft,bt,wt))
    print("Pipeline size:{}\nModel size:{}\nNumber of microbatches size:{}.".format(pp_size,model_size,nmb))
    config = {
        "pp_size": int(pp_size),
        "model_size": int(model_size),
        "virtual_stage": nvs, # times of the number of devices
        "num_microbatches": int(nmb),
        "forward_execution_time": [ft for _ in range(pp_size)],
        "backward_execution_i_time": [bt for _ in range(pp_size)],
        "backward_execution_g_time": [wt for _ in range(pp_size)],
        "communication_time": [comm for _ in range(pp_size)],
        "sequential_order_constraint_strategy": "strict",
        "recomputing_rates":[rr for _ in range(pp_size)],
        "max_activation_counts": [8 for _ in range(pp_size)],
    }

    simulator = Simulator(config)
    simulator.run()
    # simulator = SimulatorGurobipy(config)
    # simulator.run()

if __name__ == "__main__":
    main()
