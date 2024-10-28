"""
main package
"""
import sys
from simulator.config import *
from simulator.simulator import Simulator, SPSimulator, DSASimulator

def main():
    """main function"""
    print("forward_execution_time:{}\nbackward_execution_i_time:{}\nbackward_execution_g_time:{}.".format(
            ft, bt, wt
        )
    )
    print("Device size:{}\nPipeline size:{}\nModel size:{}\nNumber of microbatches size:{}.".format(
            device_size, pp_size, model_size, nmb
        )
    )
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
        simulator = DSASimulator(config)
        simulator.traverse_run()
    else:
        simulator = SPSimulator(config)
        simulator.run(draw=True)

if __name__ == "__main__":
    main()
