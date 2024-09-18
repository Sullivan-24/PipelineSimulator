"""
main package
"""
import sys
from simulator.simulator import Simulator
forward_execution_time = [4, 4]
backward_execution_time = [t * 1.5 for t in forward_execution_time]
backward_execution_time2 = [t for t in forward_execution_time]
num_micro_batch = int(len(forward_execution_time) * 1.0)
def set_execution_time(time_value, stages_num = None, nmb = None):
    global forward_execution_time, backward_execution_time, backward_execution_time2, num_micro_batch
    forward_execution_time = [int(time_value) for _ in range(stages_num)] if stages_num else [time_value for _ in range(len(forward_execution_time))]
    backward_execution_time = [int(time_value * 1.5) for t in forward_execution_time]
    backward_execution_time2 = [int(4 * 1.0) for t in forward_execution_time]
    num_micro_batch = nmb if nmb else int(len(forward_execution_time) * 1.0)
def main():
    """main function"""
    print("forward_execution_time:{},backward_execution_time:{},backward_execution_time2:{}.".format(forward_execution_time,backward_execution_time,backward_execution_time2))
    config = {
        "pp_size": len(forward_execution_time),
        # add arbitrary virtual stage situation, (x, list[i] ∈ {1,0} ) -> extend to x times stages and list[i]=0 represents reversed order of stages
        "virtual_stage": (2, [1, 0]),
        "num_microbatches": num_micro_batch,
        "forward_execution_time": forward_execution_time,
        "backward_execution_time": backward_execution_time,
        "backward_execution_time2": backward_execution_time2,
        # stratiges: "strict", "double_interleaving", "full_interleaving",
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [8 for _ in range(len(forward_execution_time))],
    }

    simulator = Simulator(config)
    simulator.run()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        set_execution_time(int(sys.argv[-1]))
    elif len(sys.argv) == 3:
        set_execution_time(int(sys.argv[-2]), int(sys.argv[-1]))
    elif len(sys.argv) == 4:
        set_execution_time(int(sys.argv[-3]), int(sys.argv[-2]), int(sys.argv[-1]))
    
    main()
