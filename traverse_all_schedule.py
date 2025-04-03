import os
import time
from simulator.abstract.Pipeline import *
from simulator.abstract.mutils import *

def check_standard_zbv_conditions():
    if gpc["EMB_F_TIME"]!=0:
        print("Required to ignore EMB layers.")
        return False
    if gpc["HEAD_B_TIME"]!=0 or gpc["HEAD_F_TIME"]!=0 or gpc["HEAD_W_TIME"]!=0:
        print("Required to ignore HEAD layers.")
        return False
    if gpc["CE_B_TIME"]!=0 or gpc["CE_F_TIME"]!=0 or gpc["CE_W_TIME"]!=0:
        print("Required to ignore CE layers.")
    if gpc["F_TIME"]==gpc["B_TIME"]==gpc["W_TIME"]:
        return True
    else:
        print("Required F=B=W")
        return False

def clear_old_file(filepath:str):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Create directory:{}".format(directory))

    if os.path.exists(filepath):
        os.remove(filepath)
        print("delete file:{}".format(filepath))

def clear_old_files():
    clear_old_file(SCH_FILE_PATH)
    clear_old_file(TEMP_PLA_PATH)
    clear_old_file(TEMP_RES_PATH)
    for did in range(DEVICE_NUM):
        workload_stat_filepath = f"schedule_results/workload_statistics/device{did}.txt"
        clear_old_file(workload_stat_filepath)
        memory_record_filepath = f"schedule_results/memory/device{did}.txt"
        clear_old_file(memory_record_filepath)

def generate_parallel_config(num_gpu:int):
    assert num_gpu > 0, "Number must be positive."
    assert (num_gpu & (num_gpu - 1)) == 0, "Number is not a power of two."
    parallel_config = []
    for tp in [4, 8]:
        for pp in [4, 8, 16]:
            if tp * pp > num_gpu:
                continue 
            dp = int(num_gpu / tp / pp)
            parallel_config.append((pp, tp, dp))
    return parallel_config

def set_env_for_zbv(step):
    gpc["SCHEDULE_METHOD"] = Schedule.ZBV
    gpc["STAGE_PLACEMENT"] = Placement.WAVELIKE
    gpc["CHUNK_NUM"] = 2
    gpc["SPLIT_BACKPROP"] = True

    if step == 1:
        gpc["RUN_SCHEDULE"] = False
        gpc["RUN_STANDARD_ZBV"] = True
        gpc["EMB_F_TIME"] = 0
        gpc["HEAD_F_TIME"] = 0
        gpc["HEAD_B_TIME"] = 0
        gpc["HEAD_W_TIME"] = 0
        gpc["CE_F_TIME"] = 0
        gpc["CE_B_TIME"] = 0
        gpc["CE_W_TIME"] = 0
    elif step == 2:
        gpc["RUN_SCHEDULE"] = True
        gpc["RUN_STANDARD_ZBV"] = False
        gpc["EMB_F_TIME"] = 0
        gpc["HEAD_F_TIME"] = 2
        gpc["HEAD_B_TIME"] = 2
        gpc["HEAD_W_TIME"] = 2
        gpc["CE_F_TIME"] = 2
        gpc["CE_B_TIME"] = 2
        gpc["CE_W_TIME"] = 0
    else:
        raise ValueError("Wrong step.")

def set_env(sim_config):
    for k in sim_config:
        gpc[k] = sim_config[k]

def run_schedule(draw=False):
    simulator = PipelineScheduler(pipeline_idx=0, run_schedule=gpc["RUN_SCHEDULE"])
    if gpc["RUN_STANDARD_ZBV"] and not gpc["RUN_SCHEDULE"] and gpc["SCHEDULE_METHOD"] == Schedule.ZBV:
        if check_standard_zbv_conditions():
            simulator.run_pipeline_parallelism()
            simulator.result2file()
            print("ZBV F=B=W Schedule saved to data.txt.")
            return -1, True
    simulator.run_pipeline_parallelism()
    simulator.show_mem_usage(show_all=True)
    if draw:
        simulator.draw()
    print("Time:{}.".format(simulator.last_workload.end_time))
    return simulator.last_workload.end_time, simulator.finish_flag

def main():
    clear_old_files()
    basic_f = 4
    basic_b = 4
    basic_w = 4
    schedules = [Schedule.ZBV]
    res = []
    for num_gpus in [32, 64]:
        parallel_configs = generate_parallel_config(num_gpu=num_gpus)
        for (pp, tp, dp) in parallel_configs:
            for model_layer in [48, 64, 80]:
                for schedule in schedules:
                    fbw = False
                    if schedule in (Schedule.STANDARD_1F1B, Schedule.STANDARD_ZBH1):
                        chunks = [1]
                        if schedule == Schedule.STANDARD_ZBH1:
                            fbw = True
                    elif schedule == Schedule.STANDARD_INTERLEAVED:
                        chunks = [model_layer // pp]
                        # chunks = [2]
                    elif schedule == Schedule.ZBV:
                        chunks = [2]
                        fbw = True
                        set_env_for_zbv(step=1)
                    else:
                        chunks = [model_layer // pp]
                        fbw = True
                    for chunk in chunks:
                        sim_config = {
                            "TP_SIZE":tp,
                            "PP_SIZE":pp,
                            "ZERO_SIZE":dp,
                            "DEVICE_NUM":pp,
                            "LAYER_NUM":model_layer,
                            "CHUNK_NUM":chunk,
                            "SPLIT_BACKPROP": fbw,
                            "WORKLOAD_TYPE_NUM": 3 if fbw else 2,
                            "MICRO_BATCH_NUM": pp * 4,
                            "SCHEDULE_METHOD": schedule,
                            "STAGE_NUM": pp * chunk,
                            "F_TIME":basic_f,
                            "B_TIME":basic_b if fbw else basic_b + basic_w,
                            "W_TIME":basic_w,
                            "MAX_ACTIVATION_COUNTS": pp, 
                        }
                        clear_old_files()
                        set_env(sim_config=sim_config)
                        time_cost, success = run_schedule(draw=False)
                        if schedule == Schedule.ZBV:
                            set_env_for_zbv(step=2)
                            time_cost, success = run_schedule(draw=False)
                        res.append((num_gpus, pp, tp, dp, model_layer, chunk, schedule.name, time_cost, success))
    
    content = ""
    for r in res:
        print(r)
        content += str(r)+"\n"
    save_to_file(f"traverse_res_{Schedule.ZBV.name}.txt", content ,'w')
if __name__ == "__main__":
    print(Gradient.INPUT * 4)
    # start_time = time.time()
    run_schedule(draw=True)
    # end_time = time.time()
    # print(f"Schedule generation time:{end_time-start_time}")