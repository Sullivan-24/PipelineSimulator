from simulator.abstract.Device import *
from simulator.abstract.mutils import *
from simulator.painter import MultiPipelinePainter as MPP
from simulator.abstract.Pipeline import PipelineScheduler
import cProfile
import pstats

class Executor:

    def __init__(self, dp_size, nmb_per_dp: list = None) -> None:
        self.time           = 0
        self.finish_flag    = False
        self.dp_size        = dp_size
        self.nmb_per_dp     = [gpc["MICRO_BATCH_NUM"] for _ in range(dp_size)] if not nmb_per_dp else nmb_per_dp
        self.mid_offsets    = [0] + [sum(self.nmb_per_dp[:i]) for i in range(1, dp_size+1)]
        self.pipelines      = [
            PipelineScheduler(
                pipeline_idx=dp_idx, 
                nmb=self.nmb_per_dp[dp_idx], 
                mid_offset=self.mid_offsets[dp_idx], 
                executor=self
            ) for dp_idx in range(dp_size)
        ]
    
    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time
    
    def get_total_workload_count(self):
        count = 0
        for pipeline in self.pipelines:
            count += pipeline.total_workload
        return count

    def update_constraints_across_dp(self, time):
        for pipeline in self.pipelines:
            for device in pipeline.devices:
                if device.current_workload and time >= device.current_workload.end_time:
                    finished_mid = device.current_workload.mid
                    for p in self.pipelines:
                        for d in p.devices:
                            if d.did == device.did or p.pipeline_idx == pipeline.pipeline_idx:
                                continue # only update constraints on other devices or pipelines
                            if finished_mid in d.held_mids:
                                d.update_constraints_within_device(time, constraint=device.current_workload)

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True):
        self.reset_time()
        workloads_failure = {}
        workloads_slow = {}

        mid_done = [[] for _ in range(len(FAILURE_PP_ID))]
        pop_num_failure = [_ for _ in range(len(FAILURE_PP_ID))]
        pop_num_slow = [_ for _ in range(len(HETER_PP_ID))]
        all_DP = set([_ for _ in range(self.dp_size)])
        transfer_info = [[[[]for _ in range(PP_SIZE)]for _ in range(self.dp_size)] for _ in range(self.dp_size)]
        # transfer_info[source_dp_rank].append({"microbatch_ids":[2,4,7], "stage_id":2, "dst_dp_rank":dst_dp_rank})
        while self.get_time() <= time_limit and not self.finish_flag:
            success_count = 0
            latest_workloads_dp = [[] for _ in range(self.dp_size)]
            exec_f_num_dp = [[] for _ in range(self.dp_size)]
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                exec_f_num_ = pipeline.check_device_status(time=self.time)
                success_count += pipeline.get_completed_workload_count()
                exec_f_num_dp[pipeline.pipeline_idx] = exec_f_num_
                # if self.get_time() == 0:
                #     if pipeline.pipeline_idx == 0:
                #         workloads = pipeline.pop_workload(mid_group=list(range(pipeline.nmb)),did_group=[2])
                # if self.get_time() == 0:
                #     if pipeline.pipeline_idx == 1:
                #         pipeline.insert_workload(workloads=workloads,did_group=[2])
                # if self.get_time() == 666:
                #     if pipeline.pipeline_idx == 1:
                #         workloads = pipeline.pop_workload(mid_group=[2],did_group=[2])
                # if self.get_time() == 666:
                #     if pipeline.pipeline_idx == 0:
                #         pipeline.insert_workload(workloads=workloads,did_group=[2])
            if gpc["DP_Transfer"]:
                if FAILURE_DEVICE:
                    failure_dids = FAILURE_PP_ID
                    failure_dps = FAILURE_DP_ID
                    other_dps = list(all_DP-set(FAILURE_DP_ID))
                    # if self.get_time() == 0:
                    for pipeline in self.pipelines:
                        if pipeline.pipeline_idx == failure_dps[0] :#and self.get_time() == pop_time:
                            if pop_num_failure < NMB_PER_DP[other_dps[0]]:
                                other_dp_exec_f_num = exec_f_num_dp[other_dps[0]][failure_dids[0]]
                                exec_f_num_diff = other_dp_exec_f_num-pop_num_failure
                                if exec_f_num_diff >= 0:
                                    mid_ = other_dp_exec_f_num+pipeline.mid_offset
                                    if mid_ not in mid_done:
                                        print(f"pop_mid_failure:{mid_} in PP{failure_dids[0]}, from DP{failure_dps[0]} to DP{other_dps[0]}")
                                        transfer_info[failure_dps[0]][other_dps[0]][failure_dids[0]].append(mid_)
                                        workloads_failure = pipeline.pop_workload(mid_group=[mid_],did_group=[failure_dids[0]])#pop 下一个f
                                        mid_done.append(mid_)
                                        pop_num_failure+=1
                            else:
                                if len(mid_done) == NMB_PER_DP[failure_dps[0]]:
                                    break
                                for mid_ in range(NMB_PER_DP[other_dps[0]],NMB_PER_DP[failure_dps[0]]):
                                    mid_ += pipeline.mid_offset
                                    if mid_ not in mid_done:
                                        print(f"pop_mid_failure:{mid_} in PP{failure_dids[0]}, from DP{failure_dps[0]} to DP{other_dps[0]}")
                                        transfer_info[failure_dps[0]][other_dps[0]][failure_dids[0]].append(mid_)
                                        workloads_failure = pipeline.pop_workload(mid_group=[mid_],did_group=[failure_dids[0]])#pop 下一个f
                                        mid_done.append(mid_)
                                        pop_num_failure+=1

                    for pipeline in self.pipelines:
                        if pipeline.pipeline_idx == other_dps[0]:#and self.get_time == insert_time:
                            if workloads_failure is not None:
                                pipeline.insert_workload(workloads=workloads_failure, did_group=[failure_dids[0]])
                                workloads_failure = None
                if HETER_DEVICE_Transfer:
                    slow_dids = HETER_PP_ID
                    slow_dps = HETER_DP_ID
                    fast_dps = list(all_DP-set(HETER_DP_ID))
                    for slow_index in range(len(slow_dids)):
                        slow_did =  slow_dids[slow_index]
                        slow_dp = slow_dps[slow_index]
                        fast_dp = None

                        latest_exec_f_num = [0 for _ in range(self.dp_size)]
                        for dp_rank in range(self.dp_size):
                            latest_exec_f_num[dp_rank] = exec_f_num_dp[dp_rank][slow_did]

                        min_f_num = latest_exec_f_num[slow_dp]
                        max_f_num = max(latest_exec_f_num)

                        for dp_index in range(self.dp_size):
                            if latest_exec_f_num[dp_index]<min_f_num:
                                min_f_num = latest_exec_f_num[dp_index]
                                slow_dp = dp_index
                            elif latest_exec_f_num[dp_index] == max_f_num:
                                fast_dp = dp_index
                        assert fast_dp is not None   
                        
                        if min_f_num+pop_num_slow[slow_index]>= max_f_num-1:
                            continue
                        elif min_f_num+pop_num_slow[slow_index] < NMB_PER_DP[slow_dp]:
                            print("diff")
                            for pipeline in self.pipelines:
                                if pipeline.pipeline_idx == slow_dp :#and self.get_time() == pop_time:
                                    pop_mid_slow = min_f_num+pop_num_slow[slow_index]+pipeline.mid_offset
                                    
                                    workloads_slow = pipeline.pop_workload(mid_group=[pop_mid_slow],did_group=[slow_did])#pop 下一个f
                                    # pop_time = None
                                    pop_num_slow[slow_index] += 1
                                    print(f"pop_mid_slow:{pop_mid_slow} in PP{slow_did}, from DP{slow_dp} to DP{fast_dp}")
                                    transfer_info[slow_dp][fast_dp][slow_did].append(pop_mid_slow)
                            for pipeline in self.pipelines:
                                if pipeline.pipeline_idx == fast_dp :#and self.get_time == insert_time:
                                    if workloads_slow is not None:
                                        pipeline.insert_workload(workloads=workloads_slow, did_group=[slow_did])
                                        workloads_slow = None
                                        # insert_time = None                
            self.finish_flag = True if success_count == self.get_total_workload_count() else False
            self.update_time()
        if show_success:
            if self.finish_flag:
                print("Success")
            else:
                print("Fail")

        write_mutex = False
        opt_put_info = [[]for _ in range(self.dp_size)]
        for slow_dp_index in range(self.dp_size):
            for fast_dp_index in range(self.dp_size):
                for slow_did_index in range(PP_SIZE):
                    transfer_mid = transfer_info[slow_dp_index][fast_dp_index][slow_did_index]
                    if len(transfer_mid)>0:
                        write_mutex = True
                        opt_put_info[slow_dp_index].append({"microbatch_ids":transfer_mid, "stage_id":slow_did_index, "dst_dp_rank":fast_dp_index})#TODO slow_did_index == stage_id just in single chunk
        if write_mutex:
            save_to_file(f"schedule_results/transfer_info.txt", str(opt_put_info), 'w')

    def draw(self) -> None:
        res_all_dp = {}
        res_all_dp["res"]={}
        res_all_dp["painter_conf"]={}
        for dp_idx in range(self.dp_size):
            pipeline = self.pipelines[dp_idx]
            fwd_time, iwd_time, pwd_time = pipeline.get_workloadload_duration()
            res = {}
            for key in pipeline.results:
                if key.startswith(("f_","b_","w_","r_")):
                    res[key] = pipeline.results[key]
            painter_conf = {
                "device_num": pipeline.device_num,
                "devices": pipeline.placement,
                "stage_num": pipeline.stage_num if not gpc["HEAD_DP"] else pipeline.stage_num + 1,
                "pp_height": gpc["PP_HEIGHT"],
                "pp_align": gpc["PP_ALIGN"],
                "pixel_base": gpc["PIXEL_BASE"],
                "nmb": pipeline.nmb,
                "mid_offset": pipeline.mid_offset,
                "forward_length": fwd_time,
                "backward_length": iwd_time,
                "backward_length2": pwd_time,
                "comm_length": [gpc["COMM_TIME"] for _ in range(pipeline.stage_num)],
            }
            res_all_dp["res"][dp_idx]=res
            res_all_dp["painter_conf"][dp_idx]=painter_conf
        MPP(res_all_dp["painter_conf"]).draw(res_all_dp["res"])

if __name__ == "__main__":
    # Example
    # executor = Executor(dp_size=4, nmb_per_dp=[15, 12, 20, 17])
    executor = Executor(dp_size=2,nmb_per_dp = NMB_PER_DP)
    
    if gpc["PROFILE_GENERATION"]:
        profiler = cProfile.Profile()
        profiler.enable()
        executor.run_all_dp()
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats("cumtime")
        stats.print_stats(20)  # 打印前 10 个耗时函数
    else:
        executor.run_all_dp()

    executor.draw()