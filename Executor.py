from simulator.abstract.Device import *
from simulator.abstract.mutils import *
from simulator.painter import MultiPipelinePainter as MPP
from simulator.abstract.Pipeline import PipelineScheduler

class Executor:

    def __init__(self, dp_size) -> None:
        self.time = 0
        self.finish_flag = False
        self.dp_size = dp_size
        self.pipelines = [PipelineScheduler(pipeline_idx=dp_idx, time=0*dp_idx, executor=self) for dp_idx in range(dp_size)]
    
    def update_time(self):
        self.time += 1

    def reset_time(self):
        self.time = 0

    def get_time(self):
        return self.time
    
    def update_constraints(self, time):
        for pipeline in self.pipelines:
            for device in pipeline.devices:
                if device.proc_workload and time >= device.proc_workload.end_time:
                    for p in self.pipelines:
                        for d in p.devices:
                            d.update_constraints(time, constraint=device.proc_workload)

    def move_mb(self, send_pid, recieve_pid):
        send_pipeline = self.pipelines[send_pid]
        recieve_pipeline = self.pipelines[recieve_pid]

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True):
        self.reset_time()
        workloads = {}
        pop_time = None
        insert_time = None
        pop_num = 0
        mid_done = []
        pop_workloads = []
        while self.get_time() <= time_limit and not self.finish_flag and not gpc["TERMINAL_FLAG"]:
            success_count = 0
            latest_workloads_dp = [[] for _ in range(self.dp_size)]
            exec_f_num_dp = [[] for _ in range(self.dp_size)]
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints(time=self.time)
                pipeline.execute_workload(time=self.time)
                success_count += pipeline.finish_flag
                latest_workloads, exec_f_num_ = pipeline.check_device_states()
                latest_workloads_dp[pipeline.pipeline_idx] = latest_workloads
                exec_f_num_dp[pipeline.pipeline_idx] = exec_f_num_
            if gpc["DP_Transfer"]:
                # for did in range(DEVICE_NUM):
                slow_did = HETER_PP_ID
                slow_dp = []#HETER_DP_ID
                fast_dp = []
                if gpc["Recycle"]:
                    slow_dp = HETER_DP_ID
                    fast_dp = NORM_DP_ID
                    # if self.get_time() == 0:
                    for pipeline in self.pipelines:
                        if pipeline.pipeline_idx == slow_dp[0] :#and self.get_time() == pop_time:
                                # for mid_ in range(MICRO_BATCH_NUM):
                                #     workloads = pipeline.pop_workload(mid_group=[mid_+ pipeline.mid_offset], did_group=slow_did)
                                #     pop_workloads.append(workloads)
                                fast_dp_exec_f_num = exec_f_num_dp[fast_dp[0]][slow_did[0]]-pop_num
                                if fast_dp_exec_f_num >= 0 and fast_dp_exec_f_num<MICRO_BATCH_NUM:
                                    mid_ = fast_dp_exec_f_num+pipeline.mid_offset
                                    if mid_ not in mid_done:
                                        workloads = pipeline.pop_workload(mid_group=[mid_],did_group=slow_did)#pop 下一个f
                                        mid_done.append(mid_)
                                        pop_num+=1
                                        print(mid_)
                    for pipeline in self.pipelines:
                        if pipeline.pipeline_idx == fast_dp[0]:#and self.get_time == insert_time:
                            # print(f"fast_dp_f_num:{exec_f_num_dp[fast_dp[0]][slow_did[0]]}, pop_num:{pop_num}, len(pop_workloads):{len(pop_workloads)}, mid_done:{mid_done}") 
                            # if exec_f_num_dp[fast_dp[0]][slow_did[0]]>=2*pop_num and len(pop_workloads)>0:
                            #     pipeline.insert_workload(workloads=pop_workloads.pop(0), did_group=slow_did)
                            #     mid_done.append(pop_num)
                            #     pop_num+=1
                            #     print(f"fast_dp_f_num:{exec_f_num_dp[fast_dp[0]][slow_did[0]]}, pop_num:{pop_num}, len(pop_workloads):{len(pop_workloads)}")  
                            if workloads is not None:
                                pipeline.insert_workload(workloads=workloads, did_group=slow_did)
                                workloads = None
                else:
                    for slow_did_ in slow_did:
                        latest_exec_f_num = [0 for _ in range(self.dp_size)]
                        latest_workloads_endtime = [None for _ in range(self.dp_size)]
                        for dp_rank in range(self.dp_size):
                            if latest_workloads_dp[dp_rank][slow_did_] is not None:
                                latest_workloads_endtime[dp_rank] = latest_workloads_dp[dp_rank][slow_did_].end_time
                            latest_exec_f_num[dp_rank] = exec_f_num_dp[dp_rank][slow_did_]
                        min_f_num = min(latest_exec_f_num)
                        max_f_num = max(latest_exec_f_num)
                        if min_f_num+pop_num >= max_f_num-1:
                            continue
                        elif min_f_num+pop_num < MICRO_BATCH_NUM:
                            print("diff")
                            for dp_rank_, did_exec_f_num in enumerate(latest_exec_f_num):
                                if did_exec_f_num == min_f_num:
                                    slow_dp.append(dp_rank_)
                                elif did_exec_f_num == max_f_num:
                                    fast_dp.append(dp_rank_)
                        # pop_time = latest_workloads_endtime[slow_dp[0]]
                        # insert_time = latest_workloads_endtime[fast_dp[0]]
                            for pipeline in self.pipelines:
                                if pipeline.pipeline_idx == slow_dp[0] :#and self.get_time() == pop_time:
                                    workloads = pipeline.pop_workload(mid_group=[min_f_num+pop_num+pipeline.mid_offset],did_group=[slow_did_])#pop 下一个f
                                    # pop_time = None
                                    pop_num += 1
                            for pipeline in self.pipelines:
                                if pipeline.pipeline_idx == fast_dp[0] :#and self.get_time == insert_time:
                                    if workloads is not None:
                                        pipeline.insert_workload(workloads=workloads, did_group=[slow_did_])
                                        workloads = None
                                        # insert_time = None

            # print(f"did:{did}, {latest_workloads_exec_f_num}")
            self.finish_flag = True if success_count == self.dp_size else False
            self.update_time()
        if show_success:
            if self.finish_flag:
                print("Success")
            else:
                print("Fail")

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
    executor = Executor(dp_size=2)
    executor.run_all_dp()
    executor.draw()
    