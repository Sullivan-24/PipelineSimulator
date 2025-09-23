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
        workloads = {}
        while self.get_time() <= time_limit and not self.finish_flag:
            success_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints_across_dp(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_status(time=self.time)
                success_count += pipeline.get_completed_workload_count()
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
            self.finish_flag = True if success_count == self.get_total_workload_count() else False
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
    executor = Executor(dp_size=1, nmb_per_dp=[15, 12, 20, 17])
    
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