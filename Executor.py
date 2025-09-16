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

    def run_all_dp(self, time_limit = gpc["TIME_LIMIT"], show_utilization=True, show_mem=True, show_success=True):
        self.reset_time()
        workloads = {}
        while self.get_time() <= time_limit and not self.finish_flag and not gpc["TERMINAL_FLAG"]:
            success_count = 0
            for pipeline in self.pipelines:
                pipeline.check_workload_status(time=self.time)
                self.update_constraints(time=self.time)
                pipeline.execute_workload(time=self.time)
                pipeline.check_device_states()
                success_count += pipeline.finish_flag
                # if self.get_time() == 250:
                #     if pipeline.pipeline_idx == 2:
                #         workloads = pipeline.pop_workload(mid_group=[18],did_group=[2])
                # if self.get_time() == 250:
                #     if pipeline.pipeline_idx == 3:
                #         pipeline.insert_workload(workloads=workloads,did_group=[2])
                # if self.get_time() == 666:
                #     if pipeline.pipeline_idx == 3:
                #         workloads = pipeline.pop_workload(mid_group=[18],did_group=[2])
                # if self.get_time() == 666:
                #     if pipeline.pipeline_idx == 2:
                #         pipeline.insert_workload(workloads=workloads,did_group=[2])
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
    