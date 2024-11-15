from abstract.Device import Device
from abstract.Workload import Workload
from abstract.mutils import *
import os
import sys
sys.path.append("/home/PJLAB/guojihu/下载/PPSimulator")
import Simulator.PipelineSimulator.simulator.painter as SP

class PipelineScheduler:
    def __init__(self) -> None:
        self.results = {}
        self.devices: list[Device] = []
        self.dsa = []
        self._init_stage()

    def _init_stage(self):
        for did in range(DEVICE_NUM):
            device = Device(device_id = did)
            device.add_stage(did)
            device.add_stage(STAGE_NUM - 1 - did)
            self.devices.append(device)
            device.show_stages()

        for did in range(DEVICE_NUM):
            self.dsa.append(self.devices[did].stages.keys())
    
    def show_record(self):
        for k in self.results:
            print(k, self.results[k])

    def update_constraints(self, constraint):
        for device in self.devices:
            device.update_constraints(constraint=constraint)

    def record_workload(self, workload: Workload):
        if workload:
            wlt = workload.workload_type.value.lower()
            mid = workload.microbatch_id
            sid = workload.stage_id
            k = '{}_{}_{}'.format(wlt,mid,sid)
            self.results[k] = workload.start_time

    def run_pipeline_parallelism(self, time_limit = 1000):
        while GET_TIME() <= time_limit:
            UPDATE_TIME_FLAG = True
            for device in self.devices:
                (completed_workload, processing_workload) = device.execute_workload()
                self.record_workload(processing_workload)
                if completed_workload:
                    self.update_constraints(constraint=completed_workload)
                    UPDATE_TIME_FLAG = False
                    break
            if UPDATE_TIME_FLAG:
                UPDATE_TIME()

    def show_mem_usage(self):
        for device in self.devices:
            print("Device {} mem usage:".format(device.device_id))
            for t, mem_record in device.mem_usage_record.items():
                print("Time {}, mem = {}.".format(t, mem_record))

    def draw(self) -> None:
        # 绘制结果的逻辑
        painter_conf = {
            "device_size": DEVICE_NUM,
            "devices": self.dsa,
            "pp_size": STAGE_NUM,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 2,
            "num_real_microbatches": MICRO_BATCH_NUM,
            "forward_length": [FPW_TIME for _ in range(STAGE_NUM)],
            "backward_length": [IGW_TIME for _ in range(STAGE_NUM)],
            "backward_length2": [PGW_TIME for _ in range(STAGE_NUM)],
            "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
        }

        SP.SchedulingPainter(painter_conf).draw(self.results)