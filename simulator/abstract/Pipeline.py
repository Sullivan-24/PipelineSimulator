from .Device import Device
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter

class PipelineScheduler:


    def __init__(self, dsa = None) -> None:
        self.results = {}
        self.devices: list[Device] = []
        self.dsa = [] if not dsa else dsa 
        self._init_stage()

    def _init_stage(self, stage_type=Stage.VSHAPE):
        for did in range(DEVICE_NUM):
            device = Device(device_id = did)
            self.devices.append(device)

        if self.dsa:
            for did in range(DEVICE_NUM):
                for pid in self.dsa[did]:
                    self.devices[did].add_stage(pid)
        else:
            if stage_type == Stage.VSHAPE:
                for pid in range(STAGE_NUM):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid)
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid)
            elif stage_type == Stage.INTERLEAVED:
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid)

        for did in range(DEVICE_NUM):
            self.devices[did].show_stages()
            if len(self.dsa) < DEVICE_NUM:
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

    def check_workload_status(self):
        for device in self.devices:
            if device._finish_proc_workload():
                device.proc_workload.complete()
                self.update_constraints(constraint=device.proc_workload)
                device.update_memory_usage()
                device.state = Device.IDLE

    def execute_workload(self):
        for device in self.devices:
            processing_workload = device.execute_workload()
            self.record_workload(processing_workload)

    # def run_pipeline_parallelism(self, time_limit = 1000):
    #     while GET_TIME() <= time_limit:
    #         UPDATE_TIME_FLAG = True
    #         for device in self.devices:
    #             (completed_workload, processing_workload) = device.execute_workload()
    #             self.record_workload(processing_workload)
    #             if completed_workload:
    #                 self.update_constraints(constraint=completed_workload)
    #                 UPDATE_TIME_FLAG = False
    #                 break
    #         if UPDATE_TIME_FLAG:
    #             UPDATE_TIME()
            
    def run_pipeline_parallelism(self, time_limit = 1000):
        while GET_TIME() <= time_limit:
            self.check_workload_status()
            self.execute_workload()
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
            "num_microbatches": MICRO_BATCH_NUM,
            "forward_length": [FPW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "backward_length": [IGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "backward_length2": [PGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
        }

        SchedulingPainter(painter_conf).draw(self.results)