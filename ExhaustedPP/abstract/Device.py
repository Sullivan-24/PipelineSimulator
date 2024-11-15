from .Stage import *
class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.proc_sid: int = -1
        self.proc_mid: int = -1
        self.proc_wlt: int = -1
    
    def show_stages(self):
        for sid in self.stages:
            print("Stage {} on Device {}".format(sid, self.device_id))

    def add_stage(self, stage_id: int) -> None:
        stage = Stage(stage_id=stage_id, memory_usage=0, activation_memory=1)
        self.stages[stage.stage_id] = stage

    def update_constraints(self, constraint):
        # constraint = (self.proc_sid, self.proc_mid, self.proc_wlt)
        for sid in self.stages:
            self.stages[sid].update_constraints(constraint=constraint)

    def execute_workload(self) -> None:
        proc_info = [None, None]
        if self.state == Device.BUSY:
            if self._check_status():
                self.stages[self.proc_sid].workloads[self.proc_mid][self.proc_wlt].complete()
                self.state = Device.IDLE
                self.update_memory_usage()
                proc_info[0] = (self.proc_sid, self.proc_mid, self.proc_wlt)

        if self.state == Device.IDLE:
            for workload_type in WorkloadType:
                for mid in range(MICRO_BATCH_NUM):
                    for sid in self.stages:
                        workload_info = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                        if workload_info:
                            (self.proc_sid, self.proc_mid, self.proc_wlt) = workload_info
                            proc_info[1] = workload_info
                            self.state = Device.BUSY
                            return proc_info
        return proc_info

    def _check_status(self):
        if GET_TIME() >= self.stages[self.proc_sid].workloads[self.proc_mid][self.proc_wlt].end_time:
            return True
        return False

    def update_memory_usage(self) -> int:
        """计算设备中所有阶段的总内存开销"""
        return sum(stage.memory_usage for stage in self.stages.values())

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}, current_stage={self.current_stage_id}, current_microbatch={self.current_microbatch_id})"