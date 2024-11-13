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
    
    def add_stage(self, stage_id: int) -> None:
        stage = Stage(stage_id=stage_id, memory_usage=0, activation_memory=1)
        self.stages[stage.stage_id] = stage

    def execute_workload(self) -> None:
        if self.state == Device.IDLE:
            for sid in self.stages:
                (self.proc_sid, self.proc_mid, self.proc_wlt) = self.stages[sid].execute_workload()
                if self.proc_mid != -1:
                    self.state = Device.BUSY
        elif self.state == Device.BUSY:
            if self._check_status():
                self.state = Device.IDLE
                self.update_memory_usage()

    def _check_status(self):
        if GLOBAL_TIME >= self.stages[self.proc_sid].workloads[self.proc_mid][self.proc_wlt].end_time:
            return True
        return False

    def update_memory_usage(self) -> int:
        """计算设备中所有阶段的总内存开销"""
        return sum(stage.memory_usage for stage in self.stages.values())

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}, current_stage={self.current_stage_id}, current_microbatch={self.current_microbatch_id})"