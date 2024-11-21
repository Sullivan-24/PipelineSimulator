from .Stage import *
class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.proc_workload: Workload = None
        self.current_mem_usage: int = 0
        self.mem_usage_record: dict[int, int] = {}
    
    def show_stages(self):
        for sid in self.stages:
            print("Stage {} on Device {}".format(sid, self.device_id))

    def add_stage(self, stage_id: int) -> None:
        stage = Stage(
                device_id=self.device_id, 
                stage_id=stage_id, 
                memory_usage=0, 
                activation_memory_increment=1
            )
        self.stages[stage.stage_id] = stage

    def update_constraints(self, constraint):
        for sid in self.stages:
            self.stages[sid].update_constraints(constraint=constraint)

    def execute_workload(self) -> None:
        proc_info = [None, None]
        if self.state == Device.BUSY:
            if self._finish_proc_workload():
                self.proc_workload.complete()
                self.update_memory_usage()
                proc_info[0] = self.proc_workload
                self.state = Device.IDLE

        elif self.state == Device.IDLE:
            for workload_type in WorkloadType:
                for mid in range(MICRO_BATCH_NUM):
                    for sid in self.stages:
                        proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                        if proc_workload:
                            proc_info[1] = proc_workload
                            self.proc_workload = proc_workload
                            self.update_memory_usage()
                            self.state = Device.BUSY
                            return proc_info
                        
        else:
            print("Wrong Device Status.")

        return proc_info

    def _finish_proc_workload(self) -> bool: 
        if GET_TIME() >= self.proc_workload.end_time:
            return True
        return False

    def update_memory_usage(self) -> int:
        if self.proc_workload.state == Workload.IN_PROGRESS and self.proc_workload.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            self.stages[self.proc_workload.stage_id].update_memory_usage(workload=self.proc_workload)
        elif self.proc_workload.state == Workload.COMPLETED and self.proc_workload.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.stages[self.proc_workload.stage_id].update_memory_usage(workload=self.proc_workload)
            
        self.current_mem_usage = sum(stage.memory_usage for stage in self.stages.values())
        self.mem_usage_record[GET_TIME()] = self.current_mem_usage
    
    def get_memory_usage(self) -> int:
        return self.current_mem_usage

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}, current_stage={self.current_stage_id}, current_microbatch={self.current_microbatch_id})"