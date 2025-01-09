from .Stage import *
import queue 

def get_required_memory(stage_id, layer_num, workload_type, workload_type_num = 3, layer_wise=True, recomp=None):
        assert workload_type_num == WORKLOAD_TYPE_NUM, "Mismatch in number of workload type"
        if workload_type ==WorkloadType.F:
            required_memory = Activation.FULL_LAYER * layer_num 
            if recomp: 
                required_memory = Activation.FULL_LAYER * layer_num * (1 - recomp)
            if layer_wise and stage_id == LAYER_NUM - 2:
                required_memory = Activation.LOSS
            elif not layer_wise and stage_id == STAGE_NUM - 1:
                    required_memory += Activation.LOSS
        else:
            if workload_type_num == 3:
                if workload_type == WorkloadType.B:
                    required_memory = Gradient.INPUT * layer_num 
                    if recomp:
                        required_memory = Gradient.INPUT * layer_num + Activation.FULL_LAYER * layer_num * recomp
                elif workload_type == WorkloadType.W:
                    required_memory = Gradient.PARAMETER * layer_num
                else:
                    raise Exception("Unsupported workload type {}".format(workload_type))
            elif workload_type_num == 2:
                if workload_type == WorkloadType.B:
                    required_memory = (Gradient.INPUT + Gradient.PARAMETER) * layer_num
                    if recomp:
                        required_memory = (Gradient.INPUT + Gradient.PARAMETER) * layer_num + Activation.FULL_LAYER * recomp
                else:
                    raise Exception("Unsupported workload type {}".format(workload_type))
            else:
                raise Exception("Wrong workload_type_num")

        if layer_wise:
            if stage_id == 0 or stage_id == LAYER_NUM - 1:
                required_memory = 0
        return required_memory

class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int, max_activation_counts: int, nmb:int, static_schedule: list = None):
        self.device_id = device_id
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.proc_workload: Workload = None
        self.optimizer_mem_usage: int = OPTIMIZER_MEMORY / (PP_SIZE * TP_SIZE)
        self.current_mem_usage: int = self.optimizer_mem_usage
        self.nmb: int = nmb
        self.max_activation_counts: int = max_activation_counts
        self.mem_usage_record: dict[int, int] = {}
        self.static_schedule: list[str] = static_schedule
        self.next_workload_idx: int = 0
        self.workload_type_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
        self.last_workload_type = None
        self.total_layers = 0
        self.microbatch_schedule_range = range(0,self.nmb)
        # To avoid simulator failure, memory need to be preserved.
        # Record each mid, if mid already started, ensure the memory is 
        # sufficient to complete the remaining workload.
        # Example:
        # mid = 0 begin, preserved_memory = max(F + B + W, F, B, W)
        # mid = 0 F finish, preserved_memory = max(B + W, B, W)
        # mid = 0 B finish, preserved_memory = W
        # mid = 0 W finish, switch to monitor mid = 1
        self.next_mid = 0
        self.released_workloads = []

    def get_available_f_workloads(self):
        current_mem_usage = self.get_memory_usage()
        return (GPU_MAX_MEM - current_mem_usage - Gradient.INPUT - Gradient.PARAMETER) // Activation.FULL_LAYER - 2

    def exist_executable_workload(self, workload_type):
        for stage_id in self.stages:
            workloads = self.stages[stage_id].workloads
            for mid in range(self.nmb):
                if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable():
                    return True
        return False

    def show_stages(self, detail_info=False):
        for sid in self.stages:
            print("Stage {} recomp={} on Device {}".format(sid, self.stages[sid].recomp, self.device_id))
            if detail_info:
                for mid in self.stages[sid].workloads:
                    if mid == 10:
                        for wlt in self.stages[sid].workloads[mid]:
                            print(self.stages[sid].workloads[mid][wlt])

    def add_stage(self, stage_id: int, recomp:bool = False) -> None:
        layer_per_stage = LAYER_NUM // STAGE_NUM
        stage_type = StageType.LAYERS
        basic_memory = 0
        if SchedulePriority.Layerwise:
            layer_per_stage = 1 
            if stage_id == 0:
                stage_type = StageType.EMBD
            elif stage_id == LAYER_NUM + 1:
                stage_type = StageType.HEAD
                basic_memory += HEAD_MEMORY
            elif stage_id == LAYER_NUM + 2:
                stage_type = StageType.CE
            else:
                stage_type = StageType.LAYER
                basic_memory += LAYER_MEMORY
        stage = Stage(
                device_id=self.device_id, 
                stage_id=stage_id,
                # memory_usage=0, 
                memory_usage=basic_memory, 
                stage_type=stage_type,
                recomp=recomp,
            )
        self.stages[stage.stage_id] = stage
        self.total_layers+=layer_per_stage

    def update_constraints(self, constraint):
        for sid in self.stages:
            self.stages[sid].update_constraints(constraint=constraint)
    
    def execute_workload(self) -> None:
        if self.state == Device.IDLE:
            if SCHEDULE_METHOD == SchedulePriority.GREEDY_v1:
                for workload_type in self.workload_type_priority_order:
                    if self.current_mem_usage == self.max_activation_counts and workload_type == WorkloadType.F:
                        workload_type = WorkloadType.W
                    for mid in range(self.nmb):
                        for sid in self.stages:
                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload
            elif SCHEDULE_METHOD == SchedulePriority.GREEDY_v2:
                now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
                if self.last_workload_type == WorkloadType.B:
                    now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
                elif self.last_workload_type == WorkloadType.F:
                    now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                
                for workload_type in now_workload_priority_order:
                    if self.current_mem_usage == self.max_activation_counts and workload_type == WorkloadType.F:
                        workload_type = WorkloadType.W
                    for mid in range(self.nmb):
                        for sid in self.stages:
                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                self.last_workload_type = workload_type
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload
            elif SCHEDULE_METHOD in (SchedulePriority.ONE_F_ONE_B, SchedulePriority.ZBH1, SchedulePriority.ZBV, SchedulePriority.INTERLEAVED):
                if self.next_workload_idx == len(self.stages) * self.nmb * WORKLOAD_TYPE_NUM:
                    return None
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx]
                proc_workload = self.stages[workload_sid].execute_workload(mid=workload_mid,workload_type=workload_type)
                if proc_workload:
                    self.proc_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.next_workload_idx += 1
                    return proc_workload
            elif SCHEDULE_METHOD == SchedulePriority.Layerwise:
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                for workload_type in now_workload_priority_order:
                    # for mid in range(self.nmb):
                    for mid in self.microbatch_schedule_range:
                        for sid in self.stages:
                            required_memory = get_required_memory(
                                stage_id=sid, 
                                layer_num=1,
                                workload_type=workload_type,
                                workload_type_num=WORKLOAD_TYPE_NUM, 
                                layer_wise=True,
                                recomp=self.stages[sid].recomp,
                            )
                            workload_type = self._reset_workload_type(
                                workload_type=workload_type,
                                required_memory=required_memory,
                                current_mem_usage=self.current_mem_usage,
                                max_memory=GPU_MAX_MEM,
                                workload_situations=None
                            )
                            # if workload_type == WorkloadType.F:
                            #     if self.get_available_f_workloads() == 0:
                            #         mid = self.released_workloads[self.next_mid]

                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                # if workload_type == WorkloadType.F:
                                #     if self.get_available_f_workloads() > 0:
                                #         if mid not in self.released_workloads:
                                #             self.released_workloads.append(mid)
                                #             if self.device_id == 0: 
                                #                 print("Add {}".format(mid))
                                #     elif self.get_available_f_workloads() == 0:
                                #         self.next_mid += 1
                                #         if self.device_id == 0: 
                                #             print("Exe {}".format(mid))
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload
            else:
                print("Schedule Not Supported")
        return None

    def _reset_workload_type(self, workload_type, required_memory, current_mem_usage, max_memory, workload_situations):
        if workload_type == WorkloadType.F:
            if current_mem_usage + required_memory >= max_memory - Gradient.INPUT - Gradient.PARAMETER:
                workload_type = WorkloadType.B
            if current_mem_usage + required_memory >= max_memory - Gradient.PARAMETER:
                workload_type = WorkloadType.W
        return workload_type


    def _finish_proc_workload(self) -> bool: 
        if self.state == Device.BUSY and GET_TIME() >= self.proc_workload.end_time:
            return True
        return False

    def update_memory_usage(self) -> int:
        if self.proc_workload.state == Workload.IN_PROGRESS and self.proc_workload.workload_type in (WorkloadType.F, WorkloadType.B):
            self.stages[self.proc_workload.stage_id].update_memory_usage(workload=self.proc_workload)
        elif self.proc_workload.state == Workload.COMPLETED and self.proc_workload.workload_type == WorkloadType.W:
            self.stages[self.proc_workload.stage_id].update_memory_usage(workload=self.proc_workload)
            
        self.current_mem_usage = self.optimizer_mem_usage + sum(stage.memory_usage for stage in self.stages.values())
        self.mem_usage_record[(self.proc_workload.start_time,self.proc_workload.end_time)] = self.current_mem_usage
    
    def get_memory_usage(self) -> int:
        return self.current_mem_usage

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}, current_stage={self.current_stage_id}, current_microbatch={self.current_microbatch_id})"