from .Stage import *
from ..utils import print_to_file

def get_required_memory_by_workload(workload:Workload):
        stage_id = workload.sid
        layer_num = LAYER_NUM // DEVICE_NUM // CHUNK_NUM
        workload_type = workload.workload_type
        workload_type_num = WORKLOAD_TYPE_NUM
        layer_wise=LAYERWISE
        recomp=workload.recomp
        # wrapper of get required memory
        return get_required_memory(
            stage_id=stage_id,
            layer_num=layer_num,
            workload_type=workload_type,
            workload_type_num=workload_type_num,
            layer_wise=layer_wise,
            recomp=recomp
        )
# TODO rethinking the head memory cost
def get_required_memory(stage_id, layer_num, workload_type, workload_type_num = 3, layer_wise=True, recomp=None):
        assert workload_type_num == WORKLOAD_TYPE_NUM, "Mismatch in number of workload type!"
        if workload_type ==WorkloadType.F:
            required_memory = Activation.FULL * layer_num 
            if recomp:
                required_memory = layer_num * (Activation.FULL * (1 - recomp) + Activation.INPUT * recomp)
            if layer_wise and stage_id == LAYER_NUM + 1:
                required_memory = Activation.LOSS
            elif not layer_wise and stage_id == STAGE_NUM - 1:
                    required_memory += Activation.LOSS
        elif workload_type == WorkloadType.B:
            required_memory = Gradient.INPUT * layer_num
            if recomp:
                required_memory += layer_num * (Activation.FULL - Activation.INPUT) * recomp 
            if workload_type_num == 2:
                required_memory += Gradient.PARAMETER * layer_num
            if layer_wise and stage_id == LAYER_NUM + 1 and SPLIT_BACKPROP:
                required_memory = Gradient.HEAD_INPUT
        elif workload_type == WorkloadType.W:
            assert workload_type_num == 3, "Workload number error!"
            required_memory = Gradient.PARAMETER * layer_num
            if layer_wise and stage_id == LAYER_NUM + 1:
                required_memory = Gradient.HEAD_PARA
        else:
            raise ValueError("Unsupported workload type!")

        if layer_wise and (stage_id == 0 or stage_id == LAYER_NUM + 2):
            required_memory = 0
        return required_memory

class MemoryMonitor:
    def __init__(self, nmb:int, stages:dict, device_id:int, max_memory:float = GPU_MAX_MEM):
        self.did = device_id
        self.nmb = nmb
        self.stages = stages
        self.max_memory = max_memory
        self.tracing_workloads:list[Workload] = []
        self.workloads_reserved_mem:list[int] = [0 for _ in range(self.nmb)]
        self.safe_workload_mids = []

    def init_monitor(self):
        self.init_reserved_mem()

    def init_reserved_mem(self):
        workload_type = WorkloadType.F
        for mid in range(self.nmb):
            for sid in self.stages:
                workload = self.stages[sid].workloads[mid][workload_type]
                required_mem = get_required_memory_by_workload(workload)
                self.workloads_reserved_mem[mid] += required_mem
            # TODO rethinking the head memory cost
            self.workloads_reserved_mem[mid] += Gradient.INPUT + Gradient.PARAMETER + Gradient.HEAD_INPUT + Gradient.HEAD_PARA
            print(f"Device {self.did} Reserve {self.workloads_reserved_mem[mid]}G for mid {mid}")

    def trace_workload(self,workload:Workload):
        self.tracing_workloads.append(workload)
        required_mem = get_required_memory_by_workload(workload)
        self.workloads_reserved_mem[workload.mid] -= required_mem
        if self.workloads_reserved_mem[workload.mid] <= 0:
            self.safe_workload_mids.append(workload.mid)
    
    def is_executable_workload(self, workload:Workload, current_mem:float):
        required_mem = get_required_memory_by_workload(workload)
        safe_count = 0
        if workload.mid in self.safe_workload_mids:
            return True
        self.workloads_reserved_mem[workload.mid] -= required_mem
        for mid in range(self.nmb):
            if mid in self.safe_workload_mids:
                continue
            if self.workloads_reserved_mem[mid] + required_mem + current_mem <= self.max_memory:
                safe_count += 1
        # TODO what about F and B? in SPLIT or not?
        # memory friendly and critical
        if workload.workload_type == WorkloadType.W and required_mem + current_mem <= self.max_memory:
            safe_count += 1
        self.workloads_reserved_mem[workload.mid] += required_mem
        return safe_count > 0

class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int, max_activation_counts: int, nmb:int, static_schedule: list = None, memory_usage_constrain_rate: float = 1):
        self.did = device_id
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
        self.mid_traverse_order:list[int] = list(range(0,self.nmb))
        self.warmup_num_f = 0
        self.warmup_num_b = 0
        self.warmup_num_w = 0
        self.exe_num_f = 0
        self.exe_num_b = 0
        self.exe_num_w = 0
        self.order_balance = True
        self.next_workload_type = None
        self.memory_usage_constrain_rate = memory_usage_constrain_rate
        print(self.memory_usage_constrain_rate)
        self.executable_workloads:list[Workload] = []

        self.workload_type_traverse_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
        self.next_mid = 0
        self.released_workloads = []
        self.available_f_num = 0
        self.executing_workload_required_mem : list = [0 for _ in range(self.nmb)]
        self.executing_mid_idx = 0
        self.ok_flag = False

        self.memory_monitor = None
        self.wait_for_schedule = 0

        self.situations = 1
        self.mid_priority = [3 * CHUNK_NUM for _ in range(self.nmb)]

    def init_memory_monitor(self):
        self.memory_monitor = MemoryMonitor(self.nmb, self.stages, self.did, max_memory=GPU_MAX_MEM)
        self.memory_monitor.init_monitor()

    def init_required_mem_for_each_microbatch(self):
        for mid in range(self.nmb):
            for sid in self.stages:
                self.executing_workload_required_mem[mid] += Activation.FULL
                if LAYERWISE and sid == LAYER_NUM - 2 or (not LAYERWISE and sid == STAGE_NUM - 1):
                    self.executing_workload_required_mem[mid] += Activation.LOSS
            self.executing_workload_required_mem[mid] += Activation.FULL
            self.executing_workload_required_mem[mid] += Gradient.INPUT 
            self.executing_workload_required_mem[mid] += Gradient.PARAMETER

    def get_executable_workload(self)->list[Workload]:
        executable_workoads = []
        workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        if self.last_workload_type == WorkloadType.B:
            workload_type_order = [WorkloadType.F,WorkloadType.W,WorkloadType.B]
        elif self.last_workload_type == WorkloadType.F:
            workload_type_order = [WorkloadType.B,WorkloadType.W,WorkloadType.F]
        elif self.last_workload_type == WorkloadType.W:
            workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
            
        # deal with long tail
        if self.exe_num_f == CHUNK_NUM * MICRO_BATCH_NUM:
            workload_type_order = [WorkloadType.F, WorkloadType.B,WorkloadType.W]

        # workload_type_order = [WorkloadType.B,WorkloadType.F,WorkloadType.W]
        
        # raise priority of head and ce
        for workload_type in [WorkloadType.W, WorkloadType.B,WorkloadType.F]:
            for mid in range(self.nmb):
                for stage_id in self.stages:
                    if stage_id > LAYER_NUM and LAYERWISE:
                        workloads = self.stages[stage_id].workloads
                        if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable():
                            executable_workoads = [workloads[mid][workload_type]] + executable_workoads
        # ensure head to be executed as quickly as possible
        if len(executable_workoads) > 0:
            if self.current_mem_usage + Activation.LOSS >= GPU_MAX_MEM:
                workload_type_order = [WorkloadType.W,WorkloadType.B,WorkloadType.F]

        for workload_type in workload_type_order:
            for mid in range(self.nmb):
                for stage_id in self.stages:
                    if stage_id > LAYER_NUM and LAYERWISE:
                        continue
                    workloads = self.stages[stage_id].workloads
                    if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable():
                        executable_workoads.append(workloads[mid][workload_type])
        
        
        # if self.exe_num_b > 0:
        #     executable_workoads.sort(key=lambda x: self.mid_priority[x.microbatch_id], reverse=True)
        
        return executable_workoads

    def overlap_aware_executable_workload_reorder(self, workload:Workload):
        if workload:
            if workload.workload_type ==  WorkloadType.F:
                next_stage_id = workload.sid + 1
            elif workload.workload_type == WorkloadType.B:
                next_stage_id = workload.sid - 1
            else:
                return
            
            if next_stage_id in self.stages.keys():
                head = []
                tail = []
                for wl in self.executable_workloads:
                    if wl.mid == workload.mid and wl.workload_type == workload.workload_type and wl.sid == next_stage_id:
                        tail.append(wl)
                    else:
                        head.append(wl)
                self.executable_workloads = head + tail
                # if head+tail != self.executable_workloads:
                #     print(head+tail)
                #     print(self.executable_workloads)
                #     input()

    def show_stages(self, detail_info=False):
        for sid in self.stages:
            # print("Stage {} recomp={} on Device {}".format(sid, self.stages[sid].recomp, self.device_id))
            if detail_info:
                for mid in self.stages[sid].workloads:
                    if mid == 10:
                        for wlt in self.stages[sid].workloads[mid]:
                            print(self.stages[sid].workloads[mid][wlt])

    def add_stage(self, stage_id: int, recomp:bool = False, layerwise:bool = False) -> None:
        layer_per_stage = LAYER_NUM // STAGE_NUM
        stage_type = StageType.LAYERS
        basic_memory = 0
        layerwise = LAYERWISE

        if layerwise:
            layer_per_stage = 1 
            if stage_id == 0:
                stage_type = StageType.EMBD
            elif stage_id == LAYER_NUM + 1:
                stage_type = StageType.HEAD
                basic_memory = HEAD_MEMORY
            elif stage_id == LAYER_NUM + 2:
                stage_type = StageType.CE
            else:
                stage_type = StageType.LAYER
                basic_memory = LAYER_MEMORY
        else:
            basic_memory = LAYER_MEMORY * layer_per_stage
            if stage_id == STAGE_NUM - 1:
                basic_memory += HEAD_MEMORY
        stage = Stage(
                device_id=self.did, 
                stage_id=stage_id,
                memory_usage=basic_memory/TP_SIZE, 
                stage_type=stage_type,
                recomp=recomp,
                layerwise=layerwise,
            )
        self.stages[stage.sid] = stage
        self.total_layers+=layer_per_stage

    def update_constraints(self, constraint):
        for sid in self.stages:
            self.stages[sid].update_constraints(constraint=constraint)
    
    def update_mid_traverse_order(self,mid=None):
        if type(self.mid_traverse_order) is not list:
            self.mid_traverse_order = list(self.mid_traverse_order)
        self.mid_traverse_order.sort()
        if mid:
            self.mid_traverse_order.remove(mid)
            self.mid_traverse_order.append(mid)
    
    def execute_workload(self, run_schedule=False) -> None:
        if self.state == Device.IDLE:
            if TEMP_TEST:
                self.executable_workloads = self.get_executable_workload()
                print_to_file(f"schedule_results/device{self.did}.txt",f"{GET_TIME()},{len(self.executable_workloads)}\n")
                for workload in self.executable_workloads:
                    workload_type = workload.workload_type
                    sid = workload.sid
                    mid = workload.mid
                    did = workload.did
                    
                    required_memory = get_required_memory(
                        stage_id=sid, 
                        layer_num=1,
                        workload_type=workload_type,
                        workload_type_num=WORKLOAD_TYPE_NUM, 
                        layer_wise=True,
                        recomp=self.stages[sid].recomp,
                    )

                    # TODO DEBUG code
                    if self.exe_num_f < CHUNK_NUM * MICRO_BATCH_NUM:
                        # if workload.mid == 0 and workload.sid == 81 and workload.workload_type == WorkloadType.B:
                        #     input("WAIT")
                        if not self.memory_monitor.is_executable_workload(workload=workload, current_mem=self.current_mem_usage):
                            if workload.mid == 0 and workload.sid == 81 and workload.workload_type == WorkloadType.B:
                                input("WRONG")
                            continue

                    else:
                        if workload.workload_type != WorkloadType.W:
                            if required_memory + self.current_mem_usage > GPU_MAX_MEM - Gradient.PARAMETER:
                                continue

                    proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                    if proc_workload:
                        self.last_workload_type = workload_type
                        if workload_type == WorkloadType.F:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_f += 1
                        elif workload_type == WorkloadType.B:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_b += 1
                        elif workload_type == WorkloadType.W:
                            if self.stages[sid].stage_type in (StageType.LAYER, StageType.LAYERS):
                                self.exe_num_w += 1
                        else:
                            raise Exception("Error workload type.")
                        
                        self.mid_priority[proc_workload.mid] -= 1
                        self.proc_workload = proc_workload
                        self.update_memory_usage()
                        self.state = Device.BUSY
                        self.memory_monitor.trace_workload(workload=workload)
                        if self.did == 0:
                            print(self.memory_monitor.workloads_reserved_mem)
                        return proc_workload
            elif LAYERWISE or SCHEDULE_METHOD in (Schedule.ONE_F_ONE_B, Schedule.ZBH1, Schedule.Layerwise):  
                # Method 1 6661   
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                if self.last_workload_type == WorkloadType.B:
                    now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
                elif self.last_workload_type == WorkloadType.F:
                    now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                if self.current_mem_usage > GPU_MAX_MEM * self.memory_usage_constrain_rate:
                    now_workload_priority_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]
                mb_range = self.mid_traverse_order
                
                # if not self.ok_flag:
                #     now_workload_priority_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]
                #     for workload_type in now_workload_priority_order:
                #         for mid in mb_range:
                #             for sid in self.stages:
                                
                #                 required_memory = get_required_memory(
                #                     stage_id=sid, 
                #                     layer_num=1,
                #                     workload_type=workload_type,
                #                     workload_type_num=WORKLOAD_TYPE_NUM, 
                #                     layer_wise=True,
                #                     recomp=self.stages[sid].recomp,
                #                 )

                #                 if mid != self.executing_mid_idx:
                #                     try:
                #                         if self.executing_mid_idx < self.nmb and GPU_MAX_MEM - required_memory - self.current_mem_usage - Activation.FULL < self.executing_workload_required_mem[self.executing_mid_idx + 1]:
                #                             mid = self.executing_mid_idx
                #                     except Exception as e:
                #                         print(self.executing_mid_idx)
                #                 proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                #                 if proc_workload:
                #                     self.last_workload_type = workload_type
                #                     if workload_type == WorkloadType.F:
                #                         if self.stages[sid].recomp:
                #                             self.executing_workload_required_mem[mid] -= Activation.INPUT
                #                         else:
                #                             self.executing_workload_required_mem[mid] -= Activation.FULL
                                        
                #                         if is_head_layer(sid=sid) or is_last_stage(sid=sid):
                #                             self.executing_workload_required_mem[mid] -= Activation.LOSS 

                #                         if self.stages[sid].stage_type == StageType.LAYER:
                #                             self.exe_num_f += 1
                #                     elif workload_type == WorkloadType.B:
                #                         if self.stages[sid].recomp:
                #                             self.executing_workload_required_mem[mid] -= (Activation.FULL - Activation.INPUT + Gradient.INPUT)
                #                         else:
                #                             self.executing_workload_required_mem[mid] -= Gradient.INPUT

                #                         if self.stages[sid].stage_type == StageType.LAYER:
                #                             self.exe_num_b += 1
                #                     elif workload_type == WorkloadType.W:
                #                         self.executing_workload_required_mem[mid] -= Gradient.PARAMETER

                #                         if self.stages[sid].stage_type == StageType.LAYER:
                #                             self.exe_num_w += 1
                #                     else:
                #                         raise Exception("Error workload type.")
                #                     self.proc_workload = proc_workload
                #                     self.update_memory_usage()
                #                     self.state = Device.BUSY
                #                     if self.executing_mid_idx is None:
                #                         self.executing_mid_idx = mid
                #                     else:
                #                         if workload_type == WorkloadType.W:
                #                             self.executing_mid_idx += 1
                #                             self.ok_flag = True
                #                     return proc_workload
                # else:
                for workload_type in now_workload_priority_order:
                    for mid in mb_range:
                        for sid in self.stages:
                            
                            required_memory = get_required_memory(
                                stage_id=sid, 
                                layer_num=1,
                                workload_type=workload_type,
                                workload_type_num=WORKLOAD_TYPE_NUM, 
                                layer_wise=True,
                                recomp=self.stages[sid].recomp,
                            )

                            if workload_type == WorkloadType.F and required_memory + self.current_mem_usage > GPU_MAX_MEM - Gradient.PARAMETER - Gradient.INPUT:
                                continue
                            
                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                self.last_workload_type = workload_type
                                if workload_type == WorkloadType.F:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_w += 1
                                else:
                                    raise Exception("Error workload type.")
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload
            elif SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
                if self.next_workload_idx == len(self.stages) * self.nmb * WORKLOAD_TYPE_NUM:
                    return None
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx]
                

                if self.wait_for_schedule != 0 and self.wait_for_schedule > GET_TIME():
                    # delay the first F to achive overlap with B
                    return None
                
                proc_workload = self.stages[workload_sid].execute_workload(mid=workload_mid,workload_type=workload_type)
                if proc_workload:
                    self.proc_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.last_workload_type = workload_type
                    if workload_type == WorkloadType.F:
                        self.exe_num_f += 1
                    elif workload_type == WorkloadType.B:
                            self.exe_num_b += 1
                    elif workload_type == WorkloadType.W:
                            self.exe_num_w += 1
                    else:
                        raise Exception("Error workload type.")
                    self.next_workload_idx += 1

                    if workload_type == WorkloadType.F and self.exe_num_f == DEVICE_NUM * (CHUNK_NUM - 1) + (DEVICE_NUM - self.did - 1) * 2:
                        f_time = F_TIME * LAYER_NUM // CHUNK_NUM // DEVICE_NUM
                        self.wait_for_schedule = GET_TIME() + (DEVICE_NUM - self.did) * f_time * 2 - f_time
                    return proc_workload

                
                
            elif run_schedule or SCHEDULE_METHOD in (Schedule.STANDARD_1F1B, Schedule.STANDARD_AFAB):
                if self.next_workload_idx == len(self.stages) * self.nmb * WORKLOAD_TYPE_NUM:
                    return None
                if self.next_workload_idx == len(self.static_schedule):
                    return None
                try:
                    (workload_type, workload_mid, workload_sid) = self.static_schedule[self.next_workload_idx][:3]
                except Exception as e:
                    print((workload_type, workload_mid, workload_sid))
                    input()
                proc_workload = self.stages[workload_sid].execute_workload(mid=workload_mid,workload_type=workload_type)
                if proc_workload:
                    self.proc_workload = proc_workload
                    self.update_memory_usage()
                    self.state = Device.BUSY
                    self.next_workload_idx += 1
                    return proc_workload
            elif SCHEDULE_METHOD == Schedule.INTERLEAVED:
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F]
                if self.last_workload_type == WorkloadType.F:
                    now_workload_priority_order = [WorkloadType.B, WorkloadType.F]
                else:
                    now_workload_priority_order = [WorkloadType.F, WorkloadType.B]
                if SPLIT_BACKPROP:
                    now_workload_priority_order.append(WorkloadType.W)
                    if self.current_mem_usage > GPU_MAX_MEM * self.memory_usage_constrain_rate:
                        now_workload_priority_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]
                else:
                    if self.current_mem_usage > GPU_MAX_MEM * self.memory_usage_constrain_rate:
                        now_workload_priority_order = [WorkloadType.B, WorkloadType.F]
                mb_range = self.mid_traverse_order

                # # warmup
                # if self.device_id == 0:
                #     temp_now_workload_priority_order = copy.deepcopy(now_workload_priority_order)
                #     if self.exe_num_f >= (CHUNK_NUM - 1) * DEVICE_NUM + (DEVICE_NUM - 1 - self.device_id) * 2 + 1:
                #         now_workload_priority_order = [WorkloadType.B]
                #         if SPLIT_BACKPROP:
                #             now_workload_priority_order.append(WorkloadType.W)
                #     if self.exe_num_b > 0:
                #         now_workload_priority_order = temp_now_workload_priority_order

                for workload_type in now_workload_priority_order: # 1
                    for mid in mb_range: # 2
                        for sid in self.stages: # 3                                
                        # for sid in list(reversed(self.stages)):
                            required_memory = get_required_memory(
                                stage_id=sid, 
                                layer_num=LAYER_NUM//CHUNK_NUM//DEVICE_NUM,
                                workload_type=workload_type,
                                workload_type_num=len(now_workload_priority_order), 
                                layer_wise=False,
                                recomp=self.stages[sid].recomp,
                            )
                            workload_type = self._reset_workload_type(
                                workload_type=workload_type,
                                required_memory=required_memory,
                                current_mem_usage=self.current_mem_usage,
                                max_memory=GPU_MAX_MEM,
                            )
                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                self.last_workload_type = workload_type
                                if workload_type == WorkloadType.F:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    if self.stages[sid].stage_type == StageType.LAYER:
                                        self.exe_num_w += 1
                                else:
                                    raise Exception("Error workload type.")
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload  
            
            elif SCHEDULE_METHOD == Schedule.GREEDY_v1:
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
            elif SCHEDULE_METHOD == Schedule.GREEDY_v2:
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
            elif SCHEDULE_METHOD == Schedule.STANDARD_ZBH1:
                if self.last_workload_type == WorkloadType.F:
                    workload_type = WorkloadType.B
                elif self.last_workload_type == WorkloadType.B:
                    workload_type = WorkloadType.W
                elif self.last_workload_type == WorkloadType.W:
                    workload_type = WorkloadType.F
                else:
                    workload_type = WorkloadType.F

                if self.warmup_num_f < DEVICE_NUM - self.did:
                    workload_type = WorkloadType.F

                if self.warmup_num_f == MICRO_BATCH_NUM:
                    workload_type = WorkloadType.B
                if self.warmup_num_b == MICRO_BATCH_NUM:
                    workload_type = WorkloadType.W
                for mid in range(MICRO_BATCH_NUM):
                    for sid in self.stages:
                        required_memory = get_required_memory(
                            stage_id=sid, 
                            layer_num=LAYER_NUM//STAGE_NUM,
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
                        )

                        proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                        if proc_workload:
                            if workload_type == WorkloadType.F:
                                self.warmup_num_f += 1
                            elif workload_type == WorkloadType.B:
                                self.warmup_num_b += 1
                            elif workload_type == WorkloadType.W:
                                self.warmup_num_w += 1
                            self.last_workload_type = workload_type
                            self.proc_workload = proc_workload
                            self.update_memory_usage()
                            self.state = Device.BUSY
                            return proc_workload
                        
                # if self.warmup_num_f < DEVICE_NUM * 2:
                #     return None
                
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                for workload_type in now_workload_priority_order:
                    for mid in range(MICRO_BATCH_NUM):
                        for sid in self.stages:
                            required_memory = get_required_memory(
                                stage_id=sid, 
                                layer_num=LAYER_NUM//STAGE_NUM,
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
                            )

                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                if workload_type == WorkloadType.F:
                                    self.warmup_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    self.warmup_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    self.warmup_num_w += 1
                                self.last_workload_type = workload_type
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload       

            elif SCHEDULE_METHOD == Schedule.ZBV:
                if self.last_workload_type == WorkloadType.F:
                    workload_type = WorkloadType.B
                elif self.last_workload_type == WorkloadType.B:
                    workload_type = WorkloadType.W
                elif self.last_workload_type == WorkloadType.W:
                    workload_type = WorkloadType.F
                else:
                    workload_type = WorkloadType.F

                if self.warmup_num_f < DEVICE_NUM * 2:
                    workload_type = WorkloadType.F
                if self.warmup_num_f == MICRO_BATCH_NUM * 2:
                    workload_type = WorkloadType.B
                if self.warmup_num_b == MICRO_BATCH_NUM * 2:
                    workload_type = WorkloadType.W

                for mid in range(MICRO_BATCH_NUM):
                    for sid in self.stages:
                        required_memory = get_required_memory(
                            stage_id=sid, 
                            layer_num=LAYER_NUM//STAGE_NUM,
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
                        )

                        proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                        if proc_workload:
                            if workload_type == WorkloadType.F:
                                self.warmup_num_f += 1
                            elif workload_type == WorkloadType.B:
                                self.warmup_num_b += 1
                            elif workload_type == WorkloadType.W:
                                self.warmup_num_w += 1
                            self.last_workload_type = workload_type
                            self.proc_workload = proc_workload
                            self.update_memory_usage()
                            self.state = Device.BUSY
                            return proc_workload
                        
                # if self.warmup_num_f < DEVICE_NUM * 2:
                #     return None
                
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                for workload_type in now_workload_priority_order:
                    for mid in range(MICRO_BATCH_NUM):
                        for sid in self.stages:
                            # required_memory = get_required_memory(
                            #     stage_id=sid, 
                            #     layer_num=LAYER_NUM//STAGE_NUM,
                            #     workload_type=workload_type,
                            #     workload_type_num=WORKLOAD_TYPE_NUM, 
                            #     layer_wise=True,
                            #     recomp=self.stages[sid].recomp,
                            # )

                            # workload_type = self._reset_workload_type(
                            #     workload_type=workload_type,
                            #     required_memory=required_memory,
                            #     current_mem_usage=self.current_mem_usage,
                            #     max_memory=GPU_MAX_MEM,
                            # )

                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                if workload_type == WorkloadType.F:
                                    self.warmup_num_f += 1
                                elif workload_type == WorkloadType.B:
                                    self.warmup_num_b += 1
                                elif workload_type == WorkloadType.W:
                                    self.warmup_num_w += 1
                                self.last_workload_type = workload_type
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload             
            elif SCHEDULE_METHOD == Schedule.Chimera:
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                if self.last_workload_type == WorkloadType.B:
                    now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
                elif self.last_workload_type == WorkloadType.F:
                    now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                
                for workload_type in now_workload_priority_order:
                    for mid in self.mid_traverse_order:
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
                            )

                            proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
                            if proc_workload:
                                self.last_workload_type = workload_type
                                self.proc_workload = proc_workload
                                self.update_memory_usage()
                                self.state = Device.BUSY
                                return proc_workload
            else:
                print("Schedule Not Supported")
        return None

    def _reset_workload_type(self, workload_type, required_memory, current_mem_usage, max_memory):
        #TODO 不同个Wave情况下，处于Wave不同边上的memory预留的值应该不同
        if current_mem_usage + required_memory >= max_memory - Gradient.INPUT - Gradient.PARAMETER:
            workload_type = WorkloadType.B
            if not SPLIT_BACKPROP:
                return workload_type
        if current_mem_usage + required_memory >= max_memory - Gradient.PARAMETER:
            workload_type = WorkloadType.W
        return workload_type


    def _finish_proc_workload(self) -> bool: 
        if self.state == Device.BUSY:
            if self.proc_workload and GET_TIME() >= self.proc_workload.end_time:
                return True
        return False

    def update_memory_usage(self) -> int:
        if self.proc_workload.state == Workload.IN_PROGRESS and self.proc_workload.workload_type in (WorkloadType.F, WorkloadType.B):
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
        elif self.proc_workload.state == Workload.COMPLETED and self.proc_workload.workload_type == WorkloadType.W:
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
            
        self.current_mem_usage = self.optimizer_mem_usage + sum(stage.memory_usage for stage in self.stages.values())
        self.mem_usage_record[(self.proc_workload.start_time,self.proc_workload.end_time)] = self.current_mem_usage
    
    def get_memory_usage(self) -> int:
        return self.current_mem_usage

    def __repr__(self) -> str:
        return f"DeviceClass(stages={self.stages.keys()}),state={self.state}"
    




    # if self.exe_num_b < self.exe_num_f:
    #     now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
    # else:
    #     now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]


    # # Method 2 6841
    # if self.last_workload_type == WorkloadType.B:
    #     now_workload_priority_order = [WorkloadType.F, WorkloadType.W, WorkloadType.B]
    # elif self.last_workload_type == WorkloadType.F:
    #     now_workload_priority_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
    # elif self.last_workload_type == WorkloadType.W:
    #     now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]

    # # 6661
    # if self.exe_num_b < self.exe_num_f:
    #     now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
    # else:
    #     now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]

    # if self.order_balance:
    #     mb_range = self.mid_traverse_order
    # else:
    #     mb_range = list(reversed(self.mid_traverse_order))

    # if self.exe_num_f > MICRO_BATCH_NUM*2.5:
    #     for workload_type in now_workload_priority_order:
    #         # for mid in list(reversed(self.microbatch_schedule_range)):
    #         for mid in mb_range:
    #             for sid in self.stages:
    #                 required_memory = get_required_memory(
    #                     stage_id=sid, 
    #                     layer_num=1,
    #                     workload_type=workload_type,
    #                     workload_type_num=WORKLOAD_TYPE_NUM, 
    #                     layer_wise=True,
    #                     recomp=self.stages[sid].recomp,
    #                 )

    #                 workload_type = self._reset_workload_type(
    #                     workload_type=workload_type,
    #                     required_memory=required_memory,
    #                     current_mem_usage=self.current_mem_usage,
    #                     max_memory=GPU_MAX_MEM,
    #                 )

    #                 proc_workload = self.stages[sid].execute_workload(mid=mid,workload_type=workload_type)
    #                 if proc_workload:
    #                     self.last_workload_type = workload_type
    #                     if workload_type == WorkloadType.F:
    #                         if self.stages[sid].stage_type == StageType.LAYER:
    #                             self.exe_num_f += 1
    #                     elif workload_type == WorkloadType.B:
    #                         if self.stages[sid].stage_type == StageType.LAYER:
    #                             self.exe_num_b += 1
    #                     elif workload_type == WorkloadType.W:
    #                         if self.stages[sid].stage_type == StageType.LAYER:
    #                             self.exe_num_w += 1
    #                     else:
    #                         raise Exception("Error workload type.")
    #                     self.proc_workload = proc_workload
    #                     self.update_memory_usage()
    #                     self.state = Device.BUSY
    #                     self.order_balance = not self.order_balance
    #                     return proc_workload