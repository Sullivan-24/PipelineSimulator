from .Stage import *
from ..utils import save_to_file

def get_required_memory_by_workload(workload:Workload):
        stage_id = workload.sid
        layer_num = LAYER_NUM // DEVICE_NUM // CHUNK_NUM
        workload_type = workload.wtype
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
            raise ValueError("get_required_memory: Unsupported workload type!")

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
            # 应该是是动态变化的，而非一成不变，memory monitor需要保证至少剩余的显存容量足够完成一个mid的所有workload
            # 每次发生memory usage减少的时候都应该重新修改reserved mem的值
            # 并且还要判断显存开销的最高峰，比如loss的显存开销大于 input+para时要以loss为准，反之要以input+para为准
            # self.workloads_reserved_mem[mid] += Gradient.INPUT + Gradient.PARAMETER + Gradient.HEAD_INPUT + Gradient.HEAD_PARA
            self.workloads_reserved_mem[mid] += Gradient.INPUT + Gradient.PARAMETER
            if self.have_head_layer() and Gradient.HEAD_INPUT + Gradient.HEAD_PARA > Gradient.INPUT + Gradient.PARAMETER:
                self.workloads_reserved_mem[mid] -= Gradient.INPUT + Gradient.PARAMETER
                self.workloads_reserved_mem[mid] += Gradient.HEAD_INPUT + Gradient.HEAD_PARA

            # print(f"Device {self.did} Reserve {self.workloads_reserved_mem[mid]}G for mid {mid}")

    def have_head_layer(self):
        if LAYERWISE and (LAYER_NUM + 1 in list(self.stages.keys())):
            return True
        elif not LAYERWISE and (STAGE_NUM - 1 in list(self.stages.keys())):
            return True
        return False
    
    def is_last_w(self, workload:Workload):
        if workload.wtype == WorkloadType.W:
            sorted_sids = sorted(list(self.stages.keys()))
            # Consider the embeding layer
            min_sid_idx = 0 if LAYERWISE and 0 not in sorted_sids else 1
            if workload.sid == sorted_sids[min_sid_idx]:
                return True
        return False
    
    def trace_workload(self,workload:Workload):
        self.tracing_workloads.append(workload)
        required_mem = get_required_memory_by_workload(workload)
        self.workloads_reserved_mem[workload.mid] -= required_mem
        # if workload.is_w:
        #     if self.is_last_w(workload=workload):
        #         self.safe_workload_mids.append(workload.mid)
        #     else:
        #         self.workloads_reserved_mem[workload.mid] += Gradient.INPUT + Gradient.PARAMETER
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
        if workload.is_w and required_mem + current_mem <= self.max_memory:
            safe_count += 1
        self.workloads_reserved_mem[workload.mid] += required_mem
        return safe_count > 0

class Device:
    
    BUSY = 1
    IDLE = 2

    def __init__(self, device_id: int, max_activation_counts: int, nmb:int, static_schedule: list = None, memory_usage_constrain_rate: float = 1, max_mem: int = GPU_MAX_MEM, comp_power: float = 1, layer_density: list = None):
        self.did = device_id
        self.stages: dict[int, Stage] = {}  # 存放各阶段的字典
        self.state: int = Device.IDLE
        self.proc_workload: Workload = None
        self.current_mem_usage: int = 0
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
        self.max_memory = max_mem
        self.comp_power = comp_power
        if layer_density is None:
            self.layer_density = [1 for _ in range(LAYER_NUM)]
        else:
            self.layer_density = layer_density

        self.workload_execute_record: list[list[Workload]] = [[] for _ in range(DEVICE_NUM)]

    def init_memory_monitor(self):
        self.memory_monitor = MemoryMonitor(self.nmb, self.stages, self.did, max_memory=self.max_memory * MEMORY_CONSTRAIN)
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

    def get_executable_workload(self, time)->list[Workload]:
        executable_workoads = []
        workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        # if self.current_mem_usage <= 0.75 * self.max_memory: cause too much memory pressure leading to low performance
        if self.current_mem_usage <= 0.00 * self.max_memory:
            if self.last_workload_type == WorkloadType.B:
                workload_type_order = [WorkloadType.F,WorkloadType.W,WorkloadType.B]
            elif self.last_workload_type == WorkloadType.F:
                workload_type_order = [WorkloadType.B,WorkloadType.W,WorkloadType.F]
            elif self.last_workload_type == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]
        else:
            if self.last_workload_type == WorkloadType.B:
                workload_type_order = [WorkloadType.W,WorkloadType.F,WorkloadType.B]
            elif self.last_workload_type == WorkloadType.F:
                workload_type_order = [WorkloadType.B,WorkloadType.W,WorkloadType.F]
            elif self.last_workload_type == WorkloadType.W:
                workload_type_order = [WorkloadType.F,WorkloadType.B,WorkloadType.W]

        # deal with long tail
        if self.exe_num_f == CHUNK_NUM * MICRO_BATCH_NUM:
            workload_type_order = [WorkloadType.F, WorkloadType.B,WorkloadType.W]

        
        # raise priority of head and ce
        head_ce_workloads = []
        for workload_type in [WorkloadType.W, WorkloadType.B,WorkloadType.F]:
            for mid in range(self.nmb):
                for stage_id in self.stages:
                    if stage_id > LAYER_NUM and LAYERWISE:
                        workloads = self.stages[stage_id].workloads
                        if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable(time=time):
                            head_ce_workloads.append(workloads[mid][workload_type])
        # ensure head to be executed as quickly as possible
        executable_workoads += head_ce_workloads
        if len(executable_workoads) > 0:
            if self.current_mem_usage + Activation.LOSS >= self.max_memory:
                workload_type_order = [WorkloadType.W,WorkloadType.B,WorkloadType.F]
        
        delayed_workload = []
        for workload_type in workload_type_order:    
            for mid in range(self.nmb):
                for stage_id in self.stages:
                    if stage_id > LAYER_NUM and LAYERWISE:
                        continue
                    workloads = self.stages[stage_id].workloads
                    if workload_type in workloads[mid] and workloads[mid][workload_type].is_executable(time=time):
                        workload = workloads[mid][workload_type]
                        # make sure warmup is finished as quickly as possible
                        if OVERLAP_AWARE_SCHEDULE and self.exe_num_b > 0 and self.should_delay_for_overlap(time=time, workload=workload):
                            if OVERLAP_DEGREE is None:
                                delayed_workload.append(workload)
                        else:
                            executable_workoads.append(workloads[mid][workload_type])
        #decrease priority of the same mb
        executable_workoads = executable_workoads + delayed_workload

        # if self.exe_num_b > 0:
        #     executable_workoads.sort(key=lambda x: self.mid_priority[x.microbatch_id], reverse=True)
        
        return executable_workoads

    def should_delay_for_overlap(self, time, workload:Workload):
        for did,executed_workloads in enumerate(self.workload_execute_record):
            if did == self.did or len(executed_workloads) == 0:
                continue
            pivot_workload = executed_workloads[-1]
            if pivot_workload.sid == workload.sid - 1 and pivot_workload.wtype == workload.wtype == WorkloadType.F:
                # print(f"Delay workload ({workload.did},{workload.sid},{workload.mid},{workload.wtype}) due to ({pivot_workload.did},{pivot_workload.sid},{pivot_workload.mid},{pivot_workload.wtype})")
                if not OVERLAP_DEGREE:
                    return True
                if OVERLAP_DEGREE and pivot_workload.end_time + pivot_workload.duration // OVERLAP_DEGREE >= time:
                    return True
            if pivot_workload.sid == workload.sid + 1 and pivot_workload.wtype == workload.wtype == WorkloadType.B:
                # print(f"Delay workload ({workload.did},{workload.sid},{workload.mid},{workload.wtype}) due to ({pivot_workload.did},{pivot_workload.sid},{pivot_workload.mid},{pivot_workload.wtype})")
                if not OVERLAP_DEGREE:
                    return True
                if OVERLAP_DEGREE and pivot_workload.end_time + pivot_workload.duration // OVERLAP_DEGREE >= time:
                    return True
        return False

    def overlap_aware_executable_workload_reorder(self, workload:Workload):
        if workload:
            if workload.wtype ==  WorkloadType.F:
                next_stage_id = workload.sid + 1
            elif workload.wtype == WorkloadType.B:
                next_stage_id = workload.sid - 1
            else:
                return
            
            if next_stage_id in self.stages.keys():
                head = []
                tail = []
                for wl in self.executable_workloads:
                    if wl.mid == workload.mid and wl.wtype == workload.wtype and wl.sid == next_stage_id:
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

    def add_stage(self, stage_id: int, 
                  recomp:bool = False, 
                  layerwise:bool = LAYERWISE, 
                  layer_num = LAYER_NUM // STAGE_NUM, 
                  basic_memory = 0) -> None:
        
        stage_type = StageType.LAYERS
        para_num = 0
        if layerwise:
            assert layer_num == 1 and CHUNK_NUM == LAYER_NUM // PP_SIZE, f"LAYERWISE require 1 layer per stage (CHUNK_NUM == LAYER_NUM // PP_SIZE) but got {layer_num} per stage"
            if stage_id == 0:
                stage_type = StageType.EMBD
                basic_memory = StateMemory.EMB
                para_num = Parameter.EMB
            elif stage_id == LAYER_NUM + 1:
                stage_type = StageType.HEAD
                basic_memory = StateMemory.HEAD
                para_num = Parameter.HEAD
            elif stage_id == LAYER_NUM + 2:
                stage_type = StageType.CE
            else:
                stage_type = StageType.LAYER
                basic_memory = StateMemory.LAYER
                para_num = Parameter.LAYER
        else:
            basic_memory = StateMemory.LAYER * layer_num
            para_num = Parameter.LAYER
            if stage_id == 0:
                basic_memory += StateMemory.EMB
                para_num += Parameter.EMB
            elif stage_id == STAGE_NUM - 1:
                basic_memory += StateMemory.HEAD
                para_num += Parameter.HEAD
        stage = Stage(
                device_id=self.did, 
                stage_id=stage_id,
                para_num=para_num,
                stage_type=stage_type,
                recomp=recomp,
                layerwise=layerwise,
                layer_num=layer_num,
                comp_power=self.comp_power,
                layer_density=self.layer_density,
            )
        self.stages[stage.sid] = stage
        self.total_layers+=layer_num

    def count_wtype_num(self, did : int, wtype : WorkloadType):
        count = 0
        for w in self.workload_execute_record[did]:
            count += 1 if w.wtype == wtype else 0
        return count

    def update_constraints(self, time, constraint):
        for sid in self.stages:
            self.stages[sid].update_constraints(time, constraint=constraint)
    
    def update_mid_traverse_order(self,mid=None):
        if type(self.mid_traverse_order) is not list:
            self.mid_traverse_order = list(self.mid_traverse_order)
        self.mid_traverse_order.sort()
        if mid:
            self.mid_traverse_order.remove(mid)
            self.mid_traverse_order.append(mid)
    
    def execute_workload(self, time, run_schedule=False) -> None:
        assert time >= 0, f"Time should be non-negative (but got {time})."
        if self.state == Device.IDLE:
            if SCHEDULE_METHOD == Schedule.UnifiedPP:
                self.executable_workloads = self.get_executable_workload(time=time)
                save_to_file(f"schedule_results/workload_statistics/device{self.did}.txt",f"{time},{len(self.executable_workloads)}\n", 'a')
                for workload in self.executable_workloads:
                    workload_type = workload.wtype
                    sid = workload.sid
                    mid = workload.mid
                    did = workload.did
                    
                    required_memory = get_required_memory_by_workload(workload=workload)

                    # TODO More optimizations?
                    if self.exe_num_f < CHUNK_NUM * MICRO_BATCH_NUM * 1: #critical coefficient
                        # MemoryMonitor: ensure workloads are safely launched
                        if not self.memory_monitor.is_executable_workload(workload=workload, current_mem=self.current_mem_usage):
                            continue
                    else:
                        if workload.wtype != WorkloadType.W:
                            if required_memory + self.current_mem_usage > self.max_memory - Gradient.PARAMETER:
                                continue

                    proc_workload = self.stages[sid].execute_workload(time, mid=mid,workload_type=workload_type)
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
                        # if self.did == 0:
                        #     print(self.memory_monitor.workloads_reserved_mem)
                        return proc_workload
            elif LAYERWISE or SCHEDULE_METHOD in (Schedule.ONE_F_ONE_B, Schedule.ZBH1, Schedule.Layerwise):  
                # Method 1 6661   
                now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                if self.last_workload_type == WorkloadType.B:
                    now_workload_priority_order = [WorkloadType.F, WorkloadType.B, WorkloadType.W]
                elif self.last_workload_type == WorkloadType.F:
                    now_workload_priority_order = [WorkloadType.B, WorkloadType.F, WorkloadType.W]
                if self.current_mem_usage > self.max_memory * self.memory_usage_constrain_rate:
                    now_workload_priority_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]
                mb_range = self.mid_traverse_order
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

                            if workload_type == WorkloadType.F and required_memory + self.current_mem_usage > self.max_memory - Gradient.PARAMETER - Gradient.INPUT:
                                continue
                            
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                
                if self.did < PP_SIZE - 1:
                    if self.exe_num_f > (CHUNK_NUM - 1) * PP_SIZE + (PP_SIZE - self.did - 1) * 2 - 1 and self.exe_num_f < CHUNK_NUM * MICRO_BATCH_NUM:
                        if workload_type == WorkloadType.F and self.count_wtype_num(self.did, WorkloadType.B) < self.count_wtype_num(self.did + 1, WorkloadType.B) and self.workload_execute_record[self.did + 1][-1].wtype == WorkloadType.B:
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        elif workload_type in (WorkloadType.B, WorkloadType.W):
                            proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)  
                        else:
                            return None
                    else:
                        proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                else:
                    proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
                
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
                proc_workload = self.stages[workload_sid].execute_workload(time=time, mid=workload_mid,workload_type=workload_type)
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
                    if self.current_mem_usage > self.max_memory * self.memory_usage_constrain_rate:
                        now_workload_priority_order = [WorkloadType.W, WorkloadType.B, WorkloadType.F]
                else:
                    if self.current_mem_usage > self.max_memory * self.memory_usage_constrain_rate:
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
                                max_memory=self.max_memory,
                            )
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                            max_memory=self.max_memory,
                        )

                        proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                                max_memory=self.max_memory,
                            )

                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                            max_memory=self.max_memory,
                        )

                        proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                            #     max_memory=self.max_memory,
                            # )

                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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
                                max_memory=self.max_memory,
                            )

                            proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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


    def _finish_proc_workload(self,time) -> bool: 
        if self.state == Device.BUSY:
            if self.proc_workload and time >= self.proc_workload.end_time:
                return True
        return False

    def update_memory_usage(self) -> int:
        if self.proc_workload.state == Workload.IN_PROGRESS and self.proc_workload.wtype in (WorkloadType.F, WorkloadType.B):
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
        elif self.proc_workload.state == Workload.COMPLETED and self.proc_workload.wtype == WorkloadType.W:
            self.stages[self.proc_workload.sid].update_memory_usage(workload=self.proc_workload)
            
        self.current_mem_usage = sum(stage.memory_usage for stage in self.stages.values())
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
    #                     max_memory=self.max_memory,
    #                 )

    #                 proc_workload = self.stages[sid].execute_workload(time=time, mid=mid,workload_type=workload_type)
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