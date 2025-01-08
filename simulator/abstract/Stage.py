from .Workload import *
import copy

@dataclass
class StageType:
    EMBD = 1
    LAYER = 2
    HEAD = 3
    CE = 4
    LAYERS = 5

class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self, device_id:int, stage_id: int, memory_usage: int, stage_type: StageType):
        self.device_id: int = device_id
        self.stage_id: int = stage_id
        self.memory_usage: int = memory_usage
        self.model_mem: int = memory_usage
        self.workloads: dict[int, {WorkloadType, Workload}] = {}  
        self.stage_type: StageType = stage_type
        self._add_workload()
    
    def _get_workload_duration(self, device_id, stage_id, stage_type, microbatch_id, workload_type):
        if SchedulePriority.Layerwise == SCHEDULE_METHOD:
            if workload_type == WorkloadType.F:
                duration = F_TIME
                if stage_type == StageType.EMBD:
                    duration = EMBEDDING_TIME
                elif stage_type == StageType.CE:
                    duration = CE_F_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_F_TIME

            elif workload_type == WorkloadType.B:
                duration = B_TIME
                if stage_type == StageType.CE:
                    duration = CE_B_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_B_TIME
            elif workload_type == WorkloadType.W:
                duration = W_TIME
                if stage_type == StageType.CE:
                    duration = CE_W_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_W_TIME
        else:
            layer_per_stage = LAYER_NUM // STAGE_NUM
            if workload_type == WorkloadType.F:
                duration = F_TIME * layer_per_stage
                if stage_id == 0:
                    duration += EMBEDDING_TIME
                elif stage_id == STAGE_NUM - 1:
                    duration += HEAD_F_TIME + CE_F_TIME
            elif workload_type == WorkloadType.B:
                duration = B_TIME * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += HEAD_B_TIME + CE_B_TIME
            elif workload_type == WorkloadType.W:
                duration = W_TIME * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += HEAD_W_TIME + CE_W_TIME
            else:
                raise Exception("Wrong workload type!")
        return duration
        
    def _add_workload(self) -> None:
        for mid in range(MICRO_BATCH_NUM):
            fpw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.F,
                duration=self._get_workload_duration(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    stage_type=self.stage_type,
                    microbatch_id=mid,
                    workload_type=WorkloadType.F,
                ),    
                total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,
            )
            if self.stage_id == 0 and SCHEDULE_METHOD == SchedulePriority.Layerwise:
                self.workloads[mid]={
                    WorkloadType.F: fpw,
                }
                continue

            igw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.B,
                duration=self._get_workload_duration(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    stage_type=self.stage_type,
                    microbatch_id=mid,
                    workload_type=WorkloadType.B,
                ),   
                total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,   
            )
            self.workloads[mid]={
                WorkloadType.F: fpw,
                WorkloadType.B: igw,
            }
            if SPLIT_BACKPROP:
                pgw = Workload(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    microbatch_id=mid,
                    workload_type=WorkloadType.W,
                    duration=self._get_workload_duration(
                        device_id=self.device_id,
                        stage_id=self.stage_id,
                        stage_type=self.stage_type,
                        microbatch_id=mid,
                        workload_type=WorkloadType.W,
                    ),     
                    total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,
                )
                if self.stage_type != StageType.CE:
                    self.workloads[mid][WorkloadType.W] = pgw

    def update_constraints(self, constraint: Workload):
        for mid in self.workloads:
            for wlt in self.workloads[mid]:
                self.workloads[mid][wlt].update_constraints(
                    WorkloadConstraint(
                        device_id=constraint.device_id,
                        stage_id=constraint.stage_id, 
                        microbatch_id=constraint.microbatch_id, 
                        workload_type=constraint.workload_type
                    )
                ) 

    def update_memory_usage(self, workload:Workload):
        if SCHEDULE_METHOD == SchedulePriority.Layerwise:
            if self.stage_type == StageType.EMBD:
                return
            if workload.workload_type == WorkloadType.F:
                if self.stage_type == StageType.HEAD:
                    self.memory_usage += Activation.LOSS
                # TODO
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    self.memory_usage += Activation.FULL_LAYER
            elif workload.workload_type == WorkloadType.B:
                if self.stage_type == StageType.HEAD:                    
                    self.memory_usage -= Activation.LOSS
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    if SPLIT_BACKPROP:
                        self.memory_usage += Gradient.INPUT
                    else:
                        self.memory_usage -= Activation.FULL_LAYER
            elif workload.workload_type == WorkloadType.W:
                if self.stage_type in (StageType.HEAD, StageType.CE):
                    return
                if SPLIT_BACKPROP:
                    self.memory_usage -= Gradient.INPUT + Activation.FULL_LAYER
        else:
            layers_per_stage = LAYER_NUM // STAGE_NUM
            if workload.workload_type == WorkloadType.F:
                self.memory_usage += Activation.FULL_LAYER * layers_per_stage
                if self.stage_id == STAGE_NUM - 1:
                    self.memory_usage += Activation.LOSS
            elif workload.workload_type == WorkloadType.W:
                self.memory_usage -= (Activation.FULL_LAYER + Gradient.INPUT) * layers_per_stage
            elif SPLIT_BACKPROP and workload.workload_type == WorkloadType.B:
                self.memory_usage += Gradient.INPUT * layers_per_stage
            elif SPLIT_BACKPROP == False and workload.workload_type == WorkloadType.B:
                self.memory_usage -= Activation.FULL_LAYER * layers_per_stage
        
    def execute_workload(self, mid=None, workload_type=None):
        if mid is not None and workload_type is not None and workload_type in self.workloads[mid]:
            w = self.workloads[mid][workload_type]
            if w.execute():
                return copy.deepcopy(w)
        else:
            if self.stage_type == StageType.EMBD and workload_type in (WorkloadType.B, WorkloadType.W):
                return None
            elif self.stage_type == StageType.CE and workload_type in (WorkloadType.W, ):
                return None
            print("Lack of workload info.")
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.stage_id}, "
                f"memory_usage={self.memory_usage})")
