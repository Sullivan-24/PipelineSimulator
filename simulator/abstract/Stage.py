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

    def __init__(self, device_id:int, stage_id: int, memory_usage: int, stage_type: StageType, layerwise:bool = False, microbatch_num:int = MICRO_BATCH_NUM, recomp: bool = False):
        self.device_id: int = device_id
        self.stage_id: int = stage_id
        self.microbatch_num: int = microbatch_num
        self.memory_usage: int = memory_usage
        self.workloads: dict[int, {WorkloadType, Workload}] = {}  
        self.stage_type: StageType = stage_type
        self.recomp = recomp
        self.layerwise = layerwise
        self._add_workload()
    
    def _get_workload_duration(self, stage_id, stage_type, workload_type, recomp):
        if self.layerwise:
            if workload_type == WorkloadType.F:
                duration = F_TIME
                if stage_type == StageType.EMBD:
                    duration = EMBEDDING_TIME
                elif stage_type == StageType.CE:
                    duration = CE_F_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_F_TIME

            elif workload_type == WorkloadType.B:
                duration = B_TIME if not recomp else B_TIME + F_TIME
                if stage_type == StageType.CE:
                    duration = CE_B_TIME if not recomp else CE_B_TIME + CE_F_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_B_TIME if not recomp else HEAD_B_TIME + HEAD_F_TIME
            elif workload_type == WorkloadType.W:
                duration = W_TIME
                if stage_type == StageType.CE:
                    duration = CE_W_TIME
                elif stage_type == StageType.HEAD:
                    duration = HEAD_W_TIME
            else:
                raise Exception("Wrong workload type!")
        else:
            layer_per_stage = LAYER_NUM // STAGE_NUM
            if workload_type == WorkloadType.F:
                duration = F_TIME * layer_per_stage
                if stage_id == 0:
                    duration += EMBEDDING_TIME
                elif stage_id == STAGE_NUM - 1:
                    duration += HEAD_F_TIME + CE_F_TIME
            elif workload_type == WorkloadType.B:
                duration = B_TIME * layer_per_stage if not recomp else (F_TIME + B_TIME) * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += HEAD_B_TIME + CE_B_TIME 
                    if recomp:
                        duration += HEAD_F_TIME + CE_F_TIME
            elif workload_type == WorkloadType.W:
                duration = W_TIME * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += HEAD_W_TIME + CE_W_TIME
            else:
                raise Exception("Wrong workload type!")
        return duration
        
    def _add_workload(self) -> None:
        for mid in range(self.microbatch_num):
            fpw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.F,
                duration=self._get_workload_duration(
                    stage_id=self.stage_id,
                    stage_type=self.stage_type,
                    workload_type=WorkloadType.F,
                    recomp=self.recomp,
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
                    stage_id=self.stage_id,
                    stage_type=self.stage_type,
                    workload_type=WorkloadType.B,
                    recomp=self.recomp,
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
                        stage_id=self.stage_id,
                        stage_type=self.stage_type,
                        workload_type=WorkloadType.W,
                        recomp=self.recomp,
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
                # TODO CE memory cost
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    if self.recomp:
                        self.memory_usage += Activation.INPUT
                        return
                    self.memory_usage += Activation.FULL
            elif workload.workload_type == WorkloadType.B:
                if self.stage_type == StageType.HEAD:                    
                    self.memory_usage -= Activation.LOSS
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    if SPLIT_BACKPROP:
                        self.memory_usage += Gradient.INPUT
                        if self.recomp:
                            self.memory_usage += (Activation.FULL - Activation.INPUT)
                    else:
                        self.memory_usage -= Activation.FULL
            elif workload.workload_type == WorkloadType.W:
                if self.stage_type in (StageType.HEAD, StageType.CE):
                    return
                if SPLIT_BACKPROP:
                    self.memory_usage -= Gradient.INPUT + Activation.FULL
        else:
            layers_per_stage = LAYER_NUM // STAGE_NUM
            if workload.workload_type == WorkloadType.F:
                self.memory_usage += Activation.FULL * layers_per_stage
                if self.stage_id == STAGE_NUM - 1:
                    self.memory_usage += Activation.LOSS
            elif workload.workload_type == WorkloadType.W:
                self.memory_usage -= (Activation.FULL + Gradient.INPUT) * layers_per_stage
                if self.stage_id == STAGE_NUM - 1:
                    self.memory_usage -= Activation.LOSS
            elif SPLIT_BACKPROP and workload.workload_type == WorkloadType.B:
                self.memory_usage += Gradient.INPUT * layers_per_stage
            elif SPLIT_BACKPROP == False and workload.workload_type == WorkloadType.B:
                self.memory_usage -= Activation.FULL * layers_per_stage
                if self.stage_id == STAGE_NUM - 1:
                    self.memory_usage -= Activation.LOSS
        
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
