from .Workload import *
import copy
class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self, device_id:int, stage_id: int, memory_usage: int):
        self.device_id: int = device_id
        self.stage_id: int = stage_id
        self.memory_usage: int = memory_usage
        self.model_mem: int = memory_usage
        self.workloads: dict[int, {WorkloadType, Workload}] = {}  
        self._add_workload()
    
    def _get_workload_duration(self, device_id, stage_id, microbatch_id, workload_type):
        if SchedulePriority.Layerwise == SCHEDULE_METHOD:
            if workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
                duration = FPW_TIME
                if stage_id == 0:
                    duration = EMBEDDING_TIME
                elif stage_id == LAYER_NUM + 2:
                    duration = LOSS_F_TIME
                elif stage_id == LAYER_NUM + 1:
                    duration = LAST_FFN_F_TIME

            elif workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
                duration = IGW_TIME
                if stage_id == LAYER_NUM + 2:
                    duration = LOSS_B_TIME
                elif stage_id == LAYER_NUM + 1:
                    duration = LAST_FFN_B_TIME
            elif workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                duration = PGW_TIME
                if stage_id == LAYER_NUM + 2:
                    duration = LOSS_W_TIME
                elif stage_id == LAYER_NUM + 1:
                    duration = LAST_FFN_W_TIME
        else:
            layer_per_stage = LAYER_NUM // STAGE_NUM
            if workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
                duration = FPW_TIME * layer_per_stage
                if stage_id == 0:
                    duration += EMBEDDING_TIME
                elif stage_id == STAGE_NUM - 1:
                    duration += LAST_FFN_F_TIME + LOSS_F_TIME
            elif workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
                duration = IGW_TIME * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += LAST_FFN_B_TIME + LOSS_B_TIME
            elif workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                duration = PGW_TIME * layer_per_stage
                if stage_id == STAGE_NUM - 1:
                    duration += LAST_FFN_W_TIME + LOSS_W_TIME
            else:
                raise Exception("Wrong workload type!")
        return duration
        
    def _add_workload(self) -> None:
        for mid in range(MICRO_BATCH_NUM):
            fpw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.FORWARD_PASS_WORKLOAD,
                duration=self._get_workload_duration(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    microbatch_id=mid,
                    workload_type=WorkloadType.FORWARD_PASS_WORKLOAD,
                ),    
                total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,
            )
            if self.stage_id == 0 and SCHEDULE_METHOD == SchedulePriority.Layerwise:
                self.workloads[mid]={
                    WorkloadType.FORWARD_PASS_WORKLOAD: fpw,
                }
                continue

            igw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.INPUT_GRADIENT_WORKLOAD,
                duration=self._get_workload_duration(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    microbatch_id=mid,
                    workload_type=WorkloadType.INPUT_GRADIENT_WORKLOAD,
                ),   
                total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,   
            )
            self.workloads[mid]={
                WorkloadType.FORWARD_PASS_WORKLOAD: fpw,
                WorkloadType.INPUT_GRADIENT_WORKLOAD: igw,
            }
            if SPLIT_BACKPROP:
                pgw = Workload(
                    device_id=self.device_id,
                    stage_id=self.stage_id,
                    microbatch_id=mid,
                    workload_type=WorkloadType.PARAMETER_GRADIENT_WORKLOAD,
                    duration=self._get_workload_duration(
                        device_id=self.device_id,
                        stage_id=self.stage_id,
                        microbatch_id=mid,
                        workload_type=WorkloadType.PARAMETER_GRADIENT_WORKLOAD,
                    ),     
                    total_stages=LAYER_NUM+3 if SchedulePriority.Layerwise == SCHEDULE_METHOD else STAGE_NUM,
                )
                self.workloads[mid][WorkloadType.PARAMETER_GRADIENT_WORKLOAD] = pgw

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
        layers_per_stage = 1 if SCHEDULE_METHOD == SchedulePriority.Layerwise else LAYER_NUM // STAGE_NUM
        if workload.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            if self.stage_id != 0:
                self.memory_usage += Activation.FULL_LAYER * layers_per_stage
        elif workload.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.memory_usage -= (Activation.FULL_LAYER + Gradient.INPUT) * layers_per_stage
        elif SPLIT_BACKPROP and workload.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            self.memory_usage += Gradient.INPUT * layers_per_stage
        elif SPLIT_BACKPROP == False and workload.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            self.memory_usage -= Activation.FULL_LAYER * layers_per_stage
        
    def execute_workload(self, mid=None, workload_type=None):
        if mid is not None and workload_type is not None and workload_type in self.workloads[mid]:
            w = self.workloads[mid][workload_type]
            if w.execute():
                return copy.deepcopy(w)
        else:
            if self.stage_id == 0 and workload_type in (WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.PARAMETER_GRADIENT_WORKLOAD):
                return None
            print("Lack of workload info.")
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.stage_id}, "
                f"memory_usage={self.memory_usage})")
