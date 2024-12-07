from .Workload import *
import copy
class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self, device_id:int, stage_id: int, memory_usage: int, activation_memory_increment: int):
        self.device_id: int = device_id
        self.stage_id: int = stage_id
        self.memory_usage: int = memory_usage
        self.model_mem: int = memory_usage
        self.activation_memory_increment: int = activation_memory_increment
        self.workloads: dict[int, {WorkloadType, Workload}] = {}  
        self._add_workload()

    def _add_workload(self) -> None:
        for mid in range(MICRO_BATCH_NUM):
            fpw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.FORWARD_PASS_WORKLOAD,
                duration=FPW_TIME // CHUNK_NUM,    
            )
            igw = Workload(
                device_id=self.device_id,
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.INPUT_GRADIENT_WORKLOAD,
                duration=IGW_TIME // CHUNK_NUM,    
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
                    duration=PGW_TIME // CHUNK_NUM,    
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
        if workload.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            self.memory_usage += self.activation_memory_increment
        elif workload.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.memory_usage -= self.activation_memory_increment
        elif SPLIT_BACKPROP == False and workload.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            self.memory_usage -= self.activation_memory_increment

    def execute_workload(self, mid=None, workload_type=None):
        if mid is not None and workload_type is not None:
            w = self.workloads[mid][workload_type]
            if w.execute():
                return copy.deepcopy(w)
        else:
            print("Lack of workload info.")
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.stage_id}, "
                f"memory_usage={self.memory_usage}, "
                f"activation_memory={self.activation_memory_increment})")
