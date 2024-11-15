from .Workload import *
class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self, stage_id: int, memory_usage: int, activation_memory: int):
        self.stage_id: int = stage_id  # 阶段编号
        self.memory_usage: int = memory_usage  # 阶段内存开销
        self.activation_memory: int = activation_memory  # 激活内存开销
        self.workloads: dict[int, {WorkloadType, Workload}] = {}  
        self._add_workload()

    def _add_workload(self) -> None:
        for mid in range(MICRO_BATCH_NUM):
            fpw = Workload(
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.FORWARD_PASS_WORKLOAD,
                duration=FPW_TIME,    
            )
            igw = Workload(
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.INPUT_GRADIENT_WORKLOAD,
                duration=IGW_TIME,    
            )
            pgw = Workload(
                stage_id=self.stage_id,
                microbatch_id=mid,
                workload_type=WorkloadType.PARAMETER_GRADIENT_WORKLOAD,
                duration=PGW_TIME,    
            )
            self.workloads[mid]={
                WorkloadType.FORWARD_PASS_WORKLOAD: fpw,
                WorkloadType.INPUT_GRADIENT_WORKLOAD: igw,
                WorkloadType.PARAMETER_GRADIENT_WORKLOAD: pgw
            }

    def update_constraints(self, constraint):
        for mid in self.workloads:
            for wlt in self.workloads[mid]:
                self.workloads[mid][wlt].update_constraints(constraint) 

    def execute_workload(self, mid=None, workload_type=None) -> tuple:
        if mid is not None and workload_type is not None:
            w = self.workloads[mid][workload_type]
            if w.execute():
                self.memory_usage += self.activation_memory
                return (self.stage_id, w.microbatch_id, w.workload_type)
        elif mid is not None:
            mwl = self.workloads[mid]
            for wlt in mwl:
                w = mwl[wlt]
                if w.execute():
                    self.memory_usage += self.activation_memory
                    return (self.stage_id, w.microbatch_id, w.workload_type)
        else:
            for mid in self.workloads:
                mwl = self.workloads[mid]
                for wlt in mwl:
                    w = mwl[wlt]
                    if w.execute():
                        self.memory_usage += self.activation_memory
                        return (self.stage_id, w.microbatch_id, w.workload_type)
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.stage_id}, "
                f"memory_usage={self.memory_usage}, "
                f"activation_memory={self.activation_memory})")
