from .mutils import *
    
class Workload:
    # 定义状态常量
    NOT_STARTED = 1
    IN_PROGRESS = 2
    COMPLETED = 3

    def __init__(self, device_id:int, microbatch_id: int, stage_id: int, duration: int, total_stages:int, workload_type: WorkloadType, recomp:bool):
        self.did = device_id
        self.mid: int = microbatch_id  # 微批次编号
        self.sid: int = stage_id              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = Workload.NOT_STARTED      # 初始状态为未开始
        self.ready_time: int = -1
        self.total_stages: int = total_stages
        self.recomp:bool = recomp
        if self.mid == 0 and self.sid == 0:
            self.ready_time = 0
        self.workload_type: WorkloadType = workload_type  # 工作负载类型
        self.constraints: set = set()               # {(i1, j1, C1), ...}表示Stage i1 上的Microbatch j1 的 C1 操作是前置约束
        self._generate_constraints()

    def _generate_constraints(self):
        if self.workload_type == WorkloadType.F:
            if self.sid > 0:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        microbatch_id = self.mid,
                        stage_id = self.sid - 1,
                        workload_type = WorkloadType.F)
                )
        elif self.workload_type == WorkloadType.B:
            if self.sid + 1 < self.total_stages:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id = self.sid+1, 
                        microbatch_id= self.mid, 
                        workload_type = WorkloadType.B)
                )
            else:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.did,
                        stage_id=self.total_stages - 1, 
                        microbatch_id=self.mid, 
                        workload_type = WorkloadType.F)
                )
        elif self.workload_type == WorkloadType.W:
            self.constraints.add(
                WorkloadConstraint(
                    device_id = self.did,
                    stage_id=self.sid, 
                    microbatch_id=self.mid, 
                    workload_type=WorkloadType.B)
            )

    def _generate_communication(self, constraint: WorkloadConstraint):
        if constraint.did != self.did:
            self.ready_time = max(self.ready_time, GET_TIME() + COMM_TIME)
        else:
            self.ready_time = max(self.ready_time, GET_TIME())

    def update_constraints(self, constraint: WorkloadConstraint):
        origin_len = len(self.constraints)
        self.constraints.discard(constraint)
        now_len = len(self.constraints)

        if origin_len != now_len:
            self._generate_communication(constraint)

    def is_executable(self):
        return len(self.constraints) == 0 and self.ready_time <= GET_TIME() and self.state == Workload.NOT_STARTED
    
    def execute(self) -> bool:
        if self.state == Workload.NOT_STARTED:
            if self.is_executable():
                self.state = Workload.IN_PROGRESS
                self.start_time = GET_TIME()
                self.end_time = self.start_time + self.duration
                return True
        return False

    def complete(self) -> None:
        """完成任务并更新状态"""
        if self.state == Workload.IN_PROGRESS and self.end_time <= GET_TIME():
            self.state = Workload.COMPLETED

    def __repr__(self):
        return (f"{self.__class__.__name__}(device_id={self.did}, "
            f"microbatch_id={self.mid}, stage_id={self.sid}, "
            f"duration={self.duration}, start_time={self.start_time}, "
            f"end_time={self.end_time}, state={self.state}, "
            f"ready_time={self.ready_time}, total_stages={self.total_stages}, "
            f"workload_type={self.workload_type}, constraints={self.constraints})")
