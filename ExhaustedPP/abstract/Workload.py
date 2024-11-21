from .mutils import *
class WorkloadConstraint:

    def __init__(self, 
                 device_id: int,
                 microbatch_id: int, 
                 stage_id: int, 
                 workload_type: WorkloadType,
                ) -> None:
        self.device_id = device_id
        self.microbatch_id: int = microbatch_id  # 微批次编号
        self.stage_id: int = stage_id              # 阶段编号
        self.workload_type: WorkloadType = workload_type  # 工作负载类型

    def __eq__(self, other):
        if not isinstance(other, WorkloadConstraint):
            return NotImplemented
        return (
            self.microbatch_id == other.microbatch_id 
            and self.stage_id == other.stage_id
            and self.workload_type == other.workload_type
        )
    
    def __hash__(self):
        return hash((self.microbatch_id, self.stage_id, self.workload_type))
    
class Workload:
    # 定义状态常量
    NOT_STARTED = 1
    IN_PROGRESS = 2
    COMPLETED = 3

    def __init__(self, device_id:int, microbatch_id: int, stage_id: int, duration: int, workload_type: WorkloadType):
        self.device_id = device_id
        self.microbatch_id: int = microbatch_id  # 微批次编号
        self.stage_id: int = stage_id              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = Workload.NOT_STARTED      # 初始状态为未开始
        self.ready_time: int = -1
        if self.microbatch_id == 0 and self.stage_id == 0:
            self.ready_time = 0
        self.workload_type: WorkloadType = workload_type  # 工作负载类型
        self.constraints: set = set()               # {(i1, j1, C1), ...}表示Stage i1 上的Microbatch j1 的 C1 操作需要完成
        self._generate_constraints()

    def _generate_constraints(self):
        if self.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            if self.stage_id - 1 >= 0:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.device_id,
                        microbatch_id = self.microbatch_id,
                        stage_id = self.stage_id - 1,
                        workload_type = WorkloadType.FORWARD_PASS_WORKLOAD)
                )
        elif self.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            if self.stage_id + 1 < STAGE_NUM:
                self.constraints.add(
                    WorkloadConstraint(
                        device_id = self.device_id,
                        stage_id = self.stage_id+1, 
                        microbatch_id= self.microbatch_id, 
                        workload_type = WorkloadType.INPUT_GRADIENT_WORKLOAD)
                )
            self.constraints.add(
                WorkloadConstraint(
                    device_id = self.device_id,
                    stage_id=STAGE_NUM - 1, 
                    microbatch_id=self.microbatch_id, 
                    workload_type = WorkloadType.FORWARD_PASS_WORKLOAD)
            )
        elif self.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.constraints.add(
                WorkloadConstraint(
                    device_id = self.device_id,
                    stage_id=self.stage_id, 
                    microbatch_id=self.microbatch_id, 
                    workload_type=WorkloadType.INPUT_GRADIENT_WORKLOAD)
            )
        # if self.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
        #     if self.stage_id - 1 >= 0:
        #         self.constraints.add((self.stage_id-1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD))
        # elif self.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
        #     if self.stage_id + 1 < STAGE_NUM:
        #         self.constraints.add((self.stage_id+1, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))
        #     self.constraints.add((STAGE_NUM - 1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD))
        # elif self.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
        #     self.constraints.add((self.stage_id, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))

    def _generate_communication(self, constraint: WorkloadConstraint):
        if constraint.device_id != self.device_id:
            self.ready_time = max(self.ready_time, GET_TIME() + COMM_TIME)
        else:
            self.ready_time = max(self.ready_time, GET_TIME())

    def update_constraints(self, constraint: WorkloadConstraint):
        origin_len = len(self.constraints)
        self.constraints.discard(constraint)
        now_len = len(self.constraints)

        if origin_len != now_len:
            self._generate_communication(constraint)

    def execute(self) -> bool:
        if self.state == Workload.NOT_STARTED:
            if len(self.constraints) > 0:
                pass
                # print("T={},\tConstraints of S={},\tMB={} are not satisfied...".format(
                #     GET_TIME(),
                #     self.stage_id,
                #     self.microbatch_id
                # ))
            # Add comm overhead
            elif self.ready_time <= GET_TIME(): 
                print("T={},\tS={},\tMB={}-{} is in progress...".format(
                    GET_TIME(),
                    self.stage_id,
                    self.microbatch_id,
                    self.workload_type.value
                ))
                self.state = Workload.IN_PROGRESS
                self.start_time = GET_TIME()
                self.end_time = self.start_time + self.duration
                return True
        return False

    def complete(self) -> None:
        """完成任务并更新状态"""
        if self.state == Workload.IN_PROGRESS and self.end_time <= GET_TIME():
            print("T={},\tS={},\tMB={}-{} is completed.".format(
                GET_TIME(),
                self.stage_id,
                self.microbatch_id,
                self.workload_type.value,
            ))
            self.state = Workload.COMPLETED

    def __repr__(self) -> str:
        return (f"{self.workload_type.value}(microbatch_id={self.microbatch_id}, "
                f"stage_id={self.stage_id}, duration={self.duration}, "
                f"state={self.state})")
