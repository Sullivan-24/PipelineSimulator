from .mutils import *
class WorkloadProcInfo:

    def __init__(self, 
                 microbatch_id: int, 
                 stage_id: int, 
                 duration: int, 
                 workload_type: WorkloadType,
                 state: int) -> None:
        
        self.microbatch_id: int = microbatch_id  # 微批次编号
        self.stage_id: int = stage_id              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = state                      # 初始状态为未开始
        self.workload_type: WorkloadType = workload_type  # 工作负载类型


class Workload:
    # 定义状态常量
    NOT_STARTED = 1
    IN_PROGRESS = 2
    COMPLETED = 3

    def __init__(self, microbatch_id: int, stage_id: int, duration: int, workload_type: WorkloadType):
        self.microbatch_id: int = microbatch_id  # 微批次编号
        self.stage_id: int = stage_id              # 阶段编号
        self.duration: int = duration               # 任务所需时间
        self.start_time: float = None               # 初始开始时间为None
        self.end_time: float = None                 # 结束时间
        self.state: int = Workload.NOT_STARTED      # 初始状态为未开始
        self.workload_type: WorkloadType = workload_type  # 工作负载类型
        self.constraints: set = set()               # {(i1, j1, C1), ...}表示Stage i1 上的Microbatch j1 的 C1 操作需要完成
        self._generate_constraints()

    def _generate_constraints(self):
        if self.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            if self.stage_id - 1 >= 0:
                self.constraints.add((self.stage_id-1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD))
        elif self.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            if self.stage_id + 1 < STAGE_NUM:
                self.constraints.add((self.stage_id+1, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))
            self.constraints.add((STAGE_NUM - 1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD))
        elif self.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.constraints.add((self.stage_id, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))

    def update_constraints(self, constraint: tuple):
        self.constraints.discard(constraint)

    def execute(self) -> bool:
        if self.state == Workload.NOT_STARTED:
            if len(self.constraints) > 0:
                pass
                # print("T={},\tConstraints of S={},\tMB={} are not satisfied...".format(
                #     GET_TIME(),
                #     self.stage_id,
                #     self.microbatch_id
                # ))
            else: 
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
