from .Workload import *
class MicrobatchWorkloadRecorder:
    def __init__(self, recorder_id: int):
        self.recorder_id: int = recorder_id
        self.workloads_constraints: list = list()
        self._generate_recorder_matrix()

    def _generate_constraints(self, sid, mid):
        if self.workload_type == WorkloadType.FORWARD_PASS_WORKLOAD:
            self.constraints.add((self.stage_id-1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD))
        elif self.workload_type == WorkloadType.INPUT_GRADIENT_WORKLOAD:
            self.constraints.add((self.stage_id+1, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))
        elif self.workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
            self.constraints.add((self.stage_id, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD))

    def _generate_recorder_matrix(self):
        self.workloads_constraints = [
            [
                [
                    (self.stage_id-1, self.microbatch_id, WorkloadType.FORWARD_PASS_WORKLOAD),
                    (self.stage_id+1, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD),
                    (self.stage_id, self.microbatch_id, WorkloadType.INPUT_GRADIENT_WORKLOAD)
                ] for mid in range(MICRO_BATCH_NUM)
            ] for sid in range(STAGE_NUM)
        ]