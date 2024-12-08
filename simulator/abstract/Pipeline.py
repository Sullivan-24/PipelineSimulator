from .Device import Device, SchedulePriority
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter

class PipelineScheduler:

    def __init__(self, dsa = None) -> None:
        self.results = {}
        self.devices: list[Device] = []
        self.dsa = [] if not dsa else dsa 
        self.schedule = [[] for _ in range(DEVICE_NUM)]
        self.generate_schedule()
        self._init_stage()

    def _init_stage(self, stage_type=Stage.VSHAPE):
        for did in range(DEVICE_NUM):
            device = Device(device_id = did, 
                            max_activation_counts=MAX_ACTIVATION_COUNTS, 
                            static_schedule=self.schedule[did],
                            )
            self.devices.append(device)

        if self.dsa:
            for did in range(DEVICE_NUM):
                for pid in self.dsa[did]:
                    self.devices[did].add_stage(pid)
        else:
            if stage_type == Stage.VSHAPE:
                for pid in range(STAGE_NUM):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid)
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid)
            elif stage_type == Stage.INTERLEAVED:
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid)

        for did in range(DEVICE_NUM):
            self.devices[did].show_stages()
            if len(self.dsa) < DEVICE_NUM:
                self.dsa.append(self.devices[did].stages.keys())

    def generate_schedule(self):
        if SCHEDULE_METHOD == SchedulePriority.ONE_F_ONE_B:
            self.generate_1f1b_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.ZBH1:
            self.generate_zbh1_schedule()
        else:
            print("Selected Schedule is Not Supported")

    def generate_1f1b_schedule(self):
        workload_type_num = 2
        workload_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_idx_in_mids = {WorkloadType.FORWARD_PASS_WORKLOAD: 0, WorkloadType.INPUT_GRADIENT_WORKLOAD : 1}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(workload_type_num)]
            # warmup
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, mids[0], did))
                mids[0] += 1
            
            iter = 0
            finish_flag = [0 for _ in range(workload_type_num)]
            while sum(finish_flag) < workload_type_num:
                next_workload_type = workload_type_order[iter % workload_type_num]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1
        for did in range(DEVICE_NUM):
            for (wt, mid, sid) in self.schedule[did]:
                print(wt.value, mid, end=" ")
            print()
    
    def generate_zbh1_schedule(self):
        def custom_round(number):
            import math
            if number - math.floor(number) >= 0.5:
                return math.ceil(number)
            else:
                return math.floor(number)
            
        workload_type_num = 3
        warmup_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.PARAMETER_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_idx_in_mids = {WorkloadType.FORWARD_PASS_WORKLOAD: 0, WorkloadType.INPUT_GRADIENT_WORKLOAD : 1, WorkloadType.PARAMETER_GRADIENT_WORKLOAD : 2}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(workload_type_num)]
            # warmup
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, mids[0], did))
                mids[0] += 1
            
            # Inject as much as possible F with limited max activation counts
            if MAX_ACTIVATION_COUNTS > DEVICE_NUM * CHUNK_NUM:
                comm_delay = (DEVICE_NUM - did - 1) * 2 * COMM_TIME
                compute_delay = (DEVICE_NUM - did - 1) * IGW_TIME
                additional_f_num = min(MAX_ACTIVATION_COUNTS - mids[0] - did, custom_round((comm_delay + compute_delay) / FPW_TIME))
                while additional_f_num:
                    self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD ,mids[0], did))
                    mids[0] += 1
                    additional_f_num -= 1

            # iter = 0
            # while iter < did * 2:
            #     if mids[0] == MICRO_BATCH_NUM:
            #         break
            #     next_workload_type = warmup_type_order[iter % 2]
            #     next_mid = mids[workload_idx_in_mids[next_workload_type]]
            #     if next_mid < MICRO_BATCH_NUM:
            #         self.schedule[did].append((next_workload_type, next_mid, did))
            #         mids[workload_idx_in_mids[next_workload_type]] += 1
            #     else:
            #         finish_flag[workload_idx_in_mids[next_workload_type]] = 1
            #     iter+=1

            # steady + cooldown
            iter = 0
            finish_flag = [0 for _ in range(workload_type_num)]
            while sum(finish_flag) < workload_type_num:
                next_workload_type = workload_type_order[iter % workload_type_num]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if MAX_ACTIVATION_COUNTS > mids[0]:
                    if next_workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                        iter += 1
                        continue 
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1

        for did in range(DEVICE_NUM):
            for (wt, mid, sid) in self.schedule[did]:
                print(wt.value, mid, end=" ")
            print()

    def show_record(self):
        for k in self.results:
            print(k, self.results[k])

    def update_constraints(self, constraint):
        for device in self.devices:
            device.update_constraints(constraint=constraint)

    def record_workload(self, workload: Workload):
        if workload:
            wlt = workload.workload_type.value.lower()
            mid = workload.microbatch_id
            sid = workload.stage_id
            k = '{}_{}_{}'.format(wlt,mid,sid)
            self.results[k] = workload.start_time

    def check_workload_status(self):
        for device in self.devices:
            if device._finish_proc_workload():
                device.proc_workload.complete()
                self.update_constraints(constraint=device.proc_workload)
                device.update_memory_usage()
                device.state = Device.IDLE

    def execute_workload(self):
        for device in self.devices:
            processing_workload = device.execute_workload()
            self.record_workload(processing_workload)
            
    def run_pipeline_parallelism(self, time_limit = 1000):
        while GET_TIME() <= time_limit:
            self.check_workload_status()
            self.execute_workload()
            UPDATE_TIME()

    def show_mem_usage(self):
        for device in self.devices:
            print("Device {} mem usage:".format(device.device_id))
            for t, mem_record in device.mem_usage_record.items():
                print("Time {}, mem = {}.".format(t, mem_record))

    def draw(self) -> None:
        # 绘制结果的逻辑
        painter_conf = {
            "device_size": DEVICE_NUM,
            "devices": self.dsa,
            "pp_size": STAGE_NUM,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 2,
            "num_microbatches": MICRO_BATCH_NUM,
            "forward_length": [FPW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "backward_length": [IGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "backward_length2": [PGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
            "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
        }

        SchedulingPainter(painter_conf).draw(self.results)