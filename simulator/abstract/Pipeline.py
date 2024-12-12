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
        self._init_stage()

        self.schedule = [[] for _ in range(DEVICE_NUM)]
        self.generate_schedule()
        self.set_schedule()

    def _init_stage(self):
        for did in range(DEVICE_NUM):
            device = Device(device_id = did, 
                            max_activation_counts=MAX_ACTIVATION_COUNTS, 
                            # static_schedule=self.schedule[did],
                            )
            self.devices.append(device)

        if self.dsa:
            for did in range(DEVICE_NUM):
                for pid in self.dsa[did]:
                    self.devices[did].add_stage(pid)
        else:
            if SCHEDULE_METHOD in (SchedulePriority.ONE_F_ONE_B, SchedulePriority.ZBH1, SchedulePriority.ZBV, SchedulePriority.GREEDY):
                for pid in range(STAGE_NUM):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid)
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid)
            elif SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid)

        for did in range(DEVICE_NUM):
            self.devices[did].show_stages()
            if len(self.dsa) < DEVICE_NUM:
                self.dsa.append(self.devices[did].stages.keys())

    def set_schedule(self):
        for did in range(DEVICE_NUM):
            self.devices[did].static_schedule = self.schedule[did]

    def generate_schedule(self):
        if SCHEDULE_METHOD == SchedulePriority.ONE_F_ONE_B:
            self.generate_1f1b_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.ZBH1:
            self.generate_zbh1_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.ZBV:
            self.generate_zbv_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
            self.generate_interleaved_1f1b_schedule()
        else:
            print("Generate Greedy Schedule instead.")

    def generate_1f1b_schedule(self):
        assert WORKLOAD_TYPE_NUM == 2
        assert not SPLIT_BACKPROP
        assert CHUNK_NUM == 1
        workload_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_idx_in_mids = {WorkloadType.FORWARD_PASS_WORKLOAD: 0, WorkloadType.INPUT_GRADIENT_WORKLOAD : 1}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            # warmup
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, mids[0], did))
                mids[0] += 1
            
            iter = 0
            finish_flag = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            while sum(finish_flag) < WORKLOAD_TYPE_NUM:
                next_workload_type = workload_type_order[iter % WORKLOAD_TYPE_NUM]
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
        assert WORKLOAD_TYPE_NUM == 3

        workload_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.PARAMETER_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_idx_in_mids = {WorkloadType.FORWARD_PASS_WORKLOAD: 0, WorkloadType.INPUT_GRADIENT_WORKLOAD : 1, WorkloadType.PARAMETER_GRADIENT_WORKLOAD : 2}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            # warmup, should not be simplified
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, mids[0], did))
                mids[0] += 1
            # Inject as much as possible F with limited max activation counts
            if MAX_ACTIVATION_COUNTS > STAGE_NUM * CHUNK_NUM:
                comm_delay = (DEVICE_NUM - did - 1) * 2 * COMM_TIME
                compute_delay = (DEVICE_NUM - did - 1) * IGW_TIME
                additional_f_num = min(MAX_ACTIVATION_COUNTS - mids[0] - did, (comm_delay + compute_delay) // FPW_TIME)
                while mids[0] < min(MAX_ACTIVATION_COUNTS, MICRO_BATCH_NUM) and additional_f_num:
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
            finish_flag = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            while sum(finish_flag) < WORKLOAD_TYPE_NUM:
                next_workload_type = workload_type_order[iter % WORKLOAD_TYPE_NUM]
                next_mid = mids[workload_idx_in_mids[next_workload_type]]
                if mids[0] < min(MICRO_BATCH_NUM, MAX_ACTIVATION_COUNTS):
                    if next_workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                        iter += 1
                        continue 
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1
                print(finish_flag)

        for did in range(DEVICE_NUM):
            for (wt, mid, sid) in self.schedule[did]:
                print(wt.value, mid, end=" ")
            print()

    def generate_zbv_schedule(self):
        assert WORKLOAD_TYPE_NUM == 3
        real_f_len = FPW_TIME // CHUNK_NUM
        real_b_len = IGW_TIME // CHUNK_NUM
        real_w_len = PGW_TIME // CHUNK_NUM
        comm_time  = COMM_TIME
        workload_type_order = [WorkloadType.INPUT_GRADIENT_WORKLOAD, WorkloadType.PARAMETER_GRADIENT_WORKLOAD, WorkloadType.FORWARD_PASS_WORKLOAD]
        workload_idx_in_mids = {WorkloadType.FORWARD_PASS_WORKLOAD: 0, WorkloadType.INPUT_GRADIENT_WORKLOAD : 1, WorkloadType.PARAMETER_GRADIENT_WORKLOAD : 2}
        warmup_f_num_in_first_device = -1
        for did in range(DEVICE_NUM):
            [sid_1, sid_2] = self.dsa[did]
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM * CHUNK_NUM)]
            # warmup, should not be simplified
            warmup_start_time = did * (real_f_len + comm_time)
            warmup_endup_time = (DEVICE_NUM * (comm_time + real_f_len) - comm_time) * 2 - real_f_len - did * (real_f_len + comm_time)
            injectable_time_len = warmup_endup_time - warmup_start_time
            additional_f_num = injectable_time_len // real_f_len
            while mids[0] < MICRO_BATCH_NUM and mids[0] < MAX_ACTIVATION_COUNTS - 1 and additional_f_num:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD ,mids[0], did))
                mids[0] += 1
                additional_f_num -= 1
            
            if did == 0:
                warmup_f_num_in_first_device = mids[0] + 1

            last_sid = sid_1
            while (mids[0] < MICRO_BATCH_NUM or mids[0 + WORKLOAD_TYPE_NUM] < MICRO_BATCH_NUM) and mids[0] + mids[0 + WORKLOAD_TYPE_NUM] < warmup_f_num_in_first_device:
                if last_sid == sid_1:
                    next_mid = mids[0 + WORKLOAD_TYPE_NUM]
                    if next_mid < did + 1:
                        self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, next_mid, sid_2))
                        mids[0 + WORKLOAD_TYPE_NUM] += 1
                    last_sid = sid_2
                elif last_sid == sid_2:
                    next_mid = mids[0]
                    if next_mid < warmup_f_num_in_first_device - did - 1:
                        self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD, next_mid, sid_1))
                        mids[0] += 1
                    last_sid = sid_1
            
            previous_sid2_f_num = mids[0 + WORKLOAD_TYPE_NUM]
            iter = 0
            # while mids[0 + WORKLOAD_TYPE_NUM] < DEVICE_NUM:
            while mids[0 + WORKLOAD_TYPE_NUM] < previous_sid2_f_num + DEVICE_NUM - did:
                next_workload_type = workload_type_order[iter % WORKLOAD_TYPE_NUM]
                next_mid = mids[workload_idx_in_mids[next_workload_type] + WORKLOAD_TYPE_NUM]
                if mids[0] + mids[0 + WORKLOAD_TYPE_NUM] < MAX_ACTIVATION_COUNTS :
                    if next_workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                        iter += 1
                        continue 
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, sid_2))
                    mids[workload_idx_in_mids[next_workload_type] + WORKLOAD_TYPE_NUM] += 1
                iter+=1

            # steady + cooldown
            continue
            iter = 0
            finish_flag = [0 for _ in range(WORKLOAD_TYPE_NUM * CHUNK_NUM)]
            while sum(finish_flag) < WORKLOAD_TYPE_NUM * CHUNK_NUM:
                next_workload_type = workload_type_order[iter % WORKLOAD_TYPE_NUM]
                next_sid = sid_1 if last_sid == sid_2 else sid_2
                mids_sid_offset = 0 if next_sid == sid_1 else WORKLOAD_TYPE_NUM
                next_mid = mids[workload_idx_in_mids[next_workload_type] + mids_sid_offset]
                if mids[0] + mids[0 + WORKLOAD_TYPE_NUM] < MAX_ACTIVATION_COUNTS :
                    if next_workload_type == WorkloadType.PARAMETER_GRADIENT_WORKLOAD:
                        iter += 1
                        continue 
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type] + mids_sid_offset] += 1
                    last_sid = next_sid
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type] + mids_sid_offset] = 1
                iter+=1

        for did in range(DEVICE_NUM):
            for (wt, mid, sid) in self.schedule[did]:
                print((wt.value, mid, sid), end=" ")
            print()
            print()

    def generate_interleaved_1f1b_schedule(self):
        assert WORKLOAD_TYPE_NUM == 2
        for did in range(DEVICE_NUM):
            sids = list(self.dsa[did])
            
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM * CHUNK_NUM)]
            f_mid_count = 0
            
            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * WORKLOAD_TYPE_NUM
        
            # warmup, inject as much microbatches as possible
            warmup_f_num = (CHUNK_NUM - 1) * DEVICE_NUM + (DEVICE_NUM - did - 1) * 2
            while mids[idx_in_f_mids] < MICRO_BATCH_NUM and f_mid_count < warmup_f_num:
                self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD ,mids[idx_in_f_mids], f_next_sid))
                mids[idx_in_f_mids] += 1
                f_mid_count += 1
                if f_mid_count % DEVICE_NUM == 0:
                    f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                    f_next_sid = sids[f_next_sid_idx]
                    idx_in_f_mids = f_next_sid_idx * WORKLOAD_TYPE_NUM

            b_mid_count = 0
            bsids = list(reversed(sids))
            b_next_sid_idx = 0
            b_next_sid = bsids[b_next_sid_idx]
            idx_in_b_mids = 1 + b_next_sid_idx * WORKLOAD_TYPE_NUM

            # Start 1f1b with F operation
            operation_flag = 'f'
            while b_mid_count + f_mid_count < MICRO_BATCH_NUM * CHUNK_NUM * WORKLOAD_TYPE_NUM:
                if operation_flag == 'f':
                    if mids[idx_in_f_mids] < MICRO_BATCH_NUM:
                        self.schedule[did].append((WorkloadType.FORWARD_PASS_WORKLOAD ,mids[idx_in_f_mids], f_next_sid))
                        mids[idx_in_f_mids] += 1
                        f_mid_count += 1
                        if f_mid_count % DEVICE_NUM == 0:
                            f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                            f_next_sid = sids[f_next_sid_idx]
                            idx_in_f_mids = f_next_sid_idx * WORKLOAD_TYPE_NUM
                    operation_flag = 'b'
                elif operation_flag == 'b':
                    if mids[idx_in_b_mids] < MICRO_BATCH_NUM:
                        self.schedule[did].append((WorkloadType.INPUT_GRADIENT_WORKLOAD ,mids[idx_in_b_mids], b_next_sid))
                        mids[idx_in_b_mids] += 1
                        b_mid_count += 1
                        if b_mid_count % DEVICE_NUM == 0:
                            b_next_sid_idx = (b_next_sid_idx + 1) % len(bsids)
                            b_next_sid = bsids[b_next_sid_idx]
                            idx_in_b_mids = 1 + b_next_sid_idx * WORKLOAD_TYPE_NUM
                    operation_flag = 'f'
                else:
                    raise("UNKOWN OPERATION FLAG")
        
        # for did in range(DEVICE_NUM):
        #     for (wt, mid, sid) in self.schedule[did]:
        #         print((wt.value, mid, sid), end=" ")
        #     print()
        #     print()
    
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
            
    def run_pipeline_parallelism(self, time_limit = 10000):
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