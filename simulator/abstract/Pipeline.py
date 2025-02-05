from .Device import Device, SchedulePriority, RunMode
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter as SP
from ..LayerwisePainter import LayerwiseSchedulingPainter as LSP
from ..utils import print_to_file
import json

workload_type_mapping = {
    'f':WorkloadType.F,
    'b':WorkloadType.B,
    'w':WorkloadType.W,
}

class PipelineScheduler:

    def __init__(self, dsa = None, run_schedule=False) -> None:
        self.results = {}
        self.devices: list[Device] = []
        self.dsa = [] if not dsa else dsa 
        self.microbatch_schedule_range = range(0,min(MICRO_BATCH_NUM//1, MICRO_BATCH_NUM))
        # self.microbatch_schedule_range = range(0,min(8, MICRO_BATCH_NUM))
        self.num_finished_microbatch = 0
        self.run_schedule = run_schedule
        self._init_stage()
        self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)
        self.schedule = [[] for _ in range(DEVICE_NUM)]
        self.generate_schedule()
        self.set_schedule()

        if run_schedule:
            print("Read schedule generated before...")
            self.file2result()
            self.result2schedule()
            self.set_schedule()

    def _sid2did(self, sid):
        for did, sids in enumerate(self.dsa):
            if sid in sids:
                return did
            
    def result2schedule(self):
        for key in self.results.keys():
            if not key.startswith(("f_","b_","w_",)):
                continue
            k, mid, sid = key.split("_")
            sid = int(sid)
            mid = int(mid)
            did = self._sid2did(sid=sid)
            self.schedule[did].append((workload_type_mapping[k], mid, sid))

    def result2file(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'w') as file:
            json.dump(self.results, file)

    def file2result(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'r') as file:
            self.results = json.load(file)

    # NOTE _reset_workload_type is efficient but 
    # lead to random order of W in some cases
    # which will break solver constraint (not affect the correctness)
    def resort_w(self):
        w_times = [[] for _ in range(LAYER_NUM + 3)]
        for res in self.results:
            if res.startswith("w"):
                w,mid,sid = res.split("_")
                sid = int(sid)
                w_times[sid].append(self.results[res])

        for sid in range(LAYER_NUM + 3):
            w_times_in_sid = sorted(w_times[sid])
            for mid in range(len(w_times_in_sid)):
                w_key = f"w_{mid}_{sid}"
                self.results[w_key] = w_times_in_sid[mid]


    def show_detail_info(self):
        for device in self.devices:
            print("Device ID:{}".format(device.device_id))
            if device.device_id == 7:
                device.show_stages(detail_info=True)

    def set_microbatch_schedule_range(self, microbatch_schedule_range):
        for device in self.devices:
            device.microbatch_schedule_range = microbatch_schedule_range

    def set_layer_recomp(self, settings:list):
        for idx, r in enumerate(settings):
            self.recomp_set[idx] = True if r else False
            self.results[f"theta_{idx}"] = r

    def _init_stage(self):
        for did in range(DEVICE_NUM):
            device = Device(device_id = did, 
                            max_activation_counts=MAX_ACTIVATION_COUNTS, 
                            nmb=MICRO_BATCH_NUM,
                            )
            self.devices.append(device)

        if self.dsa:
            for did in range(DEVICE_NUM):
                for pid in self.dsa[did]:
                    self.recomp_set = [0 for _ in range(LAYER_NUM + 3)]
                    if SEQ_LEN > 4*K:
                        recomp_list = [0] + [1 for _ in range(LAYER_NUM)] + [0,0]
                        self.set_layer_recomp(recomp_list)
                    self.devices[did].add_stage(pid, recomp=self.recomp_set[pid])
        elif SCHEDULE_METHOD == SchedulePriority.Layerwise:
            self.recomp_set = [0 for _ in range(LAYER_NUM + 3)]
            if SEQ_LEN > 4*K:
                recomp_list = [0] + [1 for _ in range(LAYER_NUM)] + [0,0]
                self.set_layer_recomp(recomp_list)
                # self.set_layer_recomp([0, #Embedding
                #                        0,0,0,0,0,0,0,0,
                #                        0,0,0,0,0,0,0,0,
                #                        0,0 # Head+CE
                #                     ]
                #                 )
            for pid in range(LAYER_NUM):
                if (pid // DEVICE_NUM) % 2 == 0:
                    self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
                else:
                    self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
            self.devices[-1].add_stage(0)
            self.devices[0].add_stage(LAYER_NUM+1)
            self.devices[1].add_stage(LAYER_NUM+2)
        else:
            self.recomp_set = [0 for _ in range(STAGE_NUM)]
            self.set_layer_recomp(self.recomp_set)
            if SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid + 1])
            else:
                for pid in range(STAGE_NUM):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid + 1])
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid + 1])

        for did in range(DEVICE_NUM):
            self.devices[did].show_stages()
            if len(self.dsa) < DEVICE_NUM:
                self.dsa.append(list(self.devices[did].stages.keys()))

    def set_schedule(self):
        for did in range(DEVICE_NUM):
            self.devices[did].static_schedule = self.schedule[did]

    def generate_schedule(self):
        if SCHEDULE_METHOD == SchedulePriority.ONE_F_ONE_B:
            self.generate_1f1b_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.ZBH1:
            self.generate_zbh1_schedule()
        elif SCHEDULE_METHOD == SchedulePriority.INTERLEAVED:
            self.generate_interleaved_1f1b_schedule()
        else:
            print("Generate Greedy Schedule instead.")

    def generate_1f1b_schedule(self):
        assert WORKLOAD_TYPE_NUM == 2
        assert not SPLIT_BACKPROP
        assert CHUNK_NUM == 1
        workload_type_order = [WorkloadType.B, WorkloadType.F]
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            # warmup
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.F, mids[0], did))
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
        # for did in range(DEVICE_NUM):
        #     for (wt, mid, sid) in self.schedule[did]:
        #         print(wt.value, mid, end=" ")
        #     print()

    def generate_zbh1_schedule(self):
        assert WORKLOAD_TYPE_NUM == 3

        workload_type_order = [WorkloadType.B, WorkloadType.W, WorkloadType.F]
        workload_idx_in_mids = {WorkloadType.F: 0, WorkloadType.B : 1, WorkloadType.W : 2}
        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            # warmup, should not be simplified
            while mids[0] < DEVICE_NUM - did:
                self.schedule[did].append((WorkloadType.F, mids[0], did))
                mids[0] += 1
            # Inject as much as possible F with limited max activation counts
            if MAX_ACTIVATION_COUNTS > STAGE_NUM * CHUNK_NUM:
                comm_delay = (DEVICE_NUM - did - 1) * 2 * COMM_TIME
                compute_delay = (DEVICE_NUM - did - 1) * B_TIME
                additional_f_num = min(MAX_ACTIVATION_COUNTS - mids[0] - did, (comm_delay + compute_delay) // F_TIME)
                while mids[0] < min(MAX_ACTIVATION_COUNTS, MICRO_BATCH_NUM) and additional_f_num:
                    self.schedule[did].append((WorkloadType.F ,mids[0], did))
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
                    if next_workload_type == WorkloadType.W:
                        iter += 1
                        continue 
                if next_mid < MICRO_BATCH_NUM:
                    self.schedule[did].append((next_workload_type, next_mid, did))
                    mids[workload_idx_in_mids[next_workload_type]] += 1
                else:
                    finish_flag[workload_idx_in_mids[next_workload_type]] = 1
                iter+=1

        # for did in range(DEVICE_NUM):
        #     for (wt, mid, sid) in self.schedule[did]:
        #         print(wt.value, mid, end=" ")
        #     print()

    def generate_interleaved_1f1b_schedule(self):
        workload_type_num = 2
        for did in range(DEVICE_NUM):
            sids = list(self.dsa[did])
            
            mids = [0 for _ in range(workload_type_num * CHUNK_NUM)]
            f_mid_count = 0
            
            f_next_sid_idx = 0
            f_next_sid = sids[f_next_sid_idx]
            idx_in_f_mids = f_next_sid_idx * workload_type_num
        
            # warmup, inject as much microbatches as possible
            warmup_f_num = (CHUNK_NUM - 1) * DEVICE_NUM + (DEVICE_NUM - did - 1) * 2
            while mids[idx_in_f_mids] < MICRO_BATCH_NUM and f_mid_count < warmup_f_num:
                self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids], f_next_sid))
                mids[idx_in_f_mids] += 1
                f_mid_count += 1
                if f_mid_count % DEVICE_NUM == 0:
                    f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                    f_next_sid = sids[f_next_sid_idx]
                    idx_in_f_mids = f_next_sid_idx * workload_type_num

            b_mid_count = 0
            bsids = list(reversed(sids))
            b_next_sid_idx = 0
            b_next_sid = bsids[b_next_sid_idx]
            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num

            # Start 1f1b with F operation
            operation_flag = 'f'
            while b_mid_count + f_mid_count < MICRO_BATCH_NUM * CHUNK_NUM * workload_type_num:
                if operation_flag == 'f':
                    if mids[idx_in_f_mids] < MICRO_BATCH_NUM:
                        self.schedule[did].append((WorkloadType.F ,mids[idx_in_f_mids], f_next_sid))
                        mids[idx_in_f_mids] += 1
                        f_mid_count += 1
                        if f_mid_count % DEVICE_NUM == 0:
                            f_next_sid_idx = (f_next_sid_idx + 1) % len(sids)
                            f_next_sid = sids[f_next_sid_idx]
                            idx_in_f_mids = f_next_sid_idx * workload_type_num
                    operation_flag = 'b'
                elif operation_flag == 'b':
                    if mids[idx_in_b_mids] < MICRO_BATCH_NUM:
                        self.schedule[did].append((WorkloadType.B ,mids[idx_in_b_mids], b_next_sid))
                        if WORKLOAD_TYPE_NUM == 3:
                            self.schedule[did].append((WorkloadType.W ,mids[idx_in_b_mids], b_next_sid))
                        mids[idx_in_b_mids] += 1
                        b_mid_count += 1
                        if b_mid_count % DEVICE_NUM == 0:
                            b_next_sid_idx = (b_next_sid_idx + 1) % len(bsids)
                            b_next_sid = bsids[b_next_sid_idx]
                            idx_in_b_mids = 1 + b_next_sid_idx * workload_type_num
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
                if device.proc_workload.workload_type == WorkloadType.W:
                    self.num_finished_microbatch += 1
                device.proc_workload.complete()
                self.update_constraints(constraint=device.proc_workload)
                device.update_memory_usage()
                device.state = Device.IDLE

        if self.num_finished_microbatch == (1 + LAYER_NUM) * len(self.microbatch_schedule_range):
            self.num_finished_microbatch = 0
            self.microbatch_schedule_range = [n + len(self.microbatch_schedule_range) for n in self.microbatch_schedule_range if n + len(self.microbatch_schedule_range) < MICRO_BATCH_NUM]
            self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)

    def execute_workload(self):
        for device in self.devices:
            processing_workload = device.execute_workload(run_schedule=self.run_schedule)
            self.record_workload(processing_workload)
        # if GET_TIME()==302:
        #     print(self.devices[1].stages[15].workloads[0])
            
            
    def run_pipeline_parallelism(self, time_limit = 10000):
        while GET_TIME() <= time_limit:
            self.check_workload_status()
            self.execute_workload()
            UPDATE_TIME()

    def show_mem_usage(self, device_id=(0,), show_all=False):
        max_mem_usage = -1
        for device in self.devices:
            if device.device_id in device_id:
                print("Device {} mem usage:".format(device.device_id))
                last_mem_record = 0
                for t, mem_record in device.mem_usage_record.items():
                    print("Time {}, mem = {}, {}.".format(t, round(mem_record,2), round((mem_record - last_mem_record), 2)))
                    last_mem_record = mem_record
                    max_mem_usage = max(max_mem_usage, mem_record)
        if max_mem_usage > GPU_MAX_MEM and SCHEDULE_METHOD == SchedulePriority.Layerwise:
            raise ValueError("Error: Out of Memory.")
        
        if show_all:
            max_mem_usage = [0 for _ in range(len(self.devices))]
            for did, device in enumerate(self.devices):
                for t, mem_record in device.mem_usage_record.items():
                    max_mem_usage[did] = max(max_mem_usage[did], mem_record)
            print(max_mem_usage)
    
    def get_workloadload_duration(self):
        fwd_time = [F_TIME for _ in range(LAYER_NUM+3)]
        iwd_time = [B_TIME for _ in range(LAYER_NUM+3)]
        pwd_time = [W_TIME for _ in range(LAYER_NUM+3)]
        for device in self.devices:
            for sid in device.stages:
                for mid in range(MICRO_BATCH_NUM):
                    fwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.F].duration
                    if WorkloadType.B in device.stages[sid].workloads[mid]:
                        iwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.B].duration
                    if WorkloadType.W in device.stages[sid].workloads[mid]:
                        pwd_time[sid] = device.stages[sid].workloads[mid][WorkloadType.W].duration
        return fwd_time, iwd_time, pwd_time
    
    def get_workload_len(self, key):
        workload_type, mid, lid = key.split("_")
        mid = int(mid)
        lid = int(lid)
        if SCHEDULE_METHOD == SchedulePriority.Layerwise:
            layers = 1
        else:
            layers = LAYER_NUM // STAGE_NUM

        if workload_type == "f":
            workload_len = F_TIME * layers
            if SCHEDULE_METHOD == SchedulePriority.Layerwise:
                if lid == 0:
                    workload_len = EMBEDDING_TIME
                elif lid == LAYER_NUM - 1:
                    workload_len = CE_F_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_F_TIME
            else:
                if lid == 0:
                    workload_len += EMBEDDING_TIME
                elif lid == STAGE_NUM - 1:
                    workload_len += CE_F_TIME + HEAD_F_TIME
        elif workload_type == "b":
            workload_len = B_TIME * layers
            if SCHEDULE_METHOD == SchedulePriority.Layerwise:
                if lid == LAYER_NUM - 1:
                    workload_len = CE_B_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_B_TIME
            else:
                if lid == STAGE_NUM - 1:
                    workload_len += CE_B_TIME + HEAD_B_TIME
        elif workload_type == "w":
            workload_len = W_TIME * layers
            if SCHEDULE_METHOD == SchedulePriority.Layerwise:
                if lid == LAYER_NUM - 1:
                    workload_len = CE_W_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_W_TIME
            else:
                if lid == STAGE_NUM - 1:
                    workload_len += CE_W_TIME + HEAD_W_TIME
        return workload_len 


    def write_fbw_to_file(self):
        for key in self.results:
            if key.startswith(("f_","b_","w_")):
                print_to_file(f"sim_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key},{self.results[key]}\n")
                
    def draw(self) -> None:
        # 绘制结果的逻辑
        self.resort_w()
        self.write_fbw_to_file()
        if SCHEDULE_METHOD == SchedulePriority.Layerwise:
            fwd_time, iwd_time, pwd_time = self.get_workloadload_duration()
            painter_conf = {
                "device_size": DEVICE_NUM,
                "devices": self.dsa,
                "num_layer": LAYER_NUM+3,
                "pp_size": LAYER_NUM+3,
                "pp_height": 50,
                "pp_align": 10,
                "pixel_base": PIXEL_BASE,
                "num_microbatches": MICRO_BATCH_NUM,
                "forward_length": fwd_time,
                "backward_length": iwd_time,
                "backward_length2": pwd_time,
                "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
            }
            LSP(painter_conf).draw(self.results)
        else:
            res = {}
            for key in self.results:
                if key.startswith(("f_","b_","w_")):
                    res[key] = self.results[key]
            
            layers_per_stage = LAYER_NUM // STAGE_NUM
            base_f = F_TIME * layers_per_stage
            base_b = B_TIME * layers_per_stage
            base_w = W_TIME * layers_per_stage
            painter_conf = {
                "device_size": DEVICE_NUM,
                "devices": self.dsa,
                "pp_size": STAGE_NUM,
                "pp_height": 50,
                "pp_align": 10,
                "pixel_base": PIXEL_BASE,
                "num_microbatches": MICRO_BATCH_NUM,
                "forward_length": [base_f + EMBEDDING_TIME if _ == 0 else (base_f + HEAD_F_TIME + CE_F_TIME if _ == STAGE_NUM - 1 else base_f) for _ in range(STAGE_NUM)],
                "backward_length": [base_b + EMBEDDING_TIME if _ == 0 else (base_b + HEAD_B_TIME + CE_B_TIME if _ == STAGE_NUM - 1 else base_b) for _ in range(STAGE_NUM)],
                "backward_length2": [base_w + EMBEDDING_TIME if _ == 0 else (base_w + HEAD_W_TIME + CE_W_TIME if _ == STAGE_NUM - 1 else base_w) for _ in range(STAGE_NUM)],
                # "backward_length": [IGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
                # "backward_length2": [PGW_TIME // CHUNK_NUM for _ in range(STAGE_NUM)],
                "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
            }
            SP(painter_conf).draw(res)