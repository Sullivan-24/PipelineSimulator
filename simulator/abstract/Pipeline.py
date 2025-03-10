from .Device import Device, Schedule, RunMode, MemoryMonitor
from .Stage import Stage
from .Workload import Workload
from .mutils import *
from ..painter import SchedulingPainter as SP
from ..LayerwisePainter import LayerwiseSchedulingPainter as LSP
from ..utils import print_to_file
from .Placement import PipelinePlacement
import itertools
import json
import os
import copy
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
        self.microbatch_schedule_range = range(0,min(SCHEDULE_UNIT, MICRO_BATCH_NUM))
        # self.microbatch_schedule_range = range(0,min(8, MICRO_BATCH_NUM))
        self.acc_finished_mb = 0
        self.finish_flag = False
        self.num_finished_microbatch = 0
        self.run_schedule = run_schedule
        self.manual_recomp_set = []
        self.fail_indexes = set()
        # pp4 tp4 zero4 I1F1B recomp set
        # self.manual_recomp_set = [0 for _ in range(LAYER_NUM)]
        # self.manual_recomp_set[2] = 1
        # self.manual_recomp_set[3] = 1
        # self.manual_recomp_set[6] = 1
        # self.manual_recomp_set[7] = 1
        # self.manual_recomp_set[11] = 1
        # self.manual_recomp_set[15] = 1

        min_value = min(list(reversed(range(LAYER_NUM))))
        max_value = max(list(reversed(range(LAYER_NUM))))

        # 缩放到 DENSITY_MIN 和 DENSITY_MAX 之间
        self.layer_density = [
            DENSITY_MIN + (value - min_value) * (DENSITY_MAX - DENSITY_MIN) / (max_value - min_value)
            for value in list(reversed(range(LAYER_NUM)))
        ]
        self._init_stage()
        self.set_microbatch_schedule_range(microbatch_schedule_range=self.microbatch_schedule_range)
        self.schedule = [[] for _ in range(DEVICE_NUM)]
        self.generate_schedule()
        self.set_schedule()
        self.temp_results = {}
        self.recomp_set_traverser = self.generate_binary_combinations()
        self.last_workload: Workload = None
        self.workload_execute_record: list[list[Workload]] = [[] for _ in range(DEVICE_NUM)]
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
            t = self.results[key]
            self.schedule[did].append((workload_type_mapping[k], mid, sid, t))
        print("Result to schedule successfully.")

    def result2file(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'w') as file:
            json.dump(self.results, file)
        print("Result to file successfully.")

    def file2result(self, filepath=None):
        if filepath is None:
            filepath = 'data.txt'
        with open(filepath, 'r') as file:
            results = json.load(file)
            self.results = {}
            for k in results.keys():
                if str(k).startswith(("f","b","w")):
                    self.results[k] = results[k]

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
            print("Device ID:{}".format(device.did))
            if device.did == 7:
                device.show_stages(detail_info=True)

    def set_microbatch_schedule_range(self, microbatch_schedule_range):
        for device in self.devices:
            device.mid_traverse_order = microbatch_schedule_range

    def record_recomp_set(self):
        for idx, r in enumerate(self.recomp_set):
            self.recomp_set[idx] = 1 if r else 0
            self.results[f"theta_{idx}"] = r

    def _init_stage(self):
        dev_compute_power = []
        for did in range(DEVICE_NUM):
            max_mem = GPU_MAX_MEM
            comp_power = 1
            if not HOMO_DEVICE:
                if did >= DEVICE_NUM // 2:
                    max_mem = GPU_MAX_MEM / 2 
                    comp_power = comp_power / 2
            device = Device(
                        device_id = did, 
                        max_activation_counts=MAX_ACTIVATION_COUNTS, 
                        nmb=MICRO_BATCH_NUM,
                        memory_usage_constrain_rate=0.85,
                        max_mem=max_mem,
                        comp_power=comp_power,
                        layer_density=self.layer_density,
                    )
            dev_compute_power.append(comp_power)
            self.devices.append(device)
        self.set_recomp()
        if not HOMO_DEVICE and not self.dsa:
            layer_computation_cost = [1 for _ in range(LAYER_NUM)]
            self.pipeline_placement_solver = PipelinePlacement(
                layer_num=LAYER_NUM,
                layer_computation_cost=layer_computation_cost,
                layer_para=[1 for _ in range(LAYER_NUM)],
                dev_num=DEVICE_NUM,
                dev_max_memory=[100000 for _ in range(DEVICE_NUM)],
                dev_compute_power=dev_compute_power,
            )
            self.dsa = self.pipeline_placement_solver.get_placements()
            if LAYERWISE:
                assert False, 'Layerwise test not ready'
        if self.dsa:
            for did in range(DEVICE_NUM):
                for pid in self.dsa[did]:
                    self.devices[did].add_stage(pid, recomp=self.recomp_set[pid])
            if LAYERWISE:
                self.devices[DEVICE_NUM - 1].add_stage(0, recomp=self.recomp_set[0])
        elif LAYERWISE:
            if STAGE_PLACEMENT == Placement.INTERLEAVED:
                print("Use Interleaved placement")
                if HOMO_DEVICE:
                    for pid in range(LAYER_NUM):
                        self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid])
                else:
                    for pid in range(DEVICE_NUM * 0):
                        self.devices[pid % (DEVICE_NUM // 2)].add_stage(pid + 1, recomp=self.recomp_set[pid])
                    for pid in range(DEVICE_NUM * 0, LAYER_NUM):
                        self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid])
                    # for pid in range(LAYER_NUM - DEVICE_NUM * 3):
                    #     self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid])
                    # for pid in range(LAYER_NUM - DEVICE_NUM * 3, LAYER_NUM):
                    #     self.devices[pid % (DEVICE_NUM // 2)].add_stage(pid + 1, recomp=self.recomp_set[pid])
            elif STAGE_PLACEMENT == Placement.RECURRENT:
                print("Use Recurrent placement")
                unit = range(DEVICE_NUM)
                orders = []
                while len(orders) < LAYER_NUM:
                    unit = list(unit)
                    orders += unit[:-1]
                    unit = reversed(unit)

                for pid in range(LAYER_NUM - 1):
                    self.devices[orders[pid]].add_stage(pid + 1, recomp=self.recomp_set[pid])
                self.devices[-1].add_stage(LAYER_NUM, recomp=self.recomp_set[pid])
            elif STAGE_PLACEMENT == Placement.CROSS:
                print("Use V+I placement")
                for pid in range(LAYER_NUM):
                    if (pid // (DEVICE_NUM)) == LAYER_NUM // (DEVICE_NUM) - 1:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
                    else:
                        self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
            else:   
                print("Use Wavelike placement")
                offset = DEVICE_NUM if REVERSE_LAST_STAGES else 0
                print(f"Reverse last {offset} stages.")
                for pid in range(LAYER_NUM - offset):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])
                for pid in range(LAYER_NUM - offset, LAYER_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid + 1, recomp=self.recomp_set[pid + 1])

            if STAGE_PLACEMENT != Placement.RECURRENT:
                self.devices[-1].add_stage(0)
                self.devices[0].add_stage(LAYER_NUM+1)
                self.devices[1].add_stage(LAYER_NUM+2)
            else:
                self.devices[-1].add_stage(0)
                self.devices[0].add_stage(LAYER_NUM+1)
                self.devices[1].add_stage(LAYER_NUM+2)
        else:
            if SCHEDULE_METHOD in (Schedule.INTERLEAVED, Schedule.STANDARD_INTERLEAVED):
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid])
            elif STAGE_PLACEMENT == Placement.INTERLEAVED:
                print("Use Interleaved placement")
                for pid in range(STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid])
            else:
                assert STAGE_NUM <= LAYER_NUM, f"STAGE should be less than LAYER ({STAGE_NUM} >= {LAYER_NUM})"                
                offset = DEVICE_NUM if REVERSE_LAST_STAGES else 0
                for pid in range(STAGE_NUM - offset):
                    if (pid // DEVICE_NUM) % 2 == 0:
                        self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid])
                    else:
                        self.devices[DEVICE_NUM - 1 - pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid])
                for pid in range(STAGE_NUM - offset, STAGE_NUM):
                    self.devices[pid % DEVICE_NUM].add_stage(pid, recomp=self.recomp_set[pid])
        # Launch MemoryMonitors
        for device in self.devices:
            device.init_required_mem_for_each_microbatch()
            device.init_memory_monitor()


        for did in range(DEVICE_NUM):
            print(list(self.devices[did].stages.keys()))
            if len(self.dsa) < DEVICE_NUM:
                self.dsa.append(list(self.devices[did].stages.keys()))

        if os.path.exists(PLA_FILE_PATH):
            os.remove(PLA_FILE_PATH)
            print("delete file:{}".format(PLA_FILE_PATH))
        print_to_file(PLA_FILE_PATH,str(self.dsa))

    def set_recomp(self):
        if LAYERWISE:
            self.recomp_set = [1 if RECOMP else 0 for _ in range(LAYER_NUM + 3)]
        else:
            self.recomp_set = [1 if RECOMP else 0 for _ in range(STAGE_NUM)]
        if self.manual_recomp_set:
            print("Use manual recomp set")
            self.recomp_set = self.manual_recomp_set
            return
        print("Set recomputation")
    def set_schedule(self):
        for did in range(DEVICE_NUM):
            self.devices[did].static_schedule = self.schedule[did]

    def generate_schedule(self):
        if SCHEDULE_METHOD == Schedule.STANDARD_1F1B:
            self.generate_1f1b_schedule()
            print("Generate STANDARD_1F1B Schedule.")

        elif SCHEDULE_METHOD == Schedule.STANDARD_AFAB:
            self.generate_afab_schedule()
            print("Generate STANDARD_AFAB Schedule.")

        elif SCHEDULE_METHOD == Schedule.STANDARD_ZBH1:
            self.generate_zbh1_schedule()
            print("Generate STANDARD_ZBH1 Schedule.")

        elif SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED and not LAYERWISE:
            self.generate_interleaved_1f1b_schedule()
            print("Generate STANDARD_INTERLEAVED Schedule.")
        else:
            print("Using UPP Schedule.")

    def generate_afab_schedule(self):
        assert CHUNK_NUM == 1
        print("Generate standard AFAB schedule...")
        workload_type_order = [WorkloadType.F, WorkloadType.B]
        if SPLIT_BACKPROP:
            workload_type_order.append(WorkloadType.W)

        for did in range(DEVICE_NUM):
            mids = [0 for _ in range(WORKLOAD_TYPE_NUM)]
            for i in range(WORKLOAD_TYPE_NUM):
                while mids[i] < MICRO_BATCH_NUM:
                    self.schedule[did].append((workload_type_order[i], mids[i], did))
                    mids[i]+=1

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
    
    def show_record(self):
        for k in self.results:
            print(k, self.results[k])

    def update_constraints(self, constraint):
        for device in self.devices:
            device.update_constraints(constraint=constraint)

    def record_workload(self, workload: Workload):
        if workload:
            wlt = workload.wtype.value.lower()
            mid = workload.mid
            sid = workload.sid
            k = '{}_{}_{}'.format(wlt,mid,sid)
            self.results[k] = workload.start_time
            if self.last_workload is None or workload.start_time + workload.duration > self.last_workload.start_time + self.last_workload.duration:
                self.last_workload = workload

    def update_workload_execution_record(self):
        for device in self.devices:
            device.workload_execute_record = self.workload_execute_record

    def change_mid_traverse_order(self, workload: Workload):
        if workload:
            for device in self.devices:
                device.overlap_aware_executable_workload_reorder(workload=workload)         

    def check_workload_status(self):
        for device in self.devices:
            if device._finish_proc_workload():
                if device.proc_workload.wtype == WorkloadType.W:
                    self.num_finished_microbatch += 1
                    self.acc_finished_mb += 1
                    if LAYERWISE:
                        if self.acc_finished_mb == (1 + LAYER_NUM) * MICRO_BATCH_NUM:
                            self.finish_flag = True
                    else:
                        if self.acc_finished_mb == STAGE_NUM * MICRO_BATCH_NUM:
                            self.finish_flag = True
                # if not SPLIT_BACKPROP and device.proc_workload.wtype == WorkloadType.B:
                #     if device.proc_workload.sid == 0:
                #         self.num_finished_microbatch += 1
                #         self.acc_finished_mb += 1
                #     if LAYERWISE:
                #         if self.acc_finished_mb == (1 + LAYER_NUM) * MICRO_BATCH_NUM:
                #             self.finish_flag = True
                #     else:
                #         if self.acc_finished_mb == STAGE_NUM * MICRO_BATCH_NUM:
                #             self.finish_flag = True 
                self.workload_execute_record[device.proc_workload.did].append(device.proc_workload)
                self.update_workload_execution_record()

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
            
    def reduce_recomp_degree(self):
        
        self.manual_recomp_set = self.recomp_set
        for index, value in reversed(list(enumerate(self.manual_recomp_set))):
            if value:
                if index not in self.fail_indexes:
                    self.manual_recomp_set[index] = 0
                    print("Index {}".format(index))
                    return index
        print("Already the best recomp config.")
        return -1
    
    def add_recomp_degree(self):
        recomp_set = self.recomp_set
        for index, value in list(enumerate(recomp_set)):
            if not value:
                self.recomp_set[index] = 1
                print("Try the added the recomp degree.")
                return True
        print("Set all stage to recomputing.")
        return False
    
    def reset_run_para(self):
        self.results = {}
        self.devices: list[Device] = []
        self.dsa = []
        self.acc_finished_mb = 0
        self.finish_flag = False
        self.num_finished_microbatch = 0
        self._init_stage()
        RESET_TIME()
    
    def generate_binary_combinations(self):
        """
        生成所有长度为 layer_num 的二进制组合0 或 1。
        每次调用 next() 返回一个未返回过的情况。
        """
        # 使用 itertools.product 生成所有可能的组合
        combinations = itertools.product([1, 0], repeat=LAYER_NUM)
        return combinations

    def recomp_set_check(self, recomp_set:list):
        # [43.37751770019531, 43.25251770019531, 43.12751770019531, 43.00251770019531, 42.87751770019531, 42.75251770019531, 42.62751770019531, 72.66658020019531]
        # Activation Layer=6.0625,Activation Input=0.0625,Activation Loss=4.640625
        # Gradient Input=1.5001983642578125,Gradient Parameters=1.5001983642578125,Gradient Head=2.3203125
        # LOSS=4.640625,VOC=152064
        # LAYER_MEM:1.5001983642578125
        # MODEL MEM:15.0
        # OPT MEM:11.0
        if SCHEDULE_METHOD == Schedule.STANDARD_INTERLEAVED:
            if sum(recomp_set[DEVICE_NUM-1::DEVICE_NUM]) < 8:
                return False
            # cut branch for 1st device
            if sum(recomp_set[0::DEVICE_NUM]) != 7 and recomp_set[0] != 1:
                return False
            
            last_recomp_num = LAYER_NUM + 1
            for did in range(DEVICE_NUM):
                layer_num_wo_recomp = recomp_set[did::DEVICE_NUM]
                if sum(layer_num_wo_recomp) <= 7:
                    return False
                if sum(layer_num_wo_recomp) > last_recomp_num:
                    return False
                last_recomp_num = sum(layer_num_wo_recomp)
        return True

    def run_pipeline_parallelism(self, time_limit = TIME_LIMIT):
        # self.run_schedule = False
        RESET_TIME()
        while GET_TIME() <= time_limit and not self.finish_flag:
            self.check_workload_status()
            self.execute_workload()
            UPDATE_TIME()
        if self.finish_flag:
            print("Success")
            self.record_recomp_set()
            if not self.show_mem_usage():
                print("Fail due to OOM")
            else:
                self.temp_results = copy.deepcopy(self.results)
        else:
            print("Fail")
        # print(self.results)
        # input("RUN OVER")
        if AUTO_RECOMP_SEARCH:
            # self.record_recomp_set()
            # self.result2file()
            self.run_schedule = True
            fail_times = 0
            while fail_times < DEVICE_NUM:
                idx = self.reduce_recomp_degree()
                if idx == -1:
                    break
                self.reset_run_para()
                print("Read schedule generated before...")
                # self.file2result()
                # self.result2schedule()
                self.set_schedule()

                print(self.recomp_set)
                
                while GET_TIME() <= time_limit and not self.finish_flag:
                    self.check_workload_status()
                    self.execute_workload()
                    UPDATE_TIME()
                if self.finish_flag:
                    self.record_recomp_set()
                    if not self.show_mem_usage():
                        print("Fail OOM")
                        fail_times += 1
                        self.fail_indexes.add(idx)
                        self.recomp_set[idx] = 1
                    else:
                        print("Success")
                        self.temp_results = copy.deepcopy(self.results)
                        print("Reset fail times to 0.")
                        fail_times = 0
                else:
                    print("Fail")
                    self.results = self.temp_results
                    break
            
            self.reset_run_para()
            self.results = self.temp_results
            # self.result2schedule()
            self.set_schedule()
            while GET_TIME() <= time_limit and not self.finish_flag:
                self.check_workload_status()
                self.execute_workload()
                UPDATE_TIME()
            if self.finish_flag:
                print("Success")
                self.record_recomp_set()
                self.result2file()
            else:
                print("Wrong answer!")


    def show_mem_usage(self, device_id=(0,1, DEVICE_NUM-1), show_all=False):
        max_mem_usages = [0 for _ in range(len(self.devices))]
        for device in self.devices:
            aim_file_path = "schedule_results/memory/device{}.txt".format(device.did)
            print_to_file(aim_file_path, "Device {} mem usage:\n".format(device.did))
            last_mem_record = 0
            for t, mem_record in device.mem_usage_record.items():
                oom_flag = "" if mem_record <= device.max_memory else "OOM"
                print_to_file(aim_file_path, "Time {}, mem = {}, {}, {}.\n".format(t, round(mem_record,2), round((mem_record - last_mem_record), 2), oom_flag))
                last_mem_record = mem_record
                max_mem_usages[device.did] = max(max_mem_usages[device.did], mem_record)
        
        oom = False
        for did, device in enumerate(self.devices):
            if device.max_memory < max_mem_usages[did]:
                print(f"Out of Memory in Device {device.did}. ({max_mem_usages[did]} > {device.max_memory})")
                oom = True

        print(max_mem_usages)
        return not oom
    
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
        if LAYERWISE:
            layers = 1
        else:
            layers = LAYER_NUM // STAGE_NUM

        if workload_type == "f":
            workload_len = F_TIME * layers
            if LAYERWISE:
                if lid == 0:
                    workload_len = EMB_TIME
                elif lid == LAYER_NUM - 1:
                    workload_len = CE_F_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_F_TIME
            else:
                if lid == 0:
                    workload_len += EMB_TIME
                elif lid == STAGE_NUM - 1:
                    workload_len += CE_F_TIME + HEAD_F_TIME
        elif workload_type == "b":
            workload_len = B_TIME * layers
            if LAYERWISE:
                if lid == LAYER_NUM - 1:
                    workload_len = CE_B_TIME
                elif lid == LAYER_NUM - 2:
                    workload_len = HEAD_B_TIME
            else:
                if lid == STAGE_NUM - 1:
                    workload_len += CE_B_TIME + HEAD_B_TIME
        elif workload_type == "w":
            workload_len = W_TIME * layers
            if LAYERWISE:
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
                print_to_file(f"sim_{SCHEDULE_METHOD}_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key},{self.results[key]}\n")
                
    def draw(self) -> None:
        # 绘制结果的逻辑
        # self.resort_w()
        # self.write_fbw_to_file()
        fwd_time, iwd_time, pwd_time = self.get_workloadload_duration()
        if LAYERWISE:
            painter_conf = {
                "device_num": DEVICE_NUM,
                "devices": self.dsa,
                "num_layer": LAYER_NUM+3,
                "stage_num": LAYER_NUM+3,
                "pp_height": PP_HEIGHT,
                "pp_align": PP_ALIGN,
                "pixel_base": PIXEL_BASE,
                "nmb": MICRO_BATCH_NUM,
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
            painter_conf = {
                "device_num": DEVICE_NUM,
                "devices": self.dsa,
                "stage_num": STAGE_NUM,
                "pp_height": PP_HEIGHT,
                "pp_align": PP_ALIGN,
                "pixel_base": PIXEL_BASE,
                "nmb": MICRO_BATCH_NUM,
                "forward_length": fwd_time,
                "backward_length": iwd_time,
                "backward_length2": pwd_time,
                "comm_length": [COMM_TIME for _ in range(STAGE_NUM)],
            }
            SP(painter_conf).draw(res)