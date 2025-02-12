from simulator.config import *
# 现在有N个F workload，N个B workload，N个W workload，
def get_base_workload_num(device_num=DEVICE_NUM, layer_num=LAYER_NUM, mb=MICRO_BATCH_NUM):
    return layer_num//device_num*mb

def get_additional_workload(device_id, device_num=DEVICE_NUM, mb=MICRO_BATCH_NUM):
    if device_id == 0:
        return mb
    # TODO
    if device_id == DEVICE_NUM - 1:
        return mb
    if device_id == DEVICE_NUM - 2:
        return mb

def get_required_memory(deivice_id, layer_id, workload_type, layer_num=LAYER_NUM):
    if workload_type == "F":
        return Activation.FULL
    elif workload_type == "B":
        return Gradient.INPUT
    else:
        return Gradient.PARAMETER
    
def get_generated_memory(deivice_id, layer_id, workload_type, layer_num=LAYER_NUM):
    if workload_type == "F":
        return Activation.FULL
    elif workload_type == "B":
        return Gradient.INPUT
    else:
        return -(Activation.FULL+Gradient.INPUT)

class LightWorkload:
    def __init__(self, device_id, stage_id, mb_id, workload_type):
        self.device_id = device_id
        self.stage_id = stage_id
        self.mb_id = mb_id
        self.workload_type = workload_type

    def __hash__(self):
        # 使用元组中的字段生成哈希值
        return hash((self.device_id, self.stage_id, self.mb_id, self.workload_type))

    def __eq__(self, other):
        # 比较两个对象的字段
        if not isinstance(other, LightWorkload):
            return NotImplemented
        return (self.device_id == other.device_id and
                self.stage_id == other.stage_id and
                self.mb_id == other.mb_id and
                self.workload_type == other.workload_type)
    def __repr__(self):
        return ("{}{}".format(self.workload_type,self.mb_id))
    
class WorkloadQueue:
    def __init__(self, base_mem=0):
        self.mem = base_mem
        self.workloads: list[LightWorkload] = []
        self.results = []
    
    def save_result(self):
        self.results.append(copy.deepcopy(self.workloads))
        if len(self.results) // 10 == 0:
            print(len(self.results))
            self.show_workloads()

    def show_workloads(self):
        for wl in self.workloads:
            print(wl, end=" ")
        print()

    def add_workload(self, workload:LightWorkload):
        self.workloads.append(workload)
        self.mem += get_generated_memory(workload.device_id, workload.stage_id, workload.workload_type)

    def pop_workload(self):
        self.mem -= get_generated_memory(self.workloads[-1].device_id, self.workloads[-1].stage_id, self.workloads[-1].workload_type)
        return self.workloads.pop()

import copy
import queue

def memory_check(now_mem, required_mem, max_mem=GPU_MAX_MEM):
    return now_mem + required_mem < max_mem

def depend_check(workload:LightWorkload, f_workload:set, b_workload:set, w_workload:set):
    if workload.workload_type == "B":
        return LightWorkload(0,0,workload.mb_id, "F") not in f_workload
    if workload.workload_type == "W":
        return LightWorkload(0,0,workload.mb_id, "B") not in b_workload
    return True


def add_workload(workload_queue : WorkloadQueue, fp, bp, wp, f_workload:set, b_workload:set, w_workload:set):
    # workload_queue.show_workloads()
    if fp == MICRO_BATCH_NUM and bp == MICRO_BATCH_NUM:
        workload_queue.save_result()
        return 
    available_mem_for_f = True
    available_mem_for_b = True

    for workload in b_workload:
        mem = get_required_memory(workload.device_id, workload.stage_id, workload.workload_type)
        if memory_check(workload_queue.mem, required_mem=mem) and depend_check(workload,f_workload,b_workload,w_workload):
            workload_queue.add_workload(workload)
            b_workload.discard(workload)
            bp += 1
            add_workload(workload_queue, fp, bp, wp, f_workload, b_workload, w_workload)
            bp -= 1
            b_workload.add(workload)
            workload_queue.pop_workload()
        else:
            available_mem_for_b = False

    for workload in f_workload:
        mem = get_required_memory(workload.device_id, workload.stage_id, workload.workload_type)
        if memory_check(workload_queue.mem, required_mem=mem) and depend_check(workload,f_workload,b_workload,w_workload):
            workload_queue.add_workload(workload)
            f_workload.discard(workload)
            fp += 1
            add_workload(workload_queue, fp, bp, wp, f_workload, b_workload, w_workload)
            fp -= 1
            f_workload.add(workload)
            workload_queue.pop_workload()
        else:
            available_mem_for_f = False
    if not available_mem_for_f and not available_mem_for_b:
        for workload in w_workload:
            mem = get_required_memory(workload.device_id, workload.stage_id, workload.workload_type)
            if memory_check(workload_queue.mem, required_mem=mem) and depend_check(workload,f_workload,b_workload,w_workload):
                workload_queue.add_workload(workload)
                w_workload.discard(workload)
                wp += 1
                add_workload(workload_queue, fp, bp, wp, f_workload, b_workload, w_workload)
                wp -= 1
                w_workload.add(workload)
                workload_queue.pop_workload()


if __name__ == "__main__":
    total_res = []
    fp = 0
    bp = 0
    wp = 0
    f_workload = set()
    b_workload = set()
    w_workload = set()
    workload_queue = WorkloadQueue(0)
    for mid in range(MICRO_BATCH_NUM):
        f_workload.add(LightWorkload(0,0,mid,"F"))
        b_workload.add(LightWorkload(0,0,mid,"B"))
        w_workload.add(LightWorkload(0,0,mid,"W"))

    print(len(f_workload))
    add_workload(workload_queue, fp, bp, wp, f_workload, b_workload, w_workload)
    print(len(workload_queue.results))
    