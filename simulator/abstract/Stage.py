from .Workload import *
import copy

@dataclass
class StageType:
    EMBD = 1
    LAYER = 2
    HEAD = 3
    CE = 4
    LAYERS = 5

def get_workload_duration(sid:int, layer_wise:bool, layer_num:int, wtype:WorkloadType, recomp, comp_power:float = 1)->float:
    if layer_wise:
        assert layer_num == 1, f"LAYERWISE require 1 layer per stage but got {layer_num}."
    if wtype == WorkloadType.F:
        duration = gpc["F_TIME"] * layer_num
        if layer_wise:
            if sid == 0:
                duration = gpc["EMB_TIME"]
            elif sid == gpc["LAYER_NUM"] + 1: # Single Head Layer
                duration = gpc["HEAD_F_TIME"]
            elif sid == gpc["LAYER_NUM"] + 2:
                duration = gpc["CE_F_TIME"]
        else:
            if sid == 0:
                duration += gpc["EMB_TIME"]
            elif sid == gpc["STAGE_NUM"] - 1:
                duration += gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]
    elif wtype == WorkloadType.B: # Consider recomputation
        duration = (gpc["B_TIME"] + gpc["F_TIME"] * recomp) * layer_num
        if layer_wise:
            if sid == gpc["LAYER_NUM"] + 1: # Single Head Layer
                duration = gpc["HEAD_B_TIME"] + gpc["HEAD_F_TIME"] * recomp
            elif sid == gpc["LAYER_NUM"] + 2:
                duration = gpc["CE_B_TIME"] + gpc["CE_F_TIME"] * recomp
        else:
            if sid == gpc["STAGE_NUM"] - 1:
                duration += gpc["HEAD_B_TIME"] + gpc["CE_B_TIME"] + (gpc["HEAD_F_TIME"] + gpc["CE_F_TIME"]) * recomp
                if not gpc["SPLIT_BACKPROP"]:
                    duration += gpc["HEAD_W_TIME"]
    elif wtype == WorkloadType.W:
        duration = gpc["W_TIME"] * layer_num
        if layer_wise:
            if sid == gpc["LAYER_NUM"] + 1: # Cross-entropy Layer has no parameter to train
                duration = gpc["HEAD_W_TIME"]
        else:
            if sid == gpc["STAGE_NUM"] - 1:
                duration += gpc["HEAD_W_TIME"]
    else:
        raise ValueError(f"Wrong workload type: {wtype}.")
    
    return int(duration / comp_power)

class Stage:
    
    INTERLEAVED = 1
    VSHAPE = 2
    WAVELIKE = 3

    def __init__(self, device_id:int, stage_id: int, para_num:int, stage_type: StageType, microbatch_num:int, layer_num: int, layerwise:bool = False, recomp: bool = False, comp_power: float = 1, layer_density: list=None):
        self.did: int = device_id
        self.sid: int = stage_id
        self.nmb: int = microbatch_num
        self.para_num: int = para_num / gpc["G"]
        self.model_memory_usage = self.para_num * gpc["FP16"] / gpc["TP_SIZE"]
        self.grad_memory_usage = self.para_num * gpc["FP16"] / gpc["TP_SIZE"]
        self.opt_memory_usage = self.para_num * 3 * gpc["FP32"] / gpc["TP_SIZE"] / gpc["ZERO_SIZE"]
        self.memory_usage: int = self.model_memory_usage + self.grad_memory_usage + self.opt_memory_usage
        self.workloads: dict[int, dict[WorkloadType, Workload]] = {}  
        self.stage_type: StageType = stage_type
        self.recomp = recomp
        self.layerwise = layerwise
        self.layer_num = layer_num
        self.comp_power = comp_power
        if layerwise: 
            assert layer_num == 1, f"LAYERWISE require 1 layer per stage but got {layer_num}"
        if layer_density is None:
            self.layer_density = [1 for _ in range(gpc["LAYER_NUM"])]
        else:
            self.layer_density = layer_density
        self._add_workload()
        
    def _add_workload(self) -> None:
        for mid in range(self.nmb):
            self.workloads[mid] = {}
            fpw = Workload(
                device_id=self.did,
                stage_id=self.sid,
                microbatch_id=mid,
                wtype=WorkloadType.F,
                duration=get_workload_duration(
                    sid=self.sid,
                    layer_wise=self.layerwise,
                    layer_num=self.layer_num,
                    wtype=WorkloadType.F,
                    recomp=self.recomp,
                    comp_power=self.comp_power,
                ),
                recomp=self.recomp,
                total_stages=gpc["LAYER_NUM"]+3 if self.layerwise else gpc["STAGE_NUM"],
            )
            self.workloads[mid][WorkloadType.F] = fpw
            if self.sid == 0 and self.layerwise: # Embedding layer only has F
                continue

            igw = Workload(
                device_id=self.did,
                stage_id=self.sid,
                microbatch_id=mid,
                wtype=WorkloadType.B,
                duration=get_workload_duration(
                    sid=self.sid,
                    layer_wise=self.layerwise,
                    layer_num=self.layer_num,
                    wtype=WorkloadType.B,
                    recomp=self.recomp,
                    comp_power=self.comp_power,
                ), 
                recomp=self.recomp,
                total_stages=gpc["LAYER_NUM"]+3 if self.layerwise else gpc["STAGE_NUM"],   
            )
            self.workloads[mid][WorkloadType.B] = igw
            if gpc["SPLIT_BACKPROP"]:
                pgw = Workload(
                    device_id=self.did,
                    stage_id=self.sid,
                    microbatch_id=mid,
                    wtype=WorkloadType.W,
                    duration=get_workload_duration(
                        sid=self.sid,
                        layer_wise=self.layerwise,
                        layer_num=self.layer_num,
                        wtype=WorkloadType.W,
                        recomp=self.recomp,
                        comp_power=self.comp_power,
                    ),
                    recomp=self.recomp,
                    total_stages=gpc["LAYER_NUM"]+3 if self.layerwise else gpc["STAGE_NUM"],
                )
                if self.stage_type == StageType.CE: # Cross-entropy layer has no W for parameter training
                    continue
                self.workloads[mid][WorkloadType.W] = pgw

    def update_constraints(self, time, constraint: Workload):
        for mid in self.workloads:
            for wlt in self.workloads[mid]:
                self.workloads[mid][wlt].update_constraints(
                    time,
                    WorkloadConstraint(
                        device_id=constraint.did,
                        stage_id=constraint.sid, 
                        microbatch_id=constraint.mid, 
                        workload_type=constraint.wtype
                    )
                ) 

    def update_memory_usage(self, workload:Workload):
        if self.layerwise:
            if self.stage_type == StageType.EMBD:
                return
            if workload.wtype == WorkloadType.F:
                if self.stage_type == StageType.HEAD:
                    self.memory_usage += Activation.LOSS
                # TODO CE memory cost
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    if self.recomp:
                        self.memory_usage += Activation.INPUT
                        return
                    self.memory_usage += Activation.FULL
            elif workload.wtype == WorkloadType.B:
                if self.stage_type == StageType.HEAD:                    
                    self.memory_usage -= Activation.LOSS
                elif self.stage_type == StageType.CE:
                    self.memory_usage += 0
                else:
                    if gpc["SPLIT_BACKPROP"]:
                        self.memory_usage += Gradient.INPUT
                        if self.recomp:
                            self.memory_usage += (Activation.FULL - Activation.INPUT)
                    else:
                        self.memory_usage -= Activation.FULL
            elif workload.wtype == WorkloadType.W:
                if self.stage_type in (StageType.HEAD, StageType.CE):
                    return
                if gpc["SPLIT_BACKPROP"]:
                    self.memory_usage -= Gradient.INPUT + Activation.FULL
        else:
            layers_per_stage = gpc["LAYER_NUM"] // gpc["STAGE_NUM"]
            if workload.wtype == WorkloadType.F:
                self.memory_usage += (Activation.FULL * (1 - self.recomp) + Activation.INPUT * self.recomp) * layers_per_stage
                if self.sid == gpc["STAGE_NUM"] - 1:
                    self.memory_usage += Activation.LOSS
            elif workload.wtype == WorkloadType.W:
                self.memory_usage -= (Activation.FULL + Gradient.INPUT) * layers_per_stage
                if self.sid == gpc["STAGE_NUM"] - 1:
                    self.memory_usage -= Activation.LOSS
            elif gpc["SPLIT_BACKPROP"] and workload.wtype == WorkloadType.B:
                self.memory_usage += ((Activation.FULL - Activation.INPUT) * self.recomp + Gradient.INPUT) * layers_per_stage
            elif gpc["SPLIT_BACKPROP"] == False and workload.wtype == WorkloadType.B:
                self.memory_usage -= (Activation.FULL * (1 - self.recomp) + Activation.INPUT * self.recomp) * layers_per_stage
                if self.sid == gpc["STAGE_NUM"] - 1:
                    self.memory_usage -= Activation.LOSS
        
    def execute_workload(self, time, mid=None, workload_type=None)->Workload:
        if mid is not None and workload_type is not None and workload_type in self.workloads[mid]:
            w = self.workloads[mid][workload_type]
            if w.execute(time=time):
                return copy.deepcopy(w)
        else:
            if self.stage_type == StageType.EMBD and workload_type in (WorkloadType.B, WorkloadType.W):
                return None
            elif self.stage_type == StageType.CE and workload_type in (WorkloadType.W, ):
                return None
            print("Lack of workload info.")
        return None

    def __repr__(self) -> str:
        return (f"StageClass(stage_id={self.sid}, "
                f"memory_usage={self.memory_usage})")
