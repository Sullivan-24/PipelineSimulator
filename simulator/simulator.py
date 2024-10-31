"""
simulator package
"""
import itertools
import time
import z3
import copy
from .config import *
from .painter import SchedulingPainter
from .utils import resort_microbatch_index, print_to_file
from gurobipy import Model, GRB, quicksum

class GSPSimulator:
    """Simulator"""

    def __init__(self, config: dict, device_stage_alignments=None, new_comm_length=None) -> None:
        self._pp_size = config["pp_size"]
        self._device_size = config["device_size"]
        self._model_size = config["model_size"]
        self.virtual_stage = config["virtual_stage"]
        self._num_real_microbatches = config["num_microbatches"]
        self._num_microbatches = config["num_microbatches"] * self.virtual_stage[0]
        self._max_activation_counts = config["max_activation_counts"]
        self._basic_forward_length = config["forward_execution_time"]
        self._basic_backward_b_length = config["backward_execution_i_time"]
        self._basic_backward_w_length = config["backward_execution_g_time"]
        self._comm_length = config["communication_time"] if not new_comm_length else new_comm_length

        self._sequential_order_constraint_strategy = config["sequential_order_constraint_strategy"]

        self._forward_length = self._basic_forward_length
        self._backward_b_length = self._basic_backward_b_length
        self._backward_w_length = self._basic_backward_w_length

        assert isinstance(self._forward_length, (list, tuple)), "forward_execution_time must be list or tuple"
        assert isinstance(self._backward_b_length, (list, tuple)), "backward_execution_time must be list or tuple"
        assert isinstance(self._backward_w_length, (list, tuple)), "backward_execution_time must be list or tuple"

        assert self._sequential_order_constraint_strategy in (
            "strict",
            "double_interleaving",
            "full_interleaving",
        ), "sequential order constraint strategy is not supported"

        self.model = Model("GSPSimulator")
        self._recomputing_rate = []
        self._layers = []
        self._forward_offsets = [[] for _ in range(self._pp_size)]
        self._backward_b_offsets = [[] for _ in range(self._pp_size)]
        self._backward_w_offsets = [[] for _ in range(self._pp_size)]

        if device_stage_alignments:
            self._devices = device_stage_alignments
        else:
            self._devices = [[] for _ in range(self._device_size)]
            self._fix_stages(stage_type="ZBV")

        self.model_result = None

    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print("Device {}: {}".format(did, ds))

    def show_solution_detail(self):
        for key in self.model_result:
            if str(key) == "max_start_offset":
                print("MinExeTime: {}".format(self.model_result[key]))

    def _construct_stages(self, stage_type=None):
        if stage_type == "ZBV":
            self._fix_stages(stage_type="ZBV")
        elif stage_type == "I1F1B":
            self._fix_stages(stage_type="I1F1B")
        else:
            ds = self._device_size
            ss = self._pp_size
            print("Modeling Stages Alignment...")
            # Define stage variables
            self._devices = [[] for _ in range(ds)]

            for i in range(ds):
                for j in range(ss // ds):
                    stage_var = self.model.addVar(vtype=GRB.INTEGER, name=f'stage_{i}_{j}')
                    self._devices[i].append(stage_var)
                    self.model.addConstr(stage_var >= 1)
                    self.model.addConstr(stage_var <= ss)

            all_stages = []
            for i in range(ds):
                for j in range(ss // ds):
                    all_stages.append(self._devices[i][j])
            self.model.addConstrs((all_stages[i] != all_stages[j] for i in range(len(all_stages)) for j in range(i+1, len(all_stages))), name='DistinctStages')

    def _fix_stages(self, stage_type="ZBV"):
        if stage_type == "ZBV":
            for pid in range(self._pp_size):
                if (pid // self._device_size) % 2 == 0:
                    self._devices[pid % self._device_size].append(pid)
                else:
                    self._devices[self._device_size - 1 - pid % self._device_size].append(pid)

        if stage_type == "I1F1B":
            for pid in range(self._pp_size):
                self._devices[pid % self._device_size].append(pid)

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            # F stages sequential constraint
            for i in range(1, self._pp_size):
                self.model.addConstr(
                    self._forward_offsets[i][mb] >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1] + self._comm_length[i-1][i]
                )

            # B stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self.model.addConstr(
                    self._backward_b_offsets[i - 1][mb] >= self._backward_b_offsets[i][mb] + self._backward_b_length[i] + self._comm_length[i][i-1]
                )

            # F-B connection sequential constraint
            self.model.addConstr(
                self._backward_b_offsets[self._pp_size - 1][mb] >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1]
            )

            # W stage sequential constraint
            for i in range(self._pp_size):
                self.model.addConstr(
                    self._backward_w_offsets[i][mb] >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                )

            # Set W stage increasing order
            for i in range(self._pp_size):
                if mb > 0:
                    self.model.addConstr(
                        self._backward_w_offsets[i][mb] >= self._backward_w_offsets[i][mb - 1] + self._backward_w_length[i]
                    )

    def _pipeline_activation_accumulation_constraint(self):
        for pp in range(self._pp_size):
            for mb in range(self._num_microbatches):
                _backward_var = self._backward_b_offsets[pp][mb]
                _activation_count = self.model.addVar(vtype=GRB.INTEGER, name=f'activation_count_{pp}_{mb}')
                self.model.addConstr(_activation_count == 1)

                for other_mb in range(self._num_microbatches):
                    if other_mb == mb:
                        continue
                    self.model.addConstrs((
                        _activation_count <= self._max_activation_counts[pp]
                    ))

    def _serial_computation_within_device_constraint(self):
        print("Stage alignment: {}".format(self._devices))
        for did in range(self._device_size):
            stages_within_device = self._devices[did]
            _pp_vars = []
            for pp in stages_within_device:
                _pp_vars += self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]

            for i in range(len(_pp_vars)):
                group_size = self._num_microbatches * 3
                i_pp = stages_within_device[i // group_size]
                _i_length = (
                    self._forward_length[i_pp] if (i % group_size) // self._num_microbatches == 0 
                    else (self._backward_b_length[i_pp] if (i % group_size) // self._num_microbatches == 1 
                          else self._backward_w_length[i_pp])
                )
                for j in range(i + 1, len(_pp_vars)):
                    j_pp = stages_within_device[j // group_size]
                    _j_length = (
                        self._forward_length[j_pp] if (j % group_size) // self._num_microbatches == 0
                        else (self._backward_b_length[j_pp] if (j % group_size) // self._num_microbatches == 1 
                              else self._backward_w_length[j_pp])
                    )
                    z = self.model.addVar(vtype=GRB.BINARY, name="z_{}_{}".format(i,j))
                    self.model.addGenConstrIndicator(z, True,  _pp_vars[j] >= _pp_vars[i] + _i_length, name="or_constraint1_{}_{}".format(i,j))
                    self.model.addGenConstrIndicator(z, False, _pp_vars[j] + _j_length <= _pp_vars[i], name="or_constraint2_{}_{}".format(i,j))
                    
                    # self.model.addConstr(
                    #     (_pp_vars[j] >= _pp_vars[i] + _i_length) | (_pp_vars[j] + _j_length <= _pp_vars[i])
                    # )

    def _build_layer_constraint(self):
        for i in range(self._pp_size):
            layer_var = self.model.addVar(vtype=GRB.INTEGER, name=f"layer_{i}")
            self._layers.append(layer_var)
            self.model.addConstr(layer_var >= 1)
            self.model.addConstr(layer_var <= self._model_size)

        self.model.addConstr(quicksum(self._layers) == self._model_size)

    def _build_recomputing_rate_constraint(self):
        for i in range(self._pp_size):
            rate_var = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"recomputing_rate_{i}")
            self._recomputing_rate.append(rate_var)
            discrete_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self.model.addConstr(rate_var == 0)

    def _update_fbw_length(self):
        self._forward_length = [self._layers[i] * self._basic_forward_length[i] for i in range(self._pp_size)]
        self._backward_b_length = [self._layers[i] * (self._basic_backward_b_length[i] + self._basic_forward_length[i] * self._recomputing_rate[i]) for i in range(self._pp_size)]
        self._backward_w_length = [self._layers[i] * self._basic_backward_w_length[i] for i in range(self._pp_size)]

    def _reset_comm_length(self, dsa):
        new_comm_length = copy.deepcopy(self._comm_length)
        for d in dsa:
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length

    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            self._layers.append(self.model.addVar(vtype=GRB.INTEGER, name=f"l_{i}"))
            self._recomputing_rate.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"r_{i}"))
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mb}_{i}"))
                self._backward_b_offsets[i].append(self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mb}_{i}"))
                self._backward_w_offsets[i].append(self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mb}_{i}"))

                self.model.addConstr(self._forward_offsets[i][-1] >= 0)

        self._build_layer_constraint()
        self._build_recomputing_rate_constraint()
        self._update_fbw_length()

        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for dsa in self._devices:
            pp = dsa[0]
            self.model.addConstr(max_var >= self._backward_w_offsets[pp][-1])
        self.model.setObjective(max_var, GRB.MINIMIZE)

    def run(self, draw=False) -> None:
        """Run simulation"""
        self._build_constraints()
        self._build_optimize_objectives()

        start_time = time.time()
        print("Gurobi Solver Solving...")
        self.model.optimize()
        end_time = time.time()

        if self.model.status == GRB.OPTIMAL:
            print(f"Result: OPTIMAL, Cost: {end_time - start_time:.2f}")
            results = {v.varName: v.x for v in self.model.getVars()}
            self.model_result = results
            if draw:
                self._draw(results)
            return results
        else:
            print(f"Result: UNSAT, Cost: {end_time - start_time:.2f}")
            return {"max_start_offset": 999999999999}

    def _draw(self, results: dict) -> None:
        # Implement your drawing logic here
        pass

class SPSimulator:
    """Simulator"""

    def __init__(self, config: dict, device_stage_alignments=None, new_comm_length=None) -> None:
        self._file_path = config["file_path"]
        print_to_file(self._file_path, "forward_execution_time:{}\nbackward_execution_i_time:{}\nbackward_execution_g_time:{}.\n".format(
                ft, bt, wt
            )
        )
        print_to_file(self._file_path, "Device size:{}\nPipeline size:{}\nModel size:{}\nNumber of microbatches size:{}.\n".format(
                device_size, pp_size, model_size, nmb
            )
        )
        self._pp_size                   = config["pp_size"]
        self._device_size               = config["device_size"]
        self._model_size                = config["model_size"]
        self.virtual_stage              = config["virtual_stage"]
        self._num_real_microbatches     = config["num_microbatches"]
        self._num_microbatches          = config["num_microbatches"] * self.virtual_stage[0]
        self._max_activation_counts     = config["max_activation_counts"]
        self._basic_forward_length      = config["forward_execution_time"]
        self._basic_backward_b_length   = config["backward_execution_i_time"]
        self._basic_backward_w_length   = config["backward_execution_g_time"]
        self._comm_length               = config["communication_time"] if not new_comm_length else new_comm_length

        self._sequential_order_constraint_strategy = config[
            "sequential_order_constraint_strategy"
        ]

        self._forward_length            = self._basic_forward_length
        self._backward_b_length         = self._basic_backward_b_length
        self._backward_w_length         = self._basic_backward_w_length
        
        assert isinstance(
            self._forward_length, (list, tuple)
        ), "forward_execution_time must be list or tuple"
        assert isinstance(
            self._backward_b_length, (list, tuple)
        ), "backward_execution_time must be list or tuple"
        assert isinstance(
            self._backward_w_length, (list, tuple)
        ), "backward_execution_time must be list or tuple"

        assert self._sequential_order_constraint_strategy in (
            "strict",
            "double_interleaving",
            "full_interleaving",
        ), "sequential order constraint strategy is not supported"

        self._solver                = z3.Optimize()
        self._recomputing_rate      = []
        self._layers                = []
        self._forward_offsets       = [[] for _ in range(self._pp_size)]
        self._backward_b_offsets    = [[] for _ in range(self._pp_size)]
        self._backward_w_offsets    = [[] for _ in range(self._pp_size)]

        if device_stage_alignments:
            self._devices = device_stage_alignments
        else:      
            self._devices = [[] for _ in range(self._device_size)]
            self._fix_stages(stage_type="ZBV")

        self.model_result = None
        # self._construct_stages()

    def show_device_stage_mapping(self):
        for did,ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        for key in self.model_result:
            print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _construct_stages(self, stage_type=None):
        if stage_type == "ZBV":
            self._fix_stages(stage_type="ZBV")
        elif stage_type == "I1F1B":
            self._fix_stages(stage_type="I1F1B")
        else:
            ds = self._device_size
            ss = self._pp_size
            print_to_file(self._file_path, "Modeling Stages Alignment...\n")
            # 定义阶段变量
            # stages[i] 是设备 i 分配的阶段集合
            self._devices = [z3.Array(f'stage_{i}', z3.IntSort(), z3.IntSort()) for i in range(ds)]

            # # 定义每个设备分配阶段的数量
            # counts = [z3.Int(f'count_{i}') for i in range(ds)]

            # # 添加约束条件
            # # 每个设备的阶段数量必须大于 0
            # for count in counts:
            #     self._solver.add(count >= 1, count <= ss)

            # # 计算总的阶段数量
            # total_count = z3.Sum(counts)
            # self._solver.add(total_count == ss)

            # 每个设备的阶段编号在 1 到 s 之间
            all_stages = []
            for i in range(ds):
                for j in range(ss // ds):
                    self._solver.add(z3.And(self._devices[i][j] >= 1, self._devices[i][j] <= ss))
                    all_stages.append(self._devices[i][j])
            self._solver.add(z3.Distinct(all_stages))
    
    def _fix_stages(self, stage_type="ZBV"):
        if stage_type == "ZBV":
            for pid in range(self._pp_size):
                if (pid // self._device_size) % 2 == 0:
                    self._devices[pid % self._device_size].append(pid)
                else:
                    self._devices[self._device_size - 1 - pid % self._device_size].append(pid)
        
        if stage_type == "I1F1B":
            for pid in range(self._pp_size):
                self._devices[pid % self._device_size].append(pid)

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            # F stages sequential constraint
            # 不同stage间的约束关系
            for i in range(1, self._pp_size):
                self._solver.add(
                    self._forward_offsets[i][mb] 
                    >= self._forward_offsets[i - 1][mb] 
                    + self._forward_length[i - 1] 
                    + self._comm_length[i - 1][i]
                )
            
            # B stages sequential constraint
            # 不同stage间的约束关系
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_b_offsets[i - 1][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                    + self._comm_length[i][i - 1]
                )
                
            # F-B connection sequential constraint
            # # # #相同stage间的约束关系，每个mb的F与B不重叠
            self._solver.add(
                self._backward_b_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] 
                + self._forward_length[self._pp_size - 1]
            )

            # W stage sequential constraint
            # # # #相同stage间的约束关系，每个mb的B和W不重叠
            for i in range(self._pp_size):
                self._solver.add(
                    self._backward_w_offsets[i][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                )

            # Set increasing order within stage, leading to faster solving
            # # # #相同stage间的约束关系，每个mb的F之间、B之间、W之间不重叠
            if mb > 0:
                for i in range(self._pp_size):
                    self._solver.add(
                        self._forward_offsets[i][mb]
                        >= self._forward_offsets[i][mb - 1]
                        + self._forward_length[i]
                    )
                    self._solver.add(
                        self._backward_b_offsets[i][mb]
                        >= self._backward_b_offsets[i][mb - 1]
                        + self._backward_b_length[i]
                    )
                    self._solver.add(
                        self._backward_w_offsets[i][mb]
                        >= self._backward_w_offsets[i][mb - 1]
                        + self._backward_w_length[i]
                    )
            # Fix the first mb to 0s, leading to faster solving
            else:
                self._solver.add(
                        self._forward_offsets[0][0] == 0
                )
                
        # Set W stage increasing order within the same device
        # for dsa in self._devices:
        #     for idx in range(0, len(dsa)):
        #         if idx < len(dsa) - 1:
        #             self._solver.add(
        #                 self._backward_w_offsets[dsa[idx]][-1]
        #                 >= self._backward_w_offsets[dsa[idx + 1]][0]
        #                 + self._backward_w_length[dsa[idx + 1]]
        #             )
                    
    def _pipeline_activation_accumulation_constraint(self):
        for pp in range(self._pp_size):
            # calculate the maximum activation value for this pp
            for mb in range(self._num_microbatches):
                _backward_var = self._backward_b_offsets[pp][mb]
                _actvaition_count = 1

                for other_mb in range(self._num_microbatches):
                    if other_mb == mb:
                        continue
                    _actvaition_count += z3.If(
                        z3.And(
                            self._backward_b_offsets[pp][other_mb] > _backward_var,
                            self._forward_offsets[pp][other_mb] < _backward_var,
                        ),
                        1,
                        0,
                    )

                self._solver.add(_actvaition_count <= self._max_activation_counts[pp])

    def _serial_computation_within_device_constraint(self):
        print_to_file(self._file_path, "Stage alignment:{}.\n".format(self._devices))
        total_constraints = 0
        same_mb_redundant_constraints = 0
        for did in range(self._device_size):
            # 加入对w的判断，同时修改_length的判断
            stages_within_device = self._devices[did]
            _pp_vars = []
            for pp in stages_within_device:
                _pp_vars += self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]
            
            group_size = self._num_microbatches * 3
            for i, _ in enumerate(_pp_vars):
                i_pp = stages_within_device[i // group_size]
                _i_length = (
                    self._forward_length[i_pp]
                    if (i % group_size) // self._num_microbatches == 0 
                    else(
                        self._backward_b_length[i_pp] 
                        if (i % group_size) // self._num_microbatches == 1 
                        else self._backward_w_length[i_pp]
                    )
                )
                for j in range(i + 1, len(_pp_vars)):
                    total_constraints += 1
                    # 根据_real_pipeline_modeling_constraint_strict中对重叠关系的分析，同一mb之间不存在重叠。
                    # 只有不同mb之间会存在重叠情况，剔除多余的约束条件后，测试d=5 mb=5时，时间变化为35s→25s，但发现d=4 mb=4时，时间基本不变化
                    if j // (self._num_microbatches * 3) == i // (self._num_microbatches * 3):
                        if j % self._num_microbatches == i % self._num_microbatches:
                            same_mb_redundant_constraints += 1
                            continue
                    j_pp = stages_within_device[j // group_size]
                    _j_length = (
                        self._forward_length[j_pp]
                        if (j % group_size) // self._num_microbatches == 0
                        else(
                            self._backward_b_length[j_pp] 
                            if (j % group_size) // self._num_microbatches == 1 
                            else self._backward_w_length[j_pp]
                        )
                    )
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )
                    # 替换成Not and后不等价
                    # self._solver.add(
                    #     z3.Not(z3.And(
                    #         _pp_vars[i] + _i_length >= _pp_vars[j],
                    #         _pp_vars[j] + _j_length >= _pp_vars[i],
                    #     ))
                    # )
                    # 类似图形学中判断两直线是否重叠，避免使用OR语句
                    # 改写后却变慢：
                    # a1 = _pp_vars[i]
                    # b1 = _pp_vars[i] + _i_length
                    # a2 = _pp_vars[j]
                    # b2 = _pp_vars[j] + _j_length
                    # self._solver.add(
                    #     (b2 - a1)*(a2 - b1) >= 0
                    # )
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _build_layer_constraint(self):
        for i in range(self._pp_size):
            self._solver.add(
                self._layers[i] >= 1,
                self._layers[i] <= self._model_size
            )
        self._solver.add(sum(self._layers) == self._model_size)
        
    def _build_recomputing_rate_constraint(self):
        # discrete_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # faster!
        # discrete_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0] # slower!
        for i in range(self._pp_size):
            self._solver.add(       # faster, in a way, z3.Or may slower solving?!
                self._recomputing_rate[i] >= 0,
                self._recomputing_rate[i] <= 1
            )
            # self._solver.add(z3.Or([self._recomputing_rate[i] == v for v in discrete_values]))
    
    def _update_fbw_length(self):
        self._forward_length = [self._layers[i] * self._basic_forward_length[i] for i in range(self._pp_size)]
        self._backward_b_length = [self._layers[i] * (self._basic_backward_b_length[i] + self._basic_forward_length[i] * self._recomputing_rate[i]) for i in range(self._pp_size)]
        self._backward_w_length = [self._layers[i] * self._basic_backward_w_length[i] for i in range(self._pp_size)]
    
    def _reset_comm_length(self, dsa):
        new_comm_length = copy.deepcopy(self._comm_length)
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length
    
    def _build_constraints(self) -> None:
        
        for i in range(self._pp_size):
            self._layers.append(z3.Int(f"l_{i}"))
            self._recomputing_rate.append(z3.Real(f"r_{i}"))
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(z3.Int(f"f_{mb}_{i}"))
                self._backward_b_offsets[i].append(z3.Int(f"b_{mb}_{i}"))
                self._backward_w_offsets[i].append(z3.Int(f"w_{mb}_{i}"))

                self._solver.add(self._forward_offsets[i][-1] >= 0)

        self._build_layer_constraint()
        self._build_recomputing_rate_constraint()
        self._update_fbw_length()

        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_device_constraint()

        # constraint 3: the accumulation count of activations does not exceed max_activation_counts
        # self._pipeline_activation_accumulation_constraint()

    def _build_optimize_objectives(self) -> None:
        # 1. minimize the execution time of each microbatch
        max_var = z3.Int("max_start_offset")
        # Add optimization objectives according to devices
        # Reduce optimize objectives to O(d), but not return the optimal result.
        # for dsa in self._devices:
        #     pp = dsa[0]
        #     self._solver.add(max_var >= self._backward_w_offsets[pp][-1])

        # Add optimization objectives according to stages
        for pp in range(self._pp_size):
            # Complexity of optimization objectives is O(s)
            # Need to ensure the W of each microbatch is in increasing order
            # self._solver.add(max_var >= self._backward_w_offsets[pp][-1])
            
            self._solver.add(max_var >= self._backward_w_offsets[pp][-1] + self._backward_w_length[pp])

            # Complexity of optimization objectives is O(s * microbatches)
            # This behavior will dramatically increase the searching complexity
            # for var in self._backward_w_offsets[pp]:
            #     self._solver.add(max_var >= var)
        
        # Reduce optimize objectives to O(1)
        # self._solver.add(max_var >= z3.Sum([self._backward_w_offsets[i][-1] for i in range(self._pp_size)]))
        
        # Set search upper bound
        # max_f_len = max(self._basic_forward_length)
        # max_b_len = max(self._basic_backward_b_length)
        # max_w_len = max(self._basic_backward_w_length)
        # sum_f_len = sum(self._basic_forward_length)
        # sum_b_len = sum(self._basic_backward_b_length)
        # sum_w_len = sum(self._basic_backward_w_length)
        # gpipe_zb_time = (sum_f_len + max(sum_b_len, sum_w_len) + (self._device_size - 1) * (max_f_len + max_b_len + max_w_len) + max_w_len + self._device_size * comm)
        # self._solver.add(max_var <= gpipe_zb_time)

        # Optimize
        self._solver.minimize(max_var)

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "device_size": self._device_size,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 2,
            "num_real_microbatches": self._num_real_microbatches,
            "forward_length": self._forward_length,
            "backward_length": self._backward_b_length,
            "backward_length2": self._backward_w_length,
            "file_path": self._file_path,
        }

        SchedulingPainter(painter_conf).draw(results)

    def run(self, draw=False) -> None:
        """run simulation"""
        # 1. builds the solver constraints.
        self._build_constraints()

        # 2. builds the solver optimize objectives.
        self._build_optimize_objectives()
        
        # 3. runs the solver.
        start_time = time.time()
        print_to_file(self._file_path, "Z3 Solver Solving...\n")
        check = self._solver.check()
        end_time = time.time()
        if  check == z3.sat:
            print_to_file(self._file_path, f"Result: SAT, Cost: {end_time - start_time:.2f}.\n")
            # tranforms the result to a dictionary.
            model = self._solver.model()
            self.model_result = model
            results = {str(key) : model[key] for key in model}
            print_to_file(self._file_path, "MinExeTime:{}.\n".format(results["max_start_offset"].as_long()))
            for i in range(self._pp_size):
                number_of_layers = model[self._layers[i]].as_long()
                recompute_rate = float(model[self._recomputing_rate[i]].as_fraction())
                self._forward_length[i] = self._basic_forward_length[i] * number_of_layers
                self._backward_b_length[i] = (self._basic_backward_b_length[i] + self._basic_forward_length[i] * recompute_rate) * number_of_layers 
                self._backward_w_length[i] = self._basic_backward_w_length[i] * number_of_layers
            if draw:
                # 4. draws the result.
                results = {str(key) : model[key].as_long() for key in model if str(key)[0:2] in ["f_","b_","w_"]}
                self._draw(resort_microbatch_index(self._num_microbatches ,results))
            return results
        else:
            print_to_file(self._file_path, f"Result: UNSAT, Cost: {end_time - start_time:.2f}.\n")
            return {"max_start_offset": 999999999999}

class DSASimulator:
    def __init__(self, config) -> None:

        self._pp_size                   = config["pp_size"]
        self._device_size               = config["device_size"]
        self.config                     = config
        self._basic_comm_length         = config["communication_time"]
        self._device_stage_alignments   = []
        self._file_path                 = config["file_path"]
    def _unique_result(self, device_stage_alignment):
        for existing_result in self._device_stage_alignments:
            acc = 0
            for stage_alignment in device_stage_alignment:
                if stage_alignment not in existing_result:
                    break
                else:
                    acc += 1
            if acc == self._device_size:
                return False
        return True

    def _prune_result(self, device_stage_alignment):
        for dsa in device_stage_alignment:
            if len(dsa) != self._pp_size // self._device_size:
                return False
            if len(dsa) < self._pp_size // self._device_size - 1:
                return False
            if len(dsa) > self._pp_size // self._device_size + 1:
                return False
            if len(dsa) == 0:
                return False
        return True

    def _reset_comm_length(self, dsa):
        new_comm_length = self._basic_comm_length
        for d in dsa:
            for i in range(len(d)):
                for j in range(i + 1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length

    def _traverse_every_stage_alignment(self, sid, device_stage_alignment):
        if sid == self._pp_size:
            if self._prune_result(device_stage_alignment) and self._unique_result(device_stage_alignment):
                self._device_stage_alignments.append(copy.deepcopy(device_stage_alignment))
        else:
            for did in range(self._device_size):
                device_stage_alignment[did].append(sid)
                self._traverse_every_stage_alignment(sid + 1, device_stage_alignment)
                device_stage_alignment[did].pop()

    def traverse_run(self) -> None:

        print_to_file(self._file_path, "Traversing every stage alignment...\n")
        device_stage_alignments = [[] for _ in range(self._device_size)]
        self._traverse_every_stage_alignment(0, device_stage_alignment=device_stage_alignments)
        print_to_file(self._file_path, "Traversing over. {} situations found.\n".format(len(self._device_stage_alignments)))

        best_result = None
        minimal_time = 999999999999
        simulators = []
        for dsa in self._device_stage_alignments:
            temp_simulator = SPSimulator(self.config, device_stage_alignments=dsa)
            simulators.append(temp_simulator)
            result = temp_simulator.run()
            result_time = result["max_start_offset"].as_long()
            if result_time < minimal_time:
                minimal_time = result_time
                best_result = temp_simulator

        
        result = {str(key) : best_result.model_result[key].as_long() for key in best_result.model_result if str(key)[0:2] in ["f_","b_","w_"]}
        end_time = time.time()
        best_result.show_device_stage_mapping()
        best_result.show_solution_detail()
        best_result._draw(resort_microbatch_index(best_result._num_microbatches ,result))

        return end_time