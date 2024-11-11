import time
from gurobipy import Model, GRB, quicksum
from .config import *
from .painter import SchedulingPainter
from .utils import resort_microbatch_index, print_to_file
class GSimulator:
    """Simulator"""

    def __init__(self, config: dict, device_stage_alignments=None, new_comm_length=None) -> None:
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
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

        self._forward_length            = self._basic_forward_length
        self._backward_b_length         = self._basic_backward_b_length
        self._backward_w_length         = self._basic_backward_w_length
        
        # 检查输入参数
        assert isinstance(self._basic_forward_length, (list, tuple))
        assert isinstance(self._basic_backward_b_length, (list, tuple))
        assert isinstance(self._basic_backward_w_length, (list, tuple))
        assert self._sequential_order_constraint_strategy in ("strict", "double_interleaving", "full_interleaving")

        # 创建 Gurobi 模型
        self.model = Model("SPSimulator")

        # 变量初始化
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
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        for key in self.model_result:
            print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _fix_stages(self, stage_type="ZBV"):
        if stage_type == "ZBV":
            for pid in range(self._pp_size):
                if (pid // self._device_size) % 2 == 0:
                    self._devices[pid % self._device_size].append(pid)
                else:
                    self._devices[self._device_size - 1 - pid % self._device_size].append(pid)
        elif stage_type == "I1F1B":
            for pid in range(self._pp_size):
                self._devices[pid % self._device_size].append(pid)

    def _reset_comm_length(self, dsa):
        new_comm_length = [[comm if i != j else 0 for j in range(self._pp_size)] for i in range(self._pp_size)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length
    def _build_constraints(self) -> None:
        for i in range(self._pp_size):
            layer_var = self.model.addVar(vtype=GRB.INTEGER, name=f"l_{i}", lb=1, ub=self._model_size)
            self._layers.append(layer_var)
            recompute_var = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"r_{i}", lb=0, ub=1)
            self._recomputing_rate.append(recompute_var)

            for mb in range(self._num_microbatches):
                f_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mb}_{i}", lb=0)
                b_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mb}_{i}", lb=0)
                w_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mb}_{i}", lb=0)
                self._forward_offsets[i].append(f_offset)
                self._backward_b_offsets[i].append(b_offset)
                self._backward_w_offsets[i].append(w_offset)

        # 添加约束
        self.model.addConstr(quicksum(self._layers) == self._model_size)
        self._comm_length = self._reset_comm_length(self._devices)

        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()
        # self._pipeline_activation_accumulation_constraint()

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            for i in range(1, self._pp_size):
                self.model.addConstr(self._forward_offsets[i][mb] >= self._forward_offsets[i - 1][mb] +
                                     self._basic_forward_length[i - 1] + self._comm_length[i - 1][i])

            for i in range(self._pp_size - 1, 0, -1):
                self.model.addConstr(self._backward_b_offsets[i - 1][mb] >= self._backward_b_offsets[i][mb] +
                                     self._basic_backward_b_length[i] + self._comm_length[i][i - 1])

            self.model.addConstr(self._backward_b_offsets[self._pp_size - 1][mb] >= self._forward_offsets[self._pp_size - 1][mb] +
                                 self._basic_forward_length[self._pp_size - 1])

            for i in range(self._pp_size):
                self.model.addConstr(self._backward_w_offsets[i][mb] >= self._backward_b_offsets[i][mb] +
                                     self._basic_backward_b_length[i])

            if mb > 0:
                for i in range(self._pp_size):
                    self.model.addConstr(self._forward_offsets[i][mb] >= self._forward_offsets[i][mb - 1] +
                                         self._basic_forward_length[i])
                    self.model.addConstr(self._backward_b_offsets[i][mb] >= self._backward_b_offsets[i][mb - 1] +
                                         self._basic_backward_b_length[i])
                    self.model.addConstr(self._backward_w_offsets[i][mb] >= self._backward_w_offsets[i][mb - 1] +
                                         self._basic_backward_w_length[i])
            else:
                self.model.addConstr(self._forward_offsets[0][0] == 0)

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
                    # z3-solver way
                    # self._solver.add(
                    #     z3.Or(
                    #         _pp_vars[j] >= _pp_vars[i] + _i_length,
                    #         _pp_vars[j] + _j_length <= _pp_vars[i],
                    #     )
                    # )
                    # gurobi way, too slow
                    # self.model.addConstr(
                    #     (_pp_vars[j] + _j_length - _pp_vars[i]) * (_pp_vars[j] - _pp_vars[i] - _i_length) >= 0
                    # )
                    y = self.model.addVar(vtype=GRB.BINARY, name=f"Do{did}_{i}_{j}")
                    M = 1e6
                    self.model.addConstr(_pp_vars[j] >= _pp_vars[i] + _i_length - (1 - y) * M) 
                    self.model.addConstr(_pp_vars[j] + _j_length <= _pp_vars[i] + y * M)
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _pipeline_activation_accumulation_constraint(self):
        for pp in range(self._pp_size):
            for mb in range(self._num_microbatches):
                _backward_var = self._backward_b_offsets[pp][mb]
                activation_count = 1

                for other_mb in range(self._num_microbatches):
                    if other_mb == mb:
                        continue
                    activation_count += self.model.addVar(vtype=GRB.CONTINUOUS, name=f"activation_{mb}_{other_mb}_{pp}", lb=0)

                self.model.addConstr(activation_count <= self._max_activation_counts[pp])

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for pp in range(self._pp_size):
            self.model.addConstr(max_var >= self._backward_w_offsets[pp][-1] + self._basic_backward_w_length[pp])
        self.model.setObjective(max_var, GRB.MINIMIZE)

    def run(self, draw=False) -> None:
        """run simulation"""
        self._build_constraints()
        self._build_optimize_objectives()

        # self.model.setParam('TimeLimit', self._time_limit)
        self.model.setParam('MIPGap', 0.01)
        self.model.optimize()

        start_time = time.time()
        print_to_file(self._file_path, "Gurobi Solver Solving...\n")
        self.model.optimize()
        end_time = time.time()
        if self.model.status == GRB.OPTIMAL:
            print_to_file(self._file_path, f"Optimal Result Cost: {end_time - start_time:.2f}.\n")
            # tranforms the result to a dictionary.
        elif self.model.status != GRB.INFEASIBLE:
            print_to_file(self._file_path, f"Result Found Cost: {end_time - start_time:.2f}.\n")
        else:
            print_to_file(self._file_path, "No Solution Found.\n")
            return {"max_start_offset": 999999999999}
        
        results = {var.varName: var.x for var in self.model.getVars()}
        self.model_result = results
        print_to_file(self._file_path, "MinExeTime:{}.\n".format(results["max_start_offset"]))
        for i in range(self._pp_size):
            number_of_layers = self.model_result[self._layers[i].varName]
            recompute_rate = float(self.model_result[self._recomputing_rate[i].varName])
            self._forward_length[i] = self._basic_forward_length[i] * number_of_layers
            self._backward_b_length[i] = (self._basic_backward_b_length[i] + self._basic_forward_length[i] * recompute_rate) * number_of_layers 
            self._backward_w_length[i] = self._basic_backward_w_length[i] * number_of_layers
        if draw:
            # 4. draws the result.
            results = {str(key) : self.model_result[key] for key in self.model_result if str(key)[0:2] in ["f_","b_","w_"]}
            self._draw(resort_microbatch_index(self._num_microbatches ,results))
        return results

    def _draw(self, results: dict) -> None:
        # 绘制结果的逻辑
        painter_conf = {
            "device_size": self._device_size,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 1,
            "num_real_microbatches": self._num_real_microbatches,
            "forward_length": self._forward_length,
            "backward_length": self._backward_b_length,
            "backward_length2": self._backward_w_length,
            "comm_length": self._comm_length,
            "file_path": self._file_path,
        }

        SchedulingPainter(painter_conf).draw(results)