import time
from gurobipy import Model, GRB, quicksum
from .painter import SchedulingPainter
from .abstract.mutils import *
from .utils import resort_microbatch_index, print_to_file
from .abstract import Pipeline
class CompCommSimulator:

    def __init__(self, config: dict) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["pp_size"]
        self._device_size = config["device_size"]
        self._model_layer_num = config["model_size"]
        self._num_microbatches = config["num_microbatches"]
        self._max_activation_counts = config["max_activation_counts"]
        self.split_backprop = config["split_backprop"]
        self.pixel_base = config["pixel_base"]
        # obtained by profiling
        self._profiled_layer_f_length = config["forward_execution_time"]
        self._profiled_layer_b_length = config["backward_execution_i_time"]
        self._profiled_layer_w_length = config["backward_execution_g_time"]
        
        # 检查输入参数
        assert isinstance(self._profiled_layer_f_length, (list, tuple))
        assert isinstance(self._profiled_layer_b_length, (list, tuple))
        assert isinstance(self._profiled_layer_w_length, (list, tuple))

        # 创建 Gurobi 模型
        self.model = Model("CompCommSimulator")

        # 变量初始化
        self._stage_f_length  = []
        self._stage_b_length = []
        self._stage_w_length = []

        self._layer_recomp_rate = []
        self._layers = []

        self._stage_f_offsets = [[] for _ in range(self._pp_size)]
        self._stage_b_offsets = [[] for _ in range(self._pp_size)]
        self._stage_w_offsets = [[] for _ in range(self._pp_size)]

        self._comm_f_offsets = [[] for _ in range(self._pp_size)]
        self._comm_b_offsets = [[] for _ in range(self._pp_size)]
        self._comm_w_offsets = [[] for _ in range(self._pp_size)]

        self._devices = [[] for _ in range(self._device_size)]
        self._fix_stages()

        self.model_result = None

    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        prefixes = ('f_', 'b_', 'w_')
        for key in self.model_result:
            if str(key).startswith(prefixes):
                print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _fix_stages(self):
        for pid in range(self._pp_size):
            if (pid // self._device_size) % 2 == 0:
                self._devices[pid % self._device_size].append(pid)
            else:
                self._devices[self._device_size - 1 - pid % self._device_size].append(pid)

    def _build_constraints(self) -> None:
        for i in range(self._pp_size):

            layer_var = self.model.addVar(vtype=GRB.INTEGER, name=f"l_{i}", lb=1, ub=self._model_layer_num)
            self._layers.append(layer_var)

            recompute_var = self.model.addVar(vtype=GRB.BINARY, name=f"theta_{i}")
            self._layer_recomp_rate.append(recompute_var)

            for mb in range(self._num_microbatches):
                
                f_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"f_{mb}_{i}", lb=0)
                self._stage_f_offsets[i].append(f_offset)

                b_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"b_{mb}_{i}", lb=0)
                self._stage_b_offsets[i].append(b_offset)

                if self.split_backprop:
                    w_offset = self.model.addVar(vtype=GRB.INTEGER, name=f"w_{mb}_{i}", lb=0)
                    self._stage_w_offsets[i].append(w_offset)

            self._stage_f_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{i}_f"))
            self.model.addConstr(self._stage_f_length[i] == self._profiled_layer_f_length[i])
            self._stage_b_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{i}_b"))
            self.model.addConstr(self._stage_b_length[i] == (self._profiled_layer_b_length[i] + self._layer_recomp_rate[i] * self._stage_f_length[i]))
            self._stage_w_length.append(self.model.addVar(vtype=GRB.INTEGER, name=f"s{i}_w"))
            self.model.addConstr(self._stage_w_length[i] == self._profiled_layer_w_length[i])

        # 添加约束
        # self.model.addConstr(quicksum(self._layers) == self._model_layer_num)

        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            for i in range(1, self._pp_size):
                self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i - 1][mb] +
                                     self._stage_f_length[i - 1])

            for i in range(self._pp_size - 1, 0, -1):
                self.model.addConstr(self._stage_b_offsets[i - 1][mb] >= self._stage_b_offsets[i][mb] +
                                     self._stage_b_length[i])

            self.model.addConstr(self._stage_b_offsets[self._pp_size - 1][mb] >= self._stage_f_offsets[self._pp_size - 1][mb] +
                                 self._stage_f_length[self._pp_size - 1])

            if self.split_backprop:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_w_offsets[i][mb] >= self._stage_b_offsets[i][mb] +
                                        self._stage_b_length[i])

            if mb > 0:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i][mb - 1] +
                                         self._stage_f_length[i])
                    self.model.addConstr(self._stage_b_offsets[i][mb] >= self._stage_b_offsets[i][mb - 1] +
                                         self._stage_b_length[i])
                    if self.split_backprop:
                        self.model.addConstr(self._stage_w_offsets[i][mb] >= self._stage_w_offsets[i][mb - 1] +
                                            self._stage_w_length[i])
        self.model.addConstr(self._stage_f_offsets[0][0] == 0)

    def _serial_computation_within_device_constraint(self):
        print_to_file(self._file_path, "Stage alignment:{}.\n".format(self._devices))
        total_constraints = 0
        same_mb_redundant_constraints = 0
        for did in range(self._device_size):
            # 加入对w的判断，同时修改_length的判断
            stages_within_device = self._devices[did]
            _pp_vars = []
            for pp in stages_within_device:
                _pp_vars += self._stage_f_offsets[pp] + self._stage_b_offsets[pp]
                if self.split_backprop:
                    _pp_vars += self._stage_w_offsets[pp]
            type_of_workload = 3 if self.split_backprop else 2
            group_size = self._num_microbatches * type_of_workload
            for i, _ in enumerate(_pp_vars):
                i_pp = stages_within_device[i // group_size]
                _i_length = (
                    self._stage_f_length[i_pp]
                    if (i % group_size) // self._num_microbatches == 0 
                    else(
                        self._stage_b_length[i_pp] 
                        if (i % group_size) // self._num_microbatches == 1 
                        else self._stage_w_length[i_pp]
                    )
                )
                for j in range(i + 1, len(_pp_vars)):
                    total_constraints += 1
                    if j // (self._num_microbatches * type_of_workload) == i // (self._num_microbatches * type_of_workload):
                        if j % self._num_microbatches == i % self._num_microbatches:
                            same_mb_redundant_constraints += 1
                            continue
                    j_pp = stages_within_device[j // group_size]
                    _j_length = (
                        self._stage_f_length[j_pp]
                        if (j % group_size) // self._num_microbatches == 0
                        else(
                            self._stage_b_length[j_pp] 
                            if (j % group_size) // self._num_microbatches == 1 
                            else self._stage_w_length[j_pp]
                        )
                    )
                    y = self.model.addVar(vtype=GRB.BINARY, name=f"Do{did}_{i}_{j}")
                    M = 1e5
                    self.model.addConstr(_pp_vars[j] >= _pp_vars[i] + _i_length - (1 - y) * M) 
                    self.model.addConstr(_pp_vars[j] + _j_length <= _pp_vars[i] + y * M)
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.INTEGER, name="max_start_offset")
        for pp in range(self._pp_size):
            if self.split_backprop:
                self.model.addConstr(max_var >= self._stage_w_offsets[pp][-1] + self._stage_w_length[pp])
            else:
                self.model.addConstr(max_var >= self._stage_b_offsets[pp][-1] + self._stage_b_length[pp])

        self.model.setObjective(max_var, GRB.MINIMIZE)

    def run(self, draw=False) -> None:
        """run simulation"""
        self._build_constraints()        
        self._build_optimize_objectives()

        self.model.setParam('TimeLimit', self._time_limit)
        # self.model.setParam('MIPGap', 0.00)

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
            self._stage_f_length[i] = self.model_result[self._stage_f_length[i].varName]
            self._stage_b_length[i] = self.model_result[self._stage_b_length[i].varName] 
            self._stage_w_length[i] = self.model_result[self._stage_w_length[i].varName]
        
        if draw:
            # 4. draws the result.
            results = {str(key) : self.model_result[key] for key in self.model_result if str(key)[0:2] in ["f_","b_","w_"]}
            self._draw(resort_microbatch_index(self._num_microbatches ,results))

        return results
    
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
        for key in self.model_result:
            if key.startswith(("f_","b_","w_")):
                print_to_file(f"gurobi_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key},{self.model_result[key]}\n")

                workload_len = self.get_workload_len(key=key)
                print_to_file(f"gurobi_mb{MICRO_BATCH_NUM}_pp{DEVICE_NUM}.txt", f"{key}_e,{self.model_result[key] + workload_len}\n")

    def _draw(self, results: dict) -> None:
        # 绘制结果的逻辑
        # self.write_fbw_to_file()
        painter_conf = {
            "device_size": self._device_size,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": self.pixel_base,
            "num_microbatches": self._num_microbatches,
            "forward_length": self._stage_f_length,
            "backward_length": self._stage_b_length,
            "backward_length2": self._stage_w_length,
            "comm_length": [[0 if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
            "file_path": self._file_path,
            "max_time": self.model_result['max_start_offset'],
        }

        SchedulingPainter(painter_conf).draw(results)


if __name__ == "__main__":
    config = {
        "run_mode": RUN_MODE,
        "device_size": int(DEVICE_NUM),
        "time_limit": int(SOLVING_TIME_LIMIT),
        "stage_order_search": STAGE_SEARCH_METHOD,
        "pp_size": int(STAGE_NUM),
        "model_size": int(LAYER_NUM),
        "num_microbatches": int(MICRO_BATCH_NUM),
        "forward_execution_time": [F_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_i_time": [B_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "backward_execution_g_time": [W_TIME / (int(STAGE_NUM) // int(DEVICE_NUM)) for _ in range(STAGE_NUM)],
        "device_mem": [GPU_MAX_MEM for _ in range(DEVICE_NUM)],
        "mix_training": MIX_TRAINING,
        "model_para_num": PARAMETER_NUM,
        "communication_time": [[COMM_TIME if i != j else 0 for j in range(STAGE_NUM)] for i in range(STAGE_NUM)],
        "sequential_order_constraint_strategy": "strict",
        "max_activation_counts": [MAX_ACTIVATION_COUNTS for _ in range(STAGE_NUM)],
        # "file_path": filename,
        "file_path": None,
        "base_solution" : BASE_SOLUTION,
        "schedule_method": SCHEDULE_METHOD,
        "emb_head_ce": SPLIT_EMB_HEAD_CE,
    }
    sim = CompCommSimulator(config=config)
    sim.run()