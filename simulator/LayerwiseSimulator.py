import time
from gurobipy import Model, GRB, quicksum
from .LayerwisePainter import SchedulingPainter
from .abstract.mutils import *
from .utils import resort_microbatch_index, print_to_file
from .abstract import Pipeline
class LayerwiseSimulator:

    def __init__(self, config: dict, device_layer_alignments=None, new_comm_length=None) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["pp_size"]
        self._num_device = config["device_size"]
        self._num_layer = config["model_size"]
        self._num_microbatches = config["num_microbatches"]
        self._max_activation_counts = config["max_activation_counts"]
        
        self._mix_training = config["mix_training"]
        self._model_para_num = config["model_para_num"]
        self._device_mem = config["device_mem"]
        # obtained by profiling
        self._profiled_layer_f_length = config["forward_execution_time"]
        self._profiled_layer_b_length = config["backward_execution_i_time"]
        self._profiled_layer_w_length = config["backward_execution_g_time"]
        
        # self._profiled_additional_layer_f_length = [EMBEDDING_TIME  if lid == 0 else (LAST_FFN_F_TIME if lid == self._num_layer-1 else 0) for lid in range(self._num_layer)]
        # self._profiled_additional_layer_b_length = [LAST_FFN_B_TIME if lid == self._num_layer-1 else 0 for lid in range(self._num_layer)]
        # self._profiled_additional_layer_w_length = [0 for _ in range(self._num_layer)]
        
        self._profiled_additional_layer_f_length = [0 for _ in range(self._num_layer)]
        self._profiled_additional_layer_b_length = [0 for _ in range(self._num_layer)]
        self._profiled_additional_layer_w_length = [0 for _ in range(self._num_layer)]
        
        #TODO Embedding, Head, Loss should not be fused into another layer since it leads to large bubble
        # self._profiled_additional_layer_f_length[0] += EMBEDDING_TIME
        # self._profiled_additional_layer_f_length[self._num_layer-2] += LAST_FFN_F_TIME + LOSS_F_TIME 
        # self._profiled_additional_layer_b_length[self._num_layer-2] += LAST_FFN_B_TIME + LOSS_B_TIME

        #TODO Embedding layer should not be binded with the first layer
        self._comm_length = config["communication_time"] if not new_comm_length else new_comm_length
        
        # 检查输入参数
        assert isinstance(self._profiled_layer_f_length, (list, tuple))
        assert isinstance(self._profiled_layer_b_length, (list, tuple))
        assert isinstance(self._profiled_layer_w_length, (list, tuple))

        # 创建 Gurobi 模型
        self.model = Model("SPSimulator")

        # 变量初始化
        self._layer_f_length  = []
        self._layer_b_length = []
        self._layer_w_length = []

        self._layer_recomp_rate = []
        self._layers = []
        self._device_layers_mapping = [[] for _ in range(self._num_device)]

        self._layer_f_offsets = [[] for _ in range(self._num_layer)]
        self._layer_b_offsets = [[] for _ in range(self._num_layer)]
        self._layer_w_offsets = [[] for _ in range(self._num_layer)]

        if device_layer_alignments:
            self._devices = device_layer_alignments
        else:
            self._devices = [[] for _ in range(self._num_device)]
            self._fix_stages()

        # baseline solution
        if self._base_solution:
            self.pipeline_scheduler = Pipeline.PipelineScheduler(dsa=self._devices)
            self.pipeline_scheduler.run_pipeline_parallelism()
            # self.pipeline_scheduler.draw()
        self.model_result = None

    def show_device_stage_mapping(self):
        for did, ds in enumerate(self._devices):
            print_to_file(self._file_path, "Device {}: {}.\n".format(did, ds))

    def show_solution_detail(self):
        prefixes = ('f_', 'b_', 'w_')
        for key in self.model_result:
            if str(key).startswith(prefixes):
                print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            # if not (str(key).startswith("Do") or str(key).startswith("act") or str(key).startswith("binary")):
            #     print_to_file(self._file_path, "{},{}.\n".format(str(key), self.model_result[key]))
            if str(key) == "max_start_offset":
                print_to_file(self._file_path, "MinExeTime:{}.\n".format(self.model_result[key]))

    def _fix_stages(self):
        idx_offset = 1 if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE else 0
        lid_range = range(idx_offset, self._num_layer - 2) if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE else range(self._num_layer)
        if self._schedule_method in (SchedulePriority.ZBV, SchedulePriority.GREEDY_v1, SchedulePriority.GREEDY_v2):
            for lid in lid_range:
                if ((lid-idx_offset) // self._num_device) % 2 == 0:
                    self._devices[(lid-idx_offset) % self._num_device].append(lid)
                else:
                    self._devices[self._num_device - 1 - (lid-idx_offset) % self._num_device].append(lid)
        elif self._schedule_method in (SchedulePriority.ZBH1, SchedulePriority.ONE_F_ONE_B, SchedulePriority.INTERLEAVED):
            for lid in lid_range:
                self._devices[(lid-1) % self._num_device].append(lid-idx_offset)
        else:
            assert("Stage alignment is not set.")

        if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE:
            # Reduce time waiting for embedding
            self._devices[self._num_device - 1] = [0] + self._devices[self._num_device - 1]
            self._devices[-2] = self._devices[-2] + [self._num_layer - 1]
            self._devices[-3] = self._devices[-3] + [self._num_layer - 2]
            

    def _reset_comm_length(self, dsa):
        new_comm_length = [[COMM_TIME if i != j else 0 for j in range(self._num_layer)] for i in range(self._num_layer)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length
    
    def _generate_variables(self):
        for lid in range(self._num_layer):
            self._layer_recomp_rate.append(self.model.addVar(vtype=GRB.BINARY, name=f"theta_{lid}"))
            #TODO
            # self.model.addConstr(self._layer_recomp_rate[-1] == 0)
            for mid in range(self._num_microbatches):
                
                self._layer_f_offsets[lid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"f_{mid}_{lid}", lb=0))
                self._layer_b_offsets[lid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"b_{mid}_{lid}", lb=0))
                if SPLIT_BACKPROP:
                    self._layer_w_offsets[lid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{mid}_{lid}", lb=0))
            
            self._layer_f_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{lid}_f"))
            self._layer_b_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{lid}_b"))
            self._layer_w_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{lid}_w"))
            
            self.model.addConstr(self._layer_f_length[lid] == self._profiled_layer_f_length[lid] + self._profiled_additional_layer_f_length[lid])
            self.model.addConstr(self._layer_b_length[lid] == self._profiled_layer_b_length[lid] + self._layer_recomp_rate[lid] * self._layer_f_length[lid] + self._profiled_additional_layer_b_length[lid])
            self.model.addConstr(self._layer_w_length[lid] == self._profiled_layer_w_length[lid] + self._profiled_additional_layer_w_length[lid])

    def _set_stage_layer_mapping_constraints(self):
        # self.model.addConstr(
        #     quicksum(
        #         [quicksum(self._device_layers_mapping[did]) for did in range(self._num_device)]
        #     )
        #     == self._num_layer
        # )
        # # print(self._num_layer)
        # for did in range(self._num_device):
        #     for lid in range(self._num_layer):
        #         if did == lid:
        #             self.model.addConstr(
        #                 self._device_layers_mapping[did][lid] == 1
        #             )
        #         else:
        #             self.model.addConstr(
        #                 self._device_layers_mapping[did][lid] == 0
        #             )
        pass

    def _build_constraints(self) -> None:
        self._generate_variables()
        self._set_stage_layer_mapping_constraints()
        # 添加约束
        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            for i in range(1, self._num_layer):
                self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i - 1][mb] +
                                     self._layer_f_length[i - 1] + self._comm_length[i - 1][i])

            for i in range(self._num_layer - 1, 0, -1):
                self.model.addConstr(self._layer_b_offsets[i - 1][mb] >= self._layer_b_offsets[i][mb] +
                                     self._layer_b_length[i] + self._comm_length[i][i - 1])

            self.model.addConstr(self._layer_b_offsets[self._num_layer - 1][mb] >= self._layer_f_offsets[self._num_layer - 1][mb] +
                                 self._layer_f_length[self._num_layer - 1])

            if SPLIT_BACKPROP:
                for i in range(self._num_layer):
                    self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_b_offsets[i][mb] +
                                        self._layer_b_length[i])

            if mb > 0:
                for i in range(self._num_layer):
                    self.model.addConstr(self._layer_f_offsets[i][mb] >= self._layer_f_offsets[i][mb - 1] +
                                         self._layer_f_length[i])
                    self.model.addConstr(self._layer_b_offsets[i][mb] >= self._layer_b_offsets[i][mb - 1] +
                                         self._layer_b_length[i])
                    if SPLIT_BACKPROP:
                        self.model.addConstr(self._layer_w_offsets[i][mb] >= self._layer_w_offsets[i][mb - 1] +
                                            self._layer_w_length[i])
        
        self.model.addConstr(self._layer_f_offsets[0][0] == 0)

    def _serial_computation_within_device_constraint(self):
        print_to_file(self._file_path, "Stage alignment:{}.\n".format(self._devices))
        total_constraints = 0
        same_mb_redundant_constraints = 0
        for did in range(self._num_device):
            # 加入对w的判断，同时修改_length的判断
            layers_within_device = self._devices[did]
            _pp_vars = []
            for pp in layers_within_device:
                _pp_vars += self._layer_f_offsets[pp] + self._layer_b_offsets[pp]
                if SPLIT_BACKPROP:
                    _pp_vars += self._layer_w_offsets[pp]
            type_of_workload = 3 if SPLIT_BACKPROP else 2
            group_size = self._num_microbatches * type_of_workload
            for i, _ in enumerate(_pp_vars):
                i_pp = layers_within_device[i // group_size]
                _i_length = (
                    self._layer_f_length[i_pp]
                    if (i % group_size) // self._num_microbatches == 0 
                    else(
                        self._layer_b_length[i_pp] 
                        if (i % group_size) // self._num_microbatches == 1 
                        else self._layer_w_length[i_pp]
                    )
                )
                for j in range(i + 1, len(_pp_vars)):
                    total_constraints += 1
                    if j // (self._num_microbatches * type_of_workload) == i // (self._num_microbatches * type_of_workload):
                        if j % self._num_microbatches == i % self._num_microbatches:
                            same_mb_redundant_constraints += 1
                            continue
                    j_pp = layers_within_device[j // group_size]
                    _j_length = (
                        self._layer_f_length[j_pp]
                        if (j % group_size) // self._num_microbatches == 0
                        else(
                            self._layer_b_length[j_pp] 
                            if (j % group_size) // self._num_microbatches == 1 
                            else self._layer_w_length[j_pp]
                        )
                    )
                    y = self.model.addVar(vtype=GRB.BINARY, name=f"Do{did}_{i}_{j}")
                    # when time increses, M also increases to ensure right answer
                    M = 1e5
                    self.model.addConstr(_pp_vars[j] >= _pp_vars[i] + _i_length - (1 - y) * M) 
                    self.model.addConstr(_pp_vars[j] + _j_length <= _pp_vars[i] + y * M)
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.CONTINUOUS, name="max_start_offset")
        for lid in range(self._num_layer):
            if SPLIT_BACKPROP:
                self.model.addConstr(max_var >= self._layer_w_offsets[lid][-1] + self._layer_w_length[lid])
            else:
                self.model.addConstr(max_var >= self._layer_b_offsets[lid][-1] + self._layer_b_length[lid])

        self.model.setObjective(max_var, GRB.MINIMIZE)

    def set_baseline_solution(self):
        self.model.update()
        for var in self.model.getVars():
            if var.VarName in self.pipeline_scheduler.results.keys():
                var.Start = self.pipeline_scheduler.results[var.VarName]

    def run(self, draw=False) -> None:
        """run simulation"""
        self._build_constraints()        
        self._build_optimize_objectives()

        self.model.setParam('TimeLimit', self._time_limit)
        # self.model.setParam('MIPGap', 0.00)

        if self._base_solution:
            self.set_baseline_solution()

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
        for i in range(self._num_layer):
            recompute_rate = float(self.model_result[self._layer_recomp_rate[i].varName])
            self._layer_f_length[i] = self._profiled_layer_f_length[i] + self._profiled_additional_layer_f_length[i]
            self._layer_b_length[i] = self._profiled_layer_b_length[i] + self._profiled_layer_f_length[i] * recompute_rate + self._profiled_additional_layer_b_length[i]
            self._layer_w_length[i] = self._profiled_layer_w_length[i] + self._profiled_additional_layer_w_length[i]
        if draw:
            # 4. draws the result.
            results = {str(key) : self.model_result[key] for key in self.model_result if str(key)[0:2] in ["f_","b_","w_"]}
            self._draw(resort_microbatch_index(self._num_microbatches ,results))

        # for s in self._de_st_mb.keys():
        #     print(self._de_st_mb[s])
        return results

    def _draw(self, results: dict) -> None:
        # 绘制结果的逻辑
        painter_conf = {
            "device_size": self._num_device,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": PIXEL_BASE,
            "num_microbatches": self._num_microbatches,
            "forward_length": self._layer_f_length,
            "backward_length": self._layer_b_length,
            "backward_length2": self._layer_w_length,
            "comm_length": self._comm_length,
            "file_path": self._file_path,
            "max_time": self.model_result['max_start_offset'],
            "num_layer": self._num_layer,
        }

        SchedulingPainter(painter_conf).draw(results)