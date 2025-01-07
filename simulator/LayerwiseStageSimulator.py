import time
from gurobipy import Model, GRB, quicksum
from .painter import SchedulingPainter
from .abstract.mutils import *
from .utils import resort_microbatch_index, print_to_file
from .abstract import Pipeline
class LayerwiseSimulator:

    def __init__(self, config: dict, device_stage_alignments=None, new_comm_length=None) -> None:
        self._base_solution = config['base_solution']
        self._schedule_method = config['schedule_method']
        self._file_path = config["file_path"]
        self._time_limit = config["time_limit"]
        self._pp_size = config["pp_size"]
        self._device_size = config["device_size"]
        self._model_layer_num = config["model_size"]
        self._num_microbatches = config["num_microbatches"]
        self._max_activation_counts = config["max_activation_counts"]
        
        self._mix_training = config["mix_training"]
        self._model_para_num = config["model_para_num"]
        self._device_mem = config["device_mem"]
        # obtained by profiling
        self._profiled_layer_f_length = config["forward_execution_time"]
        self._profiled_layer_b_length = config["backward_execution_i_time"]
        self._profiled_layer_w_length = config["backward_execution_g_time"]
        
        self._profiled_additional_layer_f_length = [EMBEDDING_TIME  if lid == 0 else (HEAD_F_TIME if lid == LAYER_NUM-1 else 0) for lid in range(LAYER_NUM)]
        self._profiled_additional_layer_b_length = [HEAD_B_TIME if lid == LAYER_NUM-1 else 0 for lid in range(LAYER_NUM)]
        self._profiled_additional_layer_w_length = [0 for _ in range(LAYER_NUM)]
        
        self._comm_length = config["communication_time"] if not new_comm_length else new_comm_length
        
        # 检查输入参数
        assert isinstance(self._profiled_layer_f_length, (list, tuple))
        assert isinstance(self._profiled_layer_b_length, (list, tuple))
        assert isinstance(self._profiled_layer_w_length, (list, tuple))

        # 创建 Gurobi 模型
        self.model = Model("SPSimulator")

        self.minimal_time_with_sync_update = (DEVICE_NUM - 1) * (F_TIME // CHUNK_NUM + COMM_TIME) + (F_TIME + B_TIME + W_TIME) * MICRO_BATCH_NUM
        print("MINIMAL TIME WITH SYNC UPDATE:{}".format(self.minimal_time_with_sync_update))

        # 变量初始化
        self._stage_f_length  = []
        self._stage_b_length = []
        self._stage_w_length = []

        self._layer_f_length  = self._profiled_layer_f_length
        self._layer_b_length = self._profiled_layer_b_length
        self._layer_w_length = self._profiled_layer_w_length

        self._layer_recomp_rate = []
        self._layers = []
        self._stage_layers_mapping = [[] for _ in range(self._pp_size)]

        self._stage_f_offsets = [[] for _ in range(self._pp_size)]
        self._stage_b_offsets = [[] for _ in range(self._pp_size)]
        self._stage_w_offsets = [[] for _ in range(self._pp_size)]

        if device_stage_alignments:
            self._devices = device_stage_alignments
        else:
            self._devices = [[] for _ in range(self._device_size)]
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
        if self._schedule_method in (SchedulePriority.ZBV, SchedulePriority.GREEDY_v1, SchedulePriority.GREEDY_v2):
            for pid in range(self._pp_size):
                if (pid // self._device_size) % 2 == 0:
                    self._devices[pid % self._device_size].append(pid)
                else:
                    self._devices[self._device_size - 1 - pid % self._device_size].append(pid)
        elif self._schedule_method in (SchedulePriority.ZBH1, SchedulePriority.ONE_F_ONE_B, SchedulePriority.INTERLEAVED):
            for pid in range(self._pp_size):
                self._devices[pid % self._device_size].append(pid)
        else:
            assert("Stage alignment is not set.")

    def _reset_comm_length(self, dsa):
        new_comm_length = [[COMM_TIME if i != j else 0 for j in range(self._pp_size)] for i in range(self._pp_size)]
        for d in dsa:
            for i in range(len(d)):
                for j in range(i+1, len(d)):
                    new_comm_length[d[i]][d[j]] = 0
                    new_comm_length[d[j]][d[i]] = 0
        return new_comm_length
    
    def _generate_variables(self):
        for l in range(self._model_layer_num):
            self._layer_recomp_rate.append(self.model.addVar(vtype=GRB.BINARY, name=f"theta_{l}"))
            self.model.addConstr(self._layer_recomp_rate[-1] == 0)
            
        for sid in range(self._pp_size):
            
            for lid in range(self._model_layer_num):
                self._stage_layers_mapping[sid].append(self.model.addVar(vtype=GRB.BINARY, name=f"l{lid}_in_s{sid}_flag"))

            for mid in range(self._num_microbatches):
                
                self._stage_f_offsets[sid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"f_{mid}_{sid}", lb=0))
                self._stage_b_offsets[sid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"b_{mid}_{sid}", lb=0))
                if SPLIT_BACKPROP:
                    self._stage_w_offsets[sid].append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"w_{mid}_{sid}", lb=0))
            
            self._stage_f_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{sid}_f"))
            self._stage_b_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{sid}_b"))
            self._stage_w_length.append(self.model.addVar(vtype=GRB.CONTINUOUS, name=f"s{sid}_w"))
            
            # Set length per stage
            f_result = quicksum([self._stage_layers_mapping[sid][lid] * self._layer_f_length[lid] for lid in range(self._model_layer_num)])
            additional_f_result = quicksum([self._stage_layers_mapping[sid][lid] * self._profiled_additional_layer_f_length[lid] for lid in range(self._model_layer_num)])
            self.model.addConstr(self._stage_f_length[sid] == f_result + additional_f_result)

            b_result = quicksum([self._stage_layers_mapping[sid][lid] * (self._layer_b_length[lid] +  self._layer_recomp_rate[lid] * self._layer_f_length[lid]) for lid in range(self._model_layer_num)])
            additional_b_result = quicksum([self._stage_layers_mapping[sid][lid] * self._profiled_additional_layer_b_length[lid] for lid in range(self._model_layer_num)])
            self.model.addConstr(self._stage_b_length[sid] == b_result + additional_b_result)

            w_result = quicksum([self._stage_layers_mapping[sid][lid] * self._layer_w_length[lid] for lid in range(self._model_layer_num)])
            additional_w_result = quicksum([self._stage_layers_mapping[sid][lid] * self._profiled_additional_layer_w_length[lid] for lid in range(self._model_layer_num)])
            self.model.addConstr(self._stage_w_length[sid] == w_result + additional_w_result)

    def _set_stage_layer_mapping_constraints(self):
        self.model.addConstr(
            quicksum(
                [quicksum(self._stage_layers_mapping[sid]) for sid in range(self._pp_size)]
            )
            == self._model_layer_num
        )
        print(self._pp_size, self._model_layer_num)
        for sid in range(self._pp_size):
            for lid in range(self._model_layer_num):
                if sid == lid:
                    self.model.addConstr(
                        self._stage_layers_mapping[sid][lid] == 1
                    )
                else:
                    self.model.addConstr(
                        self._stage_layers_mapping[sid][lid] == 0
                    )

    def _build_constraints(self) -> None:
        self._generate_variables()
        self._set_stage_layer_mapping_constraints()

        self._get_device_stage_microbatch_alignment()
        # 添加约束
        # self.model.addConstr(quicksum(self._layers) == self._model_layer_num)
        self._comm_length = self._reset_comm_length(self._devices)
        self._real_pipeline_modeling_constraint_strict()
        self._serial_computation_within_device_constraint()
        # self._pipeline_activation_accumulation_constraint()

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            for i in range(1, self._pp_size):
                self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i - 1][mb] +
                                     self._stage_f_length[i - 1] + self._comm_length[i - 1][i])

            for i in range(self._pp_size - 1, 0, -1):
                self.model.addConstr(self._stage_b_offsets[i - 1][mb] >= self._stage_b_offsets[i][mb] +
                                     self._stage_b_length[i] + self._comm_length[i][i - 1])

            self.model.addConstr(self._stage_b_offsets[self._pp_size - 1][mb] >= self._stage_f_offsets[self._pp_size - 1][mb] +
                                 self._stage_f_length[self._pp_size - 1])

            if SPLIT_BACKPROP:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_w_offsets[i][mb] >= self._stage_b_offsets[i][mb] +
                                        self._stage_b_length[i])

            if mb > 0:
                for i in range(self._pp_size):
                    self.model.addConstr(self._stage_f_offsets[i][mb] >= self._stage_f_offsets[i][mb - 1] +
                                         self._stage_f_length[i])
                    self.model.addConstr(self._stage_b_offsets[i][mb] >= self._stage_b_offsets[i][mb - 1] +
                                         self._stage_b_length[i])
                    if SPLIT_BACKPROP:
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
                if SPLIT_BACKPROP:
                    _pp_vars += self._stage_w_offsets[pp]
            type_of_workload = 3 if SPLIT_BACKPROP else 2
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
                    # when time increses, M also increases to ensure right answer
                    M = 1e5
                    self.model.addConstr(_pp_vars[j] >= _pp_vars[i] + _i_length - (1 - y) * M) 
                    self.model.addConstr(_pp_vars[j] + _j_length <= _pp_vars[i] + y * M)
                    
        print_to_file(self._file_path, "Total Constraints within Device:{}, Redundant Constraints:{}.\n".format(total_constraints, same_mb_redundant_constraints))

    def _get_device_stage_microbatch_alignment(self):
        _de_vars = {}
        for did in range(self._device_size):
            stages_within_device = self._devices[did]
            _pp_vars = {}
            for sid in stages_within_device:
                _mt_vars = {'f':[],'b':[],'w':[],'pf':[],'pb':[],'pw':[]}
                for mid in range(self._num_microbatches):
                    _mt_vars['f'].append(self._stage_f_offsets[sid][mid])
                    _mt_vars['b'].append(self._stage_b_offsets[sid][mid])
                    if SPLIT_BACKPROP:
                        _mt_vars['w'].append(self._stage_w_offsets[sid][mid])
                    
                    _mt_vars['pf'].append([self.model.addVar(name=f'act_f_{mid}_{sid}_{did}', vtype=GRB.BINARY) for mid in range(self._num_microbatches * CHUNK_NUM)])
                    # _mt_vars['pb'].append([self.model.addVar(name=f'act_b_{mid}_{sid}_{did}', vtype=GRB.BINARY) for mid in range(self._num_microbatches * CHUNK_NUM)])
                    _mt_vars['pw'].append([self.model.addVar(name=f'act_w_{mid}_{sid}_{did}', vtype=GRB.BINARY) for mid in range(self._num_microbatches * CHUNK_NUM)])

                _pp_vars[sid] = _mt_vars
            _de_vars[did] = _pp_vars
        self._de_st_mb = _de_vars

    def _offload_reload_strategy_constraint(self):
        pass

    def _pipeline_activation_accumulation_constraint(self):
        if MAX_ACTIVATION_COUNTS >= MICRO_BATCH_NUM * CHUNK_NUM:
            print("ACTIVATION LIMIT FREE")
            return
        #对每个Device上的所有W或B Microbatch进行检查：保证当前Device上的activation积累不超过上限 MAX_ACTIVATION_COUNTS * CHUNK_NUM
        mt = 'w' if SPLIT_BACKPROP else 'b' 
        for did in self._de_st_mb:
            for sid in self._de_st_mb[did]:
                for mid in range(self._num_microbatches):
                    pivot = self._de_st_mb[did][sid][mt][mid]
                    for o_mid in range(self._num_microbatches):
                        for idx, o_sid in enumerate(self._de_st_mb[did]):
                            binary_f = self.model.addVar(vtype=GRB.BINARY, name=f'binary_f_{sid}_{mid}_{o_sid}_{o_mid}')
                            binary_w = self.model.addVar(vtype=GRB.BINARY, name=f'binary_w_{sid}_{mid}_{o_sid}_{o_mid}')

                            eps = 0.001
                            M = 100000
                            self.model.addConstr(pivot >= self._de_st_mb[did][o_sid]['f'][o_mid] + eps - M * (1 - binary_f) )
                            self.model.addConstr(pivot <= self._de_st_mb[did][o_sid]['f'][o_mid] + M * binary_f)

                            self.model.addConstr(pivot >= self._de_st_mb[did][o_sid][mt][o_mid] + eps - M * (1 - binary_w) )
                            self.model.addConstr(pivot <= self._de_st_mb[did][o_sid][mt][o_mid] + M * binary_w)

                            self.model.addConstr((binary_f == 1) >> (self._de_st_mb[did][sid]['pf'][mid][o_mid + idx * self._num_microbatches] == 1))
                            self.model.addConstr((binary_f == 0) >> (self._de_st_mb[did][sid]['pf'][mid][o_mid + idx * self._num_microbatches] == 0))
                            
                            self.model.addConstr((binary_w == 1) >> (self._de_st_mb[did][sid]['pw'][mid][o_mid + idx * self._num_microbatches] == 1))
                            self.model.addConstr((binary_w == 0) >> (self._de_st_mb[did][sid]['pw'][mid][o_mid + idx * self._num_microbatches] == 0))
                    self.model.addConstr(
                        # set recomputing rate to 0 when searching procedure is slow
                        quicksum(self._de_st_mb[did][sid]['pf'][mid]) * (1 - 0) - quicksum(self._de_st_mb[did][sid]['pw'][mid]) 
                        # quicksum(self._de_st_mb[did][sid]['pf'][mid]) * (1 - self._recomputing_rate[sid]) - quicksum(self._de_st_mb[did][sid]['pw'][mid]) 
                        <= 
                        self._max_activation_counts[did]
                    )

    def _build_optimize_objectives(self) -> None:
        max_var = self.model.addVar(vtype=GRB.CONTINUOUS, name="max_start_offset")
        for pp in range(self._pp_size):
            if SPLIT_BACKPROP:
                self.model.addConstr(max_var >= self._stage_w_offsets[pp][-1] + self._stage_w_length[pp])
            else:
                self.model.addConstr(max_var >= self._stage_b_offsets[pp][-1] + self._stage_b_length[pp])

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
        for i in range(self._pp_size):
            # number_of_layers = self.model_result[self._layers[i].varName]
            number_of_layers = sum([self.model_result[var_name] for var_name in [self._stage_layers_mapping[i][j].varName for j in range(self._model_layer_num)]])
            recompute_rate = float(self.model_result[self._layer_recomp_rate[i].varName])
            self._stage_f_length[i] = self._profiled_layer_f_length[i] * number_of_layers + self._profiled_additional_layer_f_length[i]
            self._stage_b_length[i] = (self._profiled_layer_b_length[i] + self._profiled_layer_f_length[i] * recompute_rate) * number_of_layers + self._profiled_additional_layer_b_length[i]
            self._stage_w_length[i] = self._profiled_layer_w_length[i] * number_of_layers + self._profiled_additional_layer_w_length[i]
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
            "device_size": self._device_size,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": PIXEL_BASE,
            "num_microbatches": self._num_microbatches,
            "forward_length": self._stage_f_length,
            "backward_length": self._stage_b_length,
            "backward_length2": self._stage_w_length,
            "comm_length": self._comm_length,
            "file_path": self._file_path,
            "max_time": self.model_result['max_start_offset'],
        }

        SchedulingPainter(painter_conf).draw(results)