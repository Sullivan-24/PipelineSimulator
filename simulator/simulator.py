"""
simulator package
"""
import itertools
import time
import z3

from .painter import SchedulingPainter
from .utils import resort_microbatch_index

class Simulator:
    """Simulator"""

    def __init__(self, config: dict) -> None:
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
        self._comm_length               = config["communication_time"]
        
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

    def _virtual_stage_sequential_order_constraint_strict(self):
        # fix warmup stage
        # for i in range(self._pp_size):
        #     self._solver.add(
        #         self._forward_offsets[i][0]
        #         == sum(self._forward_length[0:i])
        #     )
        # Only when mb >= stage, else do not use the following contraints.
        for i in range(self._pp_size):
            for j in range(self._pp_size - i):
                self._solver.add(
                    self._forward_offsets[i][j]
                    == (sum(self._forward_length[0:i]) + sum(self._forward_length[0:j]))
                )
        mb_offset = self._num_real_microbatches
        # natural number order
        for mb in range(self._num_real_microbatches):
            for i in range(self._pp_size):
                if mb > 0:
                    self._solver.add(
                        self._forward_offsets[i][mb]
                        >= self._forward_offsets[i][mb - 1] + self._forward_length[i]
                    )
                    self._solver.add(
                        self._forward_offsets[i][mb + mb_offset]
                        >= self._forward_offsets[i][mb + mb_offset - 1] + self._forward_length[i]
                    )
                    self._solver.add(
                        self._backward_b_offsets[i][mb]
                        >= self._backward_b_offsets[i][mb - 1] + self._backward_b_length[i]
                    )
                    self._solver.add(
                        self._backward_b_offsets[i][mb + mb_offset]
                        >= self._backward_b_offsets[i][mb + mb_offset - 1] + self._backward_b_length[i]
                    )
                    self._solver.add(
                        self._backward_w_offsets[i][mb]
                        >= self._backward_w_offsets[i][mb - 1] + self._backward_w_length[i]
                    )
                    self._solver.add(
                        self._backward_w_offsets[i][mb + mb_offset]
                        >= self._backward_w_offsets[i][mb + mb_offset - 1] + self._backward_w_length[i]
                    )
        # V-shape virtual stage
        for mb in range(self._num_real_microbatches):
            # F stage constraint
            for i in range(self._pp_size):
                if i > 0:
                    self._solver.add(
                        self._forward_offsets[i][mb]
                        >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1]
                    )
                if i < self._pp_size - 1:
                    self._solver.add(
                        self._forward_offsets[i][mb + mb_offset] 
                        >= self._forward_offsets[i + 1][mb + mb_offset] + self._forward_length[i + 1]
                    )
                self._solver.add(
                        self._forward_offsets[i][mb + mb_offset]
                        >= self._forward_offsets[i][mb] + self._forward_length[i]
                    )
                self._solver.add(
                        self._backward_b_offsets[i][mb]
                        >= self._forward_offsets[i][mb + mb_offset] + self._forward_length[i]
                    )
            # B stage constraint
            for i in range(self._pp_size):
                if i > 0 :
                    self._solver.add(
                        self._backward_b_offsets[i][mb]
                        >= self._backward_b_offsets[i - 1][mb] + self._backward_b_length[i - 1]
                    )
                if i < self._pp_size - 1:
                    self._solver.add(
                        self._backward_b_offsets[i][mb + mb_offset] 
                        >= self._backward_b_offsets[i + 1][mb + mb_offset] + self._backward_b_length[i + 1]
                    )
                self._solver.add(
                        self._backward_b_offsets[i][mb + mb_offset]
                        >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                    )
            # W stage constraint
            for i in range(self._pp_size):
                self._solver.add(
                    self._backward_w_offsets[i][mb]
                    >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                )
                self._solver.add(
                    self._backward_w_offsets[i][mb + mb_offset]
                    >= self._backward_b_offsets[i][mb + mb_offset] + self._backward_b_length[i]
                )

                # constrain W order
                self._solver.add(
                    self._backward_w_offsets[i][mb + mb_offset]
                    >= self._backward_w_offsets[i][mb] + self._backward_w_length[i]
                )

    def _sequential_order_constraint_strict(self):
        for mb in range(self._num_microbatches):
            # forward stages sequential constraint
            for i in range(1, self._pp_size):
                self._solver.add(
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1]
                )
            # W stage constraint
            for i in range(self._pp_size):
                self._solver.add(
                    self._backward_w_offsets[i][mb]
                    >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                )
            # backward stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_b_offsets[i - 1][mb]
                    >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                )
                
            # forward-backward connection sequential constraint
            self._solver.add(
                self._backward_b_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1]
            )

    def _real_pipeline_modeling_constraint_strict(self):
        for mb in range(self._num_microbatches):
            # F stages sequential constraint
            for i in range(1, self._pp_size):
                self._solver.add(
                    self._forward_offsets[i][mb] 
                    >= self._forward_offsets[i - 1][mb] 
                    + self._forward_length[i-1] 
                    + self._comm_length[i-1]
                )
            
            # B stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_b_offsets[i - 1][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                    + self._comm_length[i]
                )
                
            # F-B connection sequential constraint
            self._solver.add(
                self._backward_b_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] 
                + self._forward_length[self._pp_size - 1]
            )

            # W stage sequential constraint
            for i in range(self._pp_size):
                self._solver.add(
                    self._backward_w_offsets[i][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                )

    def _sequential_order_constraint_double_interleaving(self):
        for mb in range(self._num_microbatches):
            # down pipe
            down_case = z3.And(
                *[
                    self._forward_offsets[i][mb]
                    >= self._forward_offsets[i - 1][mb] + self._forward_length[i-1]
                    for i in range(1, self._pp_size)
                ],
                *[
                    self._backward_b_offsets[i - 1][mb]
                    >= self._backward_b_offsets[i][mb] + self._backward_b_length[i]
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                self._backward_b_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] + self._forward_length[self._pp_size - 1],
            )
            # up pipe
            up_case = z3.And(
                *[
                    self._forward_offsets[i - 1][mb]
                    >= self._forward_offsets[i][mb] + self._forward_length[i]
                    for i in range(self._pp_size - 1, 0, -1)
                ],
                *[
                    self._backward_b_offsets[i][mb]
                    >= self._backward_b_offsets[i - 1][mb] + self._backward_b_length[i-1]
                    for i in range(1, self._pp_size)
                ],
                self._backward_b_offsets[0][mb]
                >= self._forward_offsets[0][mb] + self._forward_length[0],
            )

            self._solver.add(z3.Or(down_case, up_case))

    def _sequential_order_constraint_full_interleaving(self):
        for mb in range(self._num_microbatches):
            cases = []

            for perm in itertools.permutations(range(self._pp_size)):
                cases.append(
                    z3.And(
                        # forward sequential order
                        *[
                            self._forward_offsets[perm[i + 1]][mb]
                            >= self._forward_offsets[perm[i]][mb] + self._forward_length[perm[i]]
                            for i in range(len(perm) - 1)
                        ],
                        # corresponding backward order
                        *[
                            self._backward_b_offsets[perm[i - 1]][mb]
                            >= self._backward_b_offsets[perm[i]][mb]
                            + self._backward_b_length[perm[i]]
                            for i in range(len(perm) - 1, 0, -1)
                        ],
                        # forward-backward connection order
                        self._backward_b_offsets[perm[-1]][mb]
                        >= self._forward_offsets[perm[-1]][mb] + self._forward_length[perm[-1]],
                    )
                )

            # add all possibilities to z3 constraints
            self._solver.add(z3.Or(*cases))

    def _serial_computation_within_pipeline_constraint(self):
        for pp in range(self._pp_size):
            # 加入对w的判断，同时修改_length的判断
            _pp_vars = self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    _i_length = (
                        self._forward_length[pp]
                        if i // self._num_microbatches == 0 
                        else(
                            self._backward_b_length[pp] 
                            if i // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    _j_length = (
                        self._forward_length[pp]
                        if j // self._num_microbatches == 0
                        else(
                            self._backward_b_length[pp] 
                            if j // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )

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
        for pp in range(self._device_size):
            # 加入对w的判断，同时修改_length的判断
            shift = self._device_size
            _pp_vars = self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]
            while shift + pp < self._pp_size:
                _pp_vars += self._forward_offsets[shift + pp] + self._backward_b_offsets[shift + pp] + self._backward_w_offsets[shift + pp]
                shift += self._device_size
            
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    _i_length = (
                        self._forward_length[pp]
                        if i // self._num_microbatches == 0 
                        else(
                            self._backward_b_length[pp] 
                            if i // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    _j_length = (
                        self._forward_length[pp]
                        if j // self._num_microbatches == 0
                        else(
                            self._backward_b_length[pp] 
                            if j // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )

    def _build_layer_constraint(self):
        for i in range(self._pp_size):
            self._solver.add(
                self._layers[i] >= 1,
                self._layers[i] <= self._model_size
            )
        self._solver.add(sum(self._layers) == self._model_size)
        
    def _build_recomputing_rate_constraint(self):
        for i in range(self._pp_size):
            # self._solver.add(
            #     self._recomputing_rate[i] >= 0,
            #     self._recomputing_rate[i] <= 1
            # )
            discrete_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self._solver.add(z3.Or([self._recomputing_rate[i] == v for v in discrete_values]))
    
    def _update_fbw_length(self):
        self._forward_length = [self._layers[i] * self._basic_forward_length[i] for i in range(self._pp_size)]
        self._backward_b_length = [self._layers[i] * (self._basic_backward_b_length[i] + self._basic_forward_length[i] * self._recomputing_rate[i]) for i in range(self._pp_size)]
        self._backward_w_length = [self._layers[i] * self._basic_backward_w_length[i] for i in range(self._pp_size)]
    
    def _build_constraints(self) -> None:
        
        for i in range(self._pp_size):
            self._layers.append(z3.Int(f"l_{i}"))
            self._recomputing_rate.append(z3.Real(f"r_{i}"))
            for mb in range(self._num_microbatches):
                self._forward_offsets[i].append(z3.Int(f"f_{mb}_{i}"))
                self._backward_b_offsets[i].append(z3.Int(f"b_{mb}_{i}"))
                self._backward_w_offsets[i].append(z3.Int(f"w_{mb}_{i}"))

                self._solver.add(self._forward_offsets[i][-1] >= 0)
                # Skip warmup stage
                # self._solver.add(self._backward_b_offsets[i][-1] >= self._forward_length[0] * self._pp_size * self.virtual_stage[0])
                # # Skip first microbatch F+B stage
                # self._solver.add(self._backward_w_offsets[i][-1] >= self._forward_length[0] * self._pp_size * self.virtual_stage[0] + self._backward_b_length[0])
                
                # Constrain bubble size smaller to Zerobubble-GPipe, leading to up to a speedup of 15x 
                # self._solver.add(self._forward_offsets[i][-1] <= ( max(self._forward_length) + max(self._backward_b_length) ) * (self._pp_size * self.virtual_stage[0] + self._num_microbatches - 1) + self._num_microbatches * max(self._backward_w_length))
                # self._solver.add(self._backward_b_offsets[i][-1] <= ( max(self._forward_length) + max(self._backward_b_length) ) * (self._pp_size * self.virtual_stage[0] + self._num_microbatches - 1) + self._num_microbatches * max(self._backward_w_length) - self._backward_b_length[i])
                # self._solver.add(self._backward_w_offsets[i][-1] <= ( max(self._forward_length) + max(self._backward_b_length) ) * (self._pp_size * self.virtual_stage[0] + self._num_microbatches - 1) + self._num_microbatches * max(self._backward_w_length) - self._backward_w_length[i])
        
        self._build_layer_constraint()
        self._build_recomputing_rate_constraint()
        self._update_fbw_length()

        if self._sequential_order_constraint_strategy == "strict":
            # constraint 1-0: forward and backward of each microbatch
            # are executed in sequential order
            if self.virtual_stage and self.virtual_stage[0] > 1:
                self._virtual_stage_sequential_order_constraint_strict()
                # self._sequential_order_constraint_strict()
            else:
                # self._sequential_order_constraint_strict()
                self._real_pipeline_modeling_constraint_strict()
        elif self._sequential_order_constraint_strategy == "double_interleaving":
            # constraint 1-1: forward and backward of each microbatch
            # are executed in sequential order (allowing double interleaving)
            self._sequential_order_constraint_double_interleaving()
        elif self._sequential_order_constraint_strategy == "full_interleaving":
            # constraint 1-2: forward and backward of each microbatch
            # are executed in sequential order (allowing full interleaving)
            self._sequential_order_constraint_full_interleaving()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_pipeline_constraint()

        # constraint 3: the accumulation count of activations does not exceed max_activation_counts
        # self._pipeline_activation_accumulation_constraint()

    def _build_optimize_objectives(self) -> None:
        # 1. minimize the execution time of each microbatch
        max_var = z3.Int("max_start_offset")
        for pp in range(self._pp_size):
            # Change to optimize W instead of B
            for var in self._backward_w_offsets[pp]:
                self._solver.add(max_var >= var)
                # self._solver.add(max_var >= (var - self._forward_offsets[pp][0]))
        self._solver.minimize(max_var)

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 4,
            "num_real_microbatches": self._num_real_microbatches,
            "forward_length": self._forward_length,
            "backward_length": self._backward_b_length,
            "backward_length2": self._backward_w_length,
        }

        SchedulingPainter(painter_conf).draw(results)

    def run(self) -> None:
        """run simulation"""
        # 1. builds the solver constraints.
        self._build_constraints()

        # 2. builds the solver optimize objectives.
        self._build_optimize_objectives()

        # 3. runs the solver.
        start_time = time.time()
        print("Z3 Solver Solving...")
        check = self._solver.check()
        end_time = time.time()
        if  check == z3.sat:
            print(f"Result: SAT, Cost: {end_time - start_time:.2f}")
            # tranforms the result to a dictionary.
            model = self._solver.model()

            for k in model:
                # print(type(k), k, '\t', model[k])
                print(k, '\t', model[k])

            for i in range(self._pp_size):
                number_of_layers = model[self._layers[i]].as_long()
                recompute_rate = float(model[self._recomputing_rate[i]].as_fraction())
                self._forward_length[i] = self._basic_forward_length[i] * number_of_layers
                self._backward_b_length[i] = (self._basic_backward_b_length[i] + self._basic_forward_length[i] * recompute_rate) * number_of_layers 
                # self._backward_b_length[i] = z3.substitute(
                #     self._backward_b_length[i],
                #     (self._layers[i], z3.IntVal(number_of_layers)),
                #     (self._recomputing_rate[i], z3.RealVal(recompute_rate))
                # )
                self._backward_w_length[i] = self._basic_backward_w_length[i] * number_of_layers

            results = {str(key) : model[key].as_long() for key in model if str(key)[0:2] in ["f_","b_","w_"]}
            # results.pop("max_start_offset")
            # 4. draws the result.
            self._draw(resort_microbatch_index(self._num_microbatches ,results))
        else:
            print(f"Result: UNSAT, Cost: {end_time - start_time:.2f}")

class SPSimulator:
    """Simulator"""

    def __init__(self, config: dict) -> None:
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
        self._comm_length               = config["communication_time"]
        
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

        self._devices = [[] for _ in range(self._device_size)]
        self._fix_stages(stage_type="ZBV")

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
                self._solver.add(
                    self._forward_offsets[i][mb] 
                    >= self._forward_offsets[i - 1][mb] 
                    + self._forward_length[i-1] 
                    + self._comm_length[i-1]
                )
            
            # B stages sequential constraint
            for i in range(self._pp_size - 1, 0, -1):
                self._solver.add(
                    self._backward_b_offsets[i - 1][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                    + self._comm_length[i]
                )
                
            # F-B connection sequential constraint
            self._solver.add(
                self._backward_b_offsets[self._pp_size - 1][mb]
                >= self._forward_offsets[self._pp_size - 1][mb] 
                + self._forward_length[self._pp_size - 1]
            )

            # W stage sequential constraint
            for i in range(self._pp_size):
                self._solver.add(
                    self._backward_w_offsets[i][mb]
                    >= self._backward_b_offsets[i][mb]
                    + self._backward_b_length[i]
                )

    def _serial_computation_within_pipeline_constraint(self):
        for pp in range(self._pp_size):
            # 加入对w的判断，同时修改_length的判断
            _pp_vars = self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]
            for i, _ in enumerate(_pp_vars):
                for j in range(i + 1, len(_pp_vars)):
                    _i_length = (
                        self._forward_length[pp]
                        if i // self._num_microbatches == 0 
                        else(
                            self._backward_b_length[pp] 
                            if i // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    _j_length = (
                        self._forward_length[pp]
                        if j // self._num_microbatches == 0
                        else(
                            self._backward_b_length[pp] 
                            if j // self._num_microbatches == 1 
                            else self._backward_w_length[pp]
                        )
                    )
                    self._solver.add(
                        z3.Or(
                            _pp_vars[j] >= _pp_vars[i] + _i_length,
                            _pp_vars[j] + _j_length <= _pp_vars[i],
                        )
                    )

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
        for did in range(self._device_size):
            # 加入对w的判断，同时修改_length的判断
            stages_within_device = self._devices[did]
            _pp_vars = []
            for pp in stages_within_device:
                _pp_vars += self._forward_offsets[pp] + self._backward_b_offsets[pp] + self._backward_w_offsets[pp]
            
            for i, _ in enumerate(_pp_vars):
                group_size = self._num_microbatches * 3
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

    def _build_layer_constraint(self):
        for i in range(self._pp_size):
            self._solver.add(
                self._layers[i] >= 1,
                self._layers[i] <= self._model_size
            )
        self._solver.add(sum(self._layers) == self._model_size)
        
    def _build_recomputing_rate_constraint(self):
        for i in range(self._pp_size):
            # self._solver.add(
            #     self._recomputing_rate[i] >= 0,
            #     self._recomputing_rate[i] <= 1
            # )
            discrete_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            self._solver.add(z3.Or([self._recomputing_rate[i] == v for v in discrete_values]))
    
    def _update_fbw_length(self):
        self._forward_length = [self._layers[i] * self._basic_forward_length[i] for i in range(self._pp_size)]
        self._backward_b_length = [self._layers[i] * (self._basic_backward_b_length[i] + self._basic_forward_length[i] * self._recomputing_rate[i]) for i in range(self._pp_size)]
        self._backward_w_length = [self._layers[i] * self._basic_backward_w_length[i] for i in range(self._pp_size)]
    
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
        self._real_pipeline_modeling_constraint_strict()

        # constraint 2: no overlapping of forward and backward within each pipeline
        self._serial_computation_within_device_constraint()

        # constraint 3: the accumulation count of activations does not exceed max_activation_counts
        # self._pipeline_activation_accumulation_constraint()

    def _build_optimize_objectives(self) -> None:
        # 1. minimize the execution time of each microbatch
        max_var = z3.Int("max_start_offset")
        for pp in range(self._pp_size):
            # Change to optimize W instead of B
            for var in self._backward_w_offsets[pp]:
                self._solver.add(max_var >= var)
                # self._solver.add(max_var >= (var - self._forward_offsets[pp][0]))
        self._solver.minimize(max_var)

    def _draw(self, results: dict) -> None:
        painter_conf = {
            "device_size": self._device_size,
            "devices": self._devices,
            "pp_size": self._pp_size,
            "pp_height": 50,
            "pp_align": 10,
            "pixel_base": 4,
            "num_real_microbatches": self._num_real_microbatches,
            "forward_length": self._forward_length,
            "backward_length": self._backward_b_length,
            "backward_length2": self._backward_w_length,
        }

        SchedulingPainter(painter_conf).draw(results)

    def run(self) -> None:
        """run simulation"""
        # 1. builds the solver constraints.
        self._build_constraints()

        # 2. builds the solver optimize objectives.
        self._build_optimize_objectives()

        # 3. runs the solver.
        start_time = time.time()
        print("Z3 Solver Solving...")
        check = self._solver.check()
        end_time = time.time()
        if  check == z3.sat:
            print(f"Result: SAT, Cost: {end_time - start_time:.2f}")
            # tranforms the result to a dictionary.
            model = self._solver.model()

            for k in model:
                # print(type(k), k, '\t', model[k])
                print(k, '\t', model[k])

            for i in range(self._pp_size):
                number_of_layers = model[self._layers[i]].as_long()
                recompute_rate = float(model[self._recomputing_rate[i]].as_fraction())
                self._forward_length[i] = self._basic_forward_length[i] * number_of_layers
                self._backward_b_length[i] = (self._basic_backward_b_length[i] + self._basic_forward_length[i] * recompute_rate) * number_of_layers 
                # self._backward_b_length[i] = z3.substitute(
                #     self._backward_b_length[i],
                #     (self._layers[i], z3.IntVal(number_of_layers)),
                #     (self._recomputing_rate[i], z3.RealVal(recompute_rate))
                # )
                self._backward_w_length[i] = self._basic_backward_w_length[i] * number_of_layers

            results = {str(key) : model[key].as_long() for key in model if str(key)[0:2] in ["f_","b_","w_"]}
            # results.pop("max_start_offset")
            # 4. draws the result.
            self._draw(resort_microbatch_index(self._num_microbatches ,results))
        else:
            print(f"Result: UNSAT, Cost: {end_time - start_time:.2f}")
