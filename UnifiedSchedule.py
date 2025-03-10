from simulator.abstract.Pipeline import *
from simulator.abstract.Placement import PipelinePlacement
from simulator.config import *
class UnifiedScheduler:
    def __init__(self,layer_num=LAYER_NUM,
                layer_computation_cost=None,
                layer_para=None,
                dev_num=DEVICE_NUM,
                dev_max_memory=None,
                dev_compute_power=None,
                ):
        self.pipeline_scheduler = None
        self.placement_solver = None
        self.placements = None
        self.layer_num = layer_num
        self.layer_computation_cost = layer_computation_cost if layer_computation_cost else [1 for _ in range(layer_num)]
        self.layer_para = layer_para if layer_para else [1 for _ in range(layer_num)]
        self.dev_num = dev_num
        self.dev_max_memory = dev_max_memory if dev_max_memory else [layer_num + 1 for _ in range(dev_num)]
        self.dev_compute_power = dev_compute_power if dev_compute_power else [1 for _ in range(dev_num)]

        self.best_placement = None
        self.best_schedule = None
        self.best_time_cost = 1000000

    def save_to_file(self):
        with open("searched_schedule", 'w') as file:
            file.write(str(self.best_schedule))
        with open("searched_placement", 'w') as file:
            file.write(str(self.best_placement))    

    def is_better_schedule(self, schedule_result:dict, placement:list):
        time_cost = -1
        for k, v in schedule_result.items():
            if str(k).startswith(("f_","b_","w_")):
                time_cost = max(time_cost, float(v))
        if time_cost < self.best_time_cost:
            print("Update result from {} -> {}".format(self.best_time_cost, time_cost))
            self.best_time_cost = time_cost
            self.best_schedule = schedule_result
            self.best_placement = placement
            self.save_to_file()
            return True
        return False

    def generate_schedule(self):
        if not self.placement_solver:
            self.generate_placement()
        if not self.placements:
            print("Generate possible placements...")
            self.placements = self.placement_solver.get_reduced_placements()
            print(f"{len(self.placements)} Placements found.")

        print("Searching schedules on different placements...")
        for placement in self.placements:
            self.pipeline_scheduler = PipelineScheduler(dsa=placement)
            self.pipeline_scheduler.run_pipeline_parallelism()
            if self.pipeline_scheduler.finish_flag:
                print("Simulating Finished Successfully.")
                temp_result = self.pipeline_scheduler.results
                if self.is_better_schedule(temp_result, placement):
                    print("Better Schedule+Placement Found.")

    def generate_placement(self):
        if self.placement_solver is None:
            self.placement_solver = PipelinePlacement(
                layer_num=self.layer_num,
                layer_computation_cost=self.layer_computation_cost,
                layer_para=self.layer_para,
                dev_num=self.dev_num,
                dev_max_memory=self.dev_max_memory,
                dev_compute_power=self.dev_compute_power,
            )
    


if __name__ == "__main__":
    us = UnifiedScheduler()
    us.generate_schedule()