import os
import heapq
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        self.lock = threading.Lock()
        self.top_results = []  # 使用堆维护当前进程的Top10结果 (时间成本, 结果, placement)


    def process_placement(self, placement: list) -> tuple[float, dict, list]:
        """处理单个placement并返回时间成本、结果和placement"""
        scheduler = PipelineScheduler(placement=placement)
        scheduler.run_pipeline_parallelism()
        if not scheduler.finish_flag:
            return float('inf'), {}, placement
        
        time_cost = max(
            float(v) for k, v in scheduler.results.items()
            if str(k).startswith(("f_", "b_", "w_"))
        )
        return time_cost, scheduler.results, placement

    def parallel_generate_schedule(self, placements: list):
        """处理指定的placements列表"""
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.process_placement, p) for p in placements]
            
            for future in as_completed(futures):
                time_cost, result, placement = future.result()
                if time_cost == float('inf'):
                    continue
                
                with self.lock:
                    # 使用最小堆维护Top10（堆顶为当前第10小的值）
                    if len(self.top_results) < 10:
                        heapq.heappush(self.top_results, (-time_cost, result, placement))
                    else:
                        heapq.heappushpop(self.top_results, (-time_cost, result, placement))

    def save_temp_results(self, filename: str):
        """保存当前进程的临时结果"""
        with open(filename, 'w') as f:
            for t, res, pl in self.top_results:
                f.write(f"{t}\t{res}\t{pl}\n")

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
            return True
        return False

    def serial_generate_schedule(self):
        if not self.placement_solver:
            self.generate_placement()
        if not self.placements:
            print("Generate possible placements...")
            self.placements = self.placement_solver.get_reduced_placements()
            print(f"{len(self.placements)} Placements found.")

        print("Searching schedules on different placements...")
        for placement in self.placements:
            self.pipeline_scheduler = PipelineScheduler(placement=placement)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True, help="SLURM节点ID")
    parser.add_argument("--total-nodes", type=int, required=True, help="总节点数")
    args = parser.parse_args()

    # 初始化调度器
    us = UnifiedScheduler()
    us.generate_placement()
    
    # 获取所有placements并分片
    all_placements = us.placement_solver.get_reduced_placements()
    chunk_size = len(all_placements) // args.total_nodes
    start = args.node_id * chunk_size
    end = start + chunk_size if args.node_id != args.total_nodes - 1 else len(all_placements)
    
    # 执行计算
    us.parallel_generate_schedule(all_placements[start:end])
    
    # 保存临时结果（二进制格式更高效）
    temp_file = f"temp_node_{args.node_id}.bin"
    with open(temp_file, 'wb') as f:
        import pickle
        pickle.dump(us.top_results, f)