import math
import itertools

def generate_unique_placement_mapping(n):
    if n < 0:
        return None  # 根据实际情况处理非法输入
    if n == 0:
        return None  # 题目要求空数组返回None
    
    seen = set()
    result = []
    
    # 生成所有可能的二进制数组
    for arr in itertools.product([0, 1], repeat=n):
        arr_tuple = arr
        if arr_tuple not in seen:
            # 添加到结果列表
            result.append(list(arr_tuple))
            # 计算补集并标记为已处理
            complement = tuple(1 - x for x in arr_tuple)
            seen.add(arr_tuple)
            seen.add(complement)
    
    return result

class PipelinePlacement:
    def __init__(self, 
                layer_num:int, 
                layer_computation_cost:list[float], 
                layer_para: list[float],
                dev_num:int, 
                dev_max_memory:list[float], 
                dev_compute_power:list[float]):
        self.layer_num = layer_num
        self.layer_computation_cost = layer_computation_cost
        self.layer_para = layer_para
        self.dev_num = dev_num
        self.dev_max_memory = dev_max_memory
        self.dev_compute_power = dev_compute_power

    def get_placement1s(self):
        dsa = [[] for _ in range(self.dev_num)]
        """
            实现将self.layer_num个layer分配到self.pp_size个device上
            例如：
            dsa = [
                [0, 1],
                [2, 4],
                [3, 5],
                [6, 7],
                [8, 9, 11],
                [10, 12, 13],
            ]
            第i个layer的计算开销是self.layer_computation_cost[i]
            使用第j个device计算的时间为 
            time = 0
            for lid in dsa[j]
                time += self.layer_computation_cost[lid] / self.pp_compute_power[j]
            
            要求给出所有device上计算时间方差最小的layer分配结果，即dsa
        """
        return dsa
    
    def get_placements(self):
        # Sort layers by computation cost in descending order, keeping their original indices
        sorted_layers = sorted(enumerate(self.layer_computation_cost), key=lambda x: -x[1])
        
        # Initialize devices: memory_used, total_time, and layers list
        devices = [
            {
                'memory_used': 0.0,
                'total_time': 0.0,
                'layers': []
            }
            for _ in range(self.dev_num)
        ]
        
        for layer_idx, comp_cost in sorted_layers:
            layer_mem = self.layer_para[layer_idx]
            best_dev = None
            min_max_time = float('inf')
            min_new_time = float('inf')
            
            # Find the best device to place the current layer
            for did in range(self.dev_num):
                # Check if the device has enough memory
                if devices[did]['memory_used'] + layer_mem > self.dev_max_memory[did]:
                    continue
                
                # Calculate new total time for this device if the layer is placed here
                new_time = devices[did]['total_time'] + (comp_cost / self.dev_compute_power[did])
                
                # Calculate the candidate max time across all devices after placement
                current_max_time = 0.0
                for other_dev_idx in range(self.dev_num):
                    if other_dev_idx == did:
                        current_time = new_time
                    else:
                        current_time = devices[other_dev_idx]['total_time']
                    if current_time > current_max_time:
                        current_max_time = current_time
                
                # Update the best device based on the minimal max_time and new_time
                if (current_max_time < min_max_time) or \
                   (current_max_time == min_max_time and new_time < min_new_time):
                    best_dev = did
                    min_max_time = current_max_time
                    min_new_time = new_time
            
            if best_dev is None:
                raise RuntimeError(f"Cannot place layer {layer_idx} due to insufficient memory on all devices.")
            
            # Update the best device's status
            devices[best_dev]['memory_used'] += layer_mem
            devices[best_dev]['total_time'] = min_new_time
            devices[best_dev]['layers'].append(layer_idx)
        
        # Prepare the result, sorting the layer indices for each device
        dsa = []
        for dev in devices:
            dev['layers'].sort()
            dev['layer_num'] = len(dev['layers'])
            dsa.append(dev['layers'])
        for device in devices:
            print(device)
        for dsa_in_pp in dsa:
            print(dsa_in_pp)
        return dsa
    
    def get_placements_dp(self):
        # 按计算成本降序排序，保留原始索引
        sorted_layers = sorted(enumerate(self.layer_computation_cost), 
                             key=lambda x: (-x[1], x[0]))
        
        # 初始化动态规划表
        dp = [[] for _ in range(self.layer_num + 1)]
        initial_memory = tuple([0.0] * self.dev_num)
        initial_time = tuple([0.0] * self.dev_num)
        initial_layers = tuple([tuple() for _ in range(self.dev_num)])
        dp[0].append((initial_memory, initial_time, initial_layers))
        
        # 处理每个层
        for step in range(self.layer_num):
            layer_idx, comp_cost = sorted_layers[step]
            layer_mem = self.layer_para[layer_idx]
            
            next_states = []
            for state in dp[step]:
                current_memory, current_time, current_layers = state
                
                # 尝试将当前层分配到每个设备
                for dev in range(self.dev_num):
                    # 检查显存限制
                    if current_memory[dev] + layer_mem > self.dev_max_memory[dev]:
                        continue
                    
                    # 更新显存和计算时间
                    new_memory = list(current_memory)
                    new_memory[dev] += layer_mem
                    new_memory = tuple(new_memory)
                    
                    new_time = list(current_time)
                    new_time[dev] += comp_cost / self.dev_compute_power[dev]
                    new_time = tuple(new_time)
                    
                    # 更新层分配
                    new_layers = list(current_layers)
                    new_layers[dev] = tuple(list(new_layers[dev]) + [layer_idx])
                    new_layers = tuple(new_layers)
                    
                    next_states.append((new_memory, new_time, new_layers))
            
            # 剪枝：去除被支配的状态
            pruned_states = []
            for state in next_states:
                dominated = False
                for other in next_states:
                    if state == other:
                        continue
                    if self.is_dominated(state, other):
                        dominated = True
                        break
                if not dominated:
                    pruned_states.append(state)
            dp[step+1] = pruned_states
        
        # 寻找最优解
        if not dp[self.layer_num]:
            raise RuntimeError("No valid placement found")
        
        min_variance = float('inf')
        best_layers = None
        for state in dp[self.layer_num]:
            times = state[1]
            mean = sum(times) / self.dev_num
            variance = sum((t - mean)**2 for t in times) / self.dev_num
            if variance < min_variance:
                min_variance = variance
                best_layers = state[2]
        
        # 转换为要求的格式并排序
        dsa = []
        for layers in best_layers:
            sorted_layers = sorted(layers)
            dsa.append(sorted_layers)
        for dsa_in_pp in dsa:
            print(dsa_in_pp)
        return dsa
    
    def is_dominated(self, state_a, state_b):
        """检查state_a是否被state_b支配"""
        mem_a, time_a, _ = state_a
        mem_b, time_b, _ = state_b
        
        all_less_equal = True
        has_strict_less = False
        
        for m_a, m_b in zip(mem_a, mem_b):
            if m_b > m_a:
                all_less_equal = False
                break
        
        for t_a, t_b in zip(time_a, time_b):
            if t_b > t_a:
                all_less_equal = False
                break
        
        if not all_less_equal:
            return False
        
        for m_a, m_b in zip(mem_a, mem_b):
            if m_b < m_a:
                has_strict_less = True
                break
        
        for t_a, t_b in zip(time_a, time_b):
            if t_b < t_a:
                has_strict_less = True
                break
        
        return has_strict_less

# 示例测试
if __name__ == "__main__":
    # 输入参数
    # L = 64
    # D = 8
    # Tl = [1 for _ in range(L)]
    # Cd = [1 for _ in range(D)]  # 机器0算力2, 机器1算力1
    
    # # 运行分配算法
    # allocation, chunk_max, total_max = allocate_layers(L, D, Tl, Cd)
    
    # # 打印结果
    # print("层到机器的分配结果:")
    # for layer in sorted(allocation):
    #     print(f"层 {layer} → 机器 {allocation[layer]}")
    
    # print("\n每个chunk的最大时间:", chunk_max)
    # print("流水线整体最大时间:", total_max)
    pp_size = 8
    layer_num = 82
    layer_computation_cost = [1 for _ in range(layer_num)]
    layer_computation_cost[1] = 1
    layer_computation_cost[2] = 1
    pp_compute_power = [1 for _ in range(pp_size)]
    pp_compute_power[0] = 0.5
    pp_compute_power[1] = 0.5
    pp_compute_power[2] = 0.5
    pp_compute_power[3] = 0.5
    
    test_placement = PipelinePlacement(layer_num=layer_num, 
                                    layer_computation_cost=layer_computation_cost,
                                    layer_para=[1 for _ in range(layer_num)],
                                    dev_num=pp_size,
                                    dev_max_memory=[100000 for _ in range(layer_num)],
                                    dev_compute_power=pp_compute_power,
                                    )
    test_placement.get_placements()
    # pp4 layer8 也不行
    # test_placement.get_placements_dp()
