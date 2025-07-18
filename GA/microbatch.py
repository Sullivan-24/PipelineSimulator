import random
from itertools import permutations
import matplotlib.pyplot as plt

# 批次数
num_batches = 4

# 正向加工时间（D1 -> D4）
F = [3, 2, 4, 5]

# 反向加工时间（D4 -> D1）
B = [5, 4, 2, 3]

# 设备定义
devices_forward = ['D1', 'D2', 'D3', 'D4']
devices_backward = ['D4', 'D3', 'D2', 'D1']
device_names = devices_forward + devices_backward
num_devices = len(device_names)

# 解码调度（正向→反向依赖）
def decode_with_fb_dependency(chromosome):
    device_timeline = [0] * num_devices
    batch_times = {}

    for batch in chromosome:
        batch_key = f"B{batch}"

        # Forward (D1 → D4)
        start_time = 0
        for i, dev in enumerate(devices_forward):
            idx = i
            proc_time = F[i]
            start = max(start_time, device_timeline[idx])
            end = start + proc_time
            device_timeline[idx] = end
            start_time = end
            batch_times[(batch_key, dev)] = (start, end)

        forward_finish_time = start_time

        # Backward (D4 → D1)
        for i, dev in enumerate(devices_backward):
            idx = i + len(devices_forward)
            proc_time = B[i]
            start = max(forward_finish_time, device_timeline[idx])
            end = start + proc_time
            device_timeline[idx] = end
            forward_finish_time = end
            batch_times[(batch_key, dev)] = (start, end)

    makespan = max(device_timeline)
    return makespan, batch_times

# 适应度函数
def fitness(chrom):
    makespan, _ = decode_with_fb_dependency(chrom)
    return 1 / makespan

# 初始化种群
def init_population(size):
    population = []
    for _ in range(size):
        p = list(range(1, num_batches + 1))
        random.shuffle(p)
        population.append(p)
    return population

# 选择（锦标赛）
def selection(population, k=3):
    selected = []
    for _ in range(len(population)):
        aspirants = random.sample(population, k)
        aspirants.sort(key=lambda x: fitness(x), reverse=True)
        selected.append(aspirants[0])
    return selected

# 交叉（部分映射交叉 PMX）
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    p1, p2 = parent1[:], parent2[:]
    mapping = dict(zip(p1[a:b], p2[a:b]))
    child = [None] * size
    child[a:b] = p2[a:b]
    for i in list(range(0, a)) + list(range(b, size)):
        n = p1[i]
        while n in mapping:
            n = mapping[n]
        child[i] = n
    return child

# 变异（交换）
def mutate(chrom):
    a, b = random.sample(range(len(chrom)), 2)
    chrom[a], chrom[b] = chrom[b], chrom[a]
    return chrom

# 遗传算法主函数
def genetic_algorithm_fb(generations=30, pop_size=20, early_stop=10):
    population = init_population(pop_size)
    best = min(population, key=lambda x: decode_with_fb_dependency(x)[0])
    best_makespan = decode_with_fb_dependency(best)[0]
    no_improve_count = 0

    for _ in range(generations):
        selected = selection(population)
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                c1 = crossover(selected[i], selected[i + 1])
                c2 = crossover(selected[i + 1], selected[i])
                offspring.extend([mutate(c1), mutate(c2)])
        population = offspring
        current_best = min(population, key=lambda x: decode_with_fb_dependency(x)[0])
        current_makespan = decode_with_fb_dependency(current_best)[0]
        if current_makespan < best_makespan:
            best = current_best
            best_makespan = current_makespan
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= early_stop:
            break
    return best, decode_with_fb_dependency(best)

# 甘特图可视化
def plot_schedule(schedule):
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum']
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, dev in enumerate(device_names):
        for (batch, d), (start, end) in schedule.items():
            if d == dev:
                ax.barh(i, end - start, left=start, color=colors[int(batch[1]) - 1], edgecolor='black')
                ax.text((start + end) / 2, i, batch, va='center', ha='center', fontsize=8, color='black')
    ax.set_yticks(range(len(device_names)))
    ax.set_yticklabels(device_names)
    ax.set_xlabel("Time")
    ax.set_title("Batch Scheduling Gantt Chart")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 运行算法
best_chromosome, (best_makespan, best_schedule) = genetic_algorithm_fb(generations=2, pop_size=20)
print("最佳调度顺序:", best_chromosome)
print("最小完工时间:", best_makespan)

# 可视化结果
plot_schedule(best_schedule)
