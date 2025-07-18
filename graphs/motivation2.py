import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

def calculate_total_variable_partitions(L: int, D: int) -> int:
    """
    计算一个 L 层的模型，在可以切分为 N 块 ( N 在 D 到 L 的范围内 )
    这个条件下的总切分情况数。

    Args:
        L (int): 模型的总层数，必须为正整数。
        D (int): 设备的数量，也是 N 的下限，必须为正整数。

    Returns:
        int: 总的切分方法数量。
    """
    # --- 1. 输入验证 ---
    if not (isinstance(L, int) and L > 0 and isinstance(D, int) and D > 0):
        raise ValueError("L 和 D 都必须是大于0的整数。")

    # 根据问题约束，N >= D 且 N <= L，这意味着必须有 L >= D 才能构成有效区间。
    if L < D:
        # 如果 L < D，那么 N 的取值范围 [D, L] 为空，所以没有可能的情况。
        return 0

    # --- 2. 循环求和 ---
    total_methods = 0
    # N 的取值范围是从 D 到 L
    for N in range(D, L + 1):
        # 对于每个 N，计算 C(L-1, N-1)
        # n = L - 1
        # k = N - 1
        # partitions_for_N = math.comb(n, k)
        partitions_for_N = math.comb(L - 1, N - 1)
        total_methods += partitions_for_N
    
    return total_methods

def count_surjective_distributions(N: int, D: int) -> int:
    """
    计算将 N 个可区分的块分配给 D 台可区分的设备，
    且每台设备至少分得一块的总方法数。
    
    这等价于计算从 N 元集到 D 元集的满射数量。

    Args:
        N (int): 块的数量，必须为正整数。
        D (int): 设备的数量，必须为正整数。

    Returns:
        int: 总的分配方法数量。
    """
    # --- 1. 输入验证 ---
    if not (isinstance(N, int) and N > 0 and isinstance(D, int) and D > 0):
        raise ValueError("N 和 D 都必须是大于0的整数。")

    # 如果块的数量 N 小于设备数量 D，则不可能每台设备都分到，方法数为 0。
    if N < D:
        return 0

    # --- 2. 应用容斥原理公式 ---
    # 公式: sum_{k=0 to D} [(-1)^k * C(D, k) * (D-k)^N]
    total_methods = 0
    for k in range(D + 1):
        term = math.comb(D, k) * ((D - k) ** N)
        if k % 2 == 1: # k是奇数
            total_methods -= term
        else: # k是偶数
            total_methods += term
            
    return total_methods

def calculate_task_sequences(M: int) -> int:
    """
    计算M个任务，每个任务有F,B,W三个有序子任务的总执行情况数。

    Args:
        M (int): 任务的数量，必须为正整数。

    Returns:
        int: 总的合法执行序列数量。
    """
    # 1. 输入验证
    if not isinstance(M, int) or M <= 0:
        raise ValueError("任务数量 M 必须是大于0的整数。")

    # 2. 计算总子任务数
    total_subtasks = 3 * M

    # 3. 计算无约束的总排列数 (3M)!
    try:
        total_permutations = math.factorial(total_subtasks)
    except OverflowError:
        # 对于非常大的M，阶乘会溢出，返回错误信息
        raise OverflowError(f"M={M} 太大，(3*M)! 的计算结果超出了标准浮点数的表示范围。")

    # 4. 计算分母 6^M
    denominator = 6 ** M

    # 5. 计算最终结果。结果保证是整数，使用整数除法 //
    # 因为这是一个组合问题，其解必然是整数。
    result = total_permutations // denominator
    
    return result


def sci_notation_e_formatter(value, pos):
    """
    接收一个数值，返回 '1eN' 格式的字符串。
    pos 参数是必需的，但在此函数中未使用。
    """
    # 处理非常小或为零的值，避免 log(0) 错误
    if value <= 0:
        return ""
    
    # 计算以10为底的对数，得到指数
    exponent = int(math.log10(value))
    return f'1e{exponent}'

# 创建一个 FuncFormatter 实例
e_formatter = FuncFormatter(sci_notation_e_formatter)

L_values = [24, 28, 32, 36, 48, 64, 80]
D_values = [4, 8, 16, 32]
M_values = [4, 8, 16, 32]
M_values = list(range(4,32,2))

offset = 5
fontsize=20
titlesize=26 + offset + 8
labelsize=26 + offset + 8
ticksize=24 + offset
legendsize=24 + offset
markersize=10 + offset
def searching_space():
    def sinplot():
        # ==============================================================================
        # 3. 创建图表和子图
        # ==============================================================================
        # 创建一个 1x3 的子图布局，并设置整体尺寸
        fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))
        # fig.suptitle('Analysis of Combinatorial Scenarios', fontsize=16)

        # --- 子图 1: calculate_total_variable_partitions ---
        ax1 = axes[0]
        for d in D_values:
            # 筛选出有效的L值 (L必须大于等于D)
            valid_L = [l for l in L_values if l >= d]
            if not valid_L:
                continue
            results = [calculate_total_variable_partitions(l, d) for l in valid_L]
            print(results)
            ax1.plot(valid_L, results, marker='o', markersize=markersize, linestyle='--', label=f'P={d}')

        ax1.set_title('Model Partition', fontsize=titlesize)
        ax1.set_xlabel('Layers#', fontsize=labelsize)
        ax1.set_ylabel('Cases# (log scale)', fontsize=labelsize)
        ax1.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
        ax1.tick_params(axis='x', which='both', length=5, labelsize=ticksize)
        ax1.set_yscale('log') # 使用对数尺度
        ax1.grid(True, which="both", ls="--")
        ax1.legend(
            fontsize=legendsize,
            labelspacing=0.25,   # 控制行间距（可调）
            columnspacing=0.25,  # 控制列间距（可调）
            handletextpad=0.25,  # 控制图例中图标与文字的间距
            borderaxespad=0.1,  # 控制图例与图像边缘的距离
            handlelength=1.25,      # 控制图例中图标的长度
        )
        x_major_locator=MultipleLocator(10)
        ax1.xaxis.set_major_locator(x_major_locator)

        # --- 子图 2: count_surjective_distributions ---
        ax2 = axes[1]
        # 注意：这里我们将 L 视为 N (块的数量)
        max_y2 = -1
        for d in D_values:
            # 筛选出有效的N值 (N必须大于等于D)
            valid_N = [n for n in L_values if n >= d]
            if not valid_N:
                continue
            results = [count_surjective_distributions(n, d) for n in valid_N]
            ax2.plot(valid_N, results, marker='o', markersize=markersize, linestyle='--', label=f'P={d}')
            max_y2 = max(max_y2, max(results))
        
        ax2.set_title('Model Placement', fontsize=labelsize)
        ax2.set_xlabel('Stages#', fontsize=labelsize)
        ax2.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
        ax2.tick_params(axis='x', which='both', length=5, labelsize=ticksize)
        ax2.set_yscale('log') # 使用对数尺度
        ax2.grid(True, which="both", ls="--")
        # ax2.legend(
        #     fontsize=legendsize,
        #     labelspacing=0.25,   # 控制行间距（可调）
        #     columnspacing=0.25,  # 控制列间距（可调）
        #     handletextpad=0.25,  # 控制图例中图标与文字的间距
        #     borderaxespad=0.1,  # 控制图例与图像边缘的距离
        #     handlelength=1.25,      # 控制图例中图标的长度
        # )
        x_major_locator=MultipleLocator(10)
        ax2.xaxis.set_major_locator(x_major_locator)

        # --- 子图 3: calculate_task_sequences ---
        ax3 = axes[2]
        valid_M = []
        results = []
        for m in M_values:
            res = calculate_task_sequences(m)
            if res != -1: # 检查是否溢出
                valid_M.append(m)
                results.append(res)
            else:
                print(f"Warning: Calculation for M={m} in calculate_task_sequences overflowed and will not be plotted.")

        if valid_M:
            ax3.plot(valid_M, results, marker='^', linestyle='-.', markersize=markersize, color='purple', label=f'P=1')

        ax3.set_title('Workload Scheduling', fontsize=titlesize)
        ax3.set_xlabel('Micro-batches#', fontsize=labelsize)
        ax3.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
        ax3.tick_params(axis='x', which='both', length=5, labelsize=ticksize)
        ax3.set_yscale('log') # 使用对数尺度
        ax3.grid(True, which="both", ls="--")
        x_major_locator=MultipleLocator(5)
        ax3.xaxis.set_major_locator(x_major_locator)
        # ax3.text(0.5, -0.5, 'Workload Scheduling', 
        #     fontsize=titlesize, 
        #     ha='center', 
        #     va='center',
        #     transform=ax3.transAxes
        # )
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((1, 10))  # 强制使用科学记数法
        # ax3.yaxis.set_major_formatter(formatter)
        # ax3.yaxis.set_major_formatter(e_formatter)
        if results:
            max_y3 = max(results)
            current_ylim = ax3.get_ylim()
            ax3.set_ylim(current_ylim[0], max_y3 * 10000.0)

        ax3.legend(
            fontsize=legendsize,
            labelspacing=0.25,   # 控制行间距（可调）
            columnspacing=0.25,  # 控制列间距（可调）
            handletextpad=0.25,  # 控制图例中图标与文字的间距
            borderaxespad=0.1,  # 控制图例与图像边缘的距离
            handlelength=1.25,      # 控制图例中图标的长度
        )
        # ==============================================================================
        # 4. 调整布局并显示图表
        # ==============================================================================
        # plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/searching_space.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

searching_space()