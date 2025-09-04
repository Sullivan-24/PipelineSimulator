from e2e_data import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

fontsize=6
titlesize=20 + fontsize
labelsize=20 + fontsize
ticksize=16 + fontsize
legendsize=16 + fontsize - 0.5
fig_width = 10
fig_height = 5.5
bar_width = 0.17
def draw_strong_scaling_bar():
    # 提取x轴和方法
    x = list(strong_scaling_data.keys())
    methods = list(next(iter(strong_scaling_data.values())).keys())
    
    # 归一化基准：P=8, T=1 的 1F1B
    # baseline = strong_scaling_data["GPU#=8"]["S-1F1B"]
    baseline = strong_scaling_data["8 GPUs"]["S-1F1B"]

    # 数据矩阵 (len(x) × len(methods))
    data = np.array([[strong_scaling_data[seq][method] / baseline for method in methods] for seq in x])
    data_self = np.array([[strong_scaling_data[seq][method] / strong_scaling_data[seq]["S-1F1B"] for method in methods] for seq in x])
    max_data = max(max(data.tolist()))

    # 设置柱状图位置
    x_pos = np.arange(len(x))

    plt.figure(figsize=(fig_width, fig_height))
    for i, method in enumerate(methods):
        y_vals = data[:, i]
        y_vals = data_self[:, i]
        bars = plt.bar(x_pos + i*bar_width, data[:, i], width=bar_width,
                label=method, hatch=hatches[method],edgecolor='black', color=colors.get(method))
        
        for bar, y in zip(bars, y_vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{y:.2f}x", ha="center", va="bottom", rotation=90, color="#000000", fontsize=ticksize)
    # 坐标轴和标签
    plt.xticks(x_pos + bar_width * (len(methods) - 1) / 2, x, fontsize=ticksize-1, rotation=0)
    # plt.xlabel("Configuration (GPU#, P, T)", fontsize=labelsize)
    plt.ylabel("Throughput (Normalized)", fontsize=labelsize)
    plt.yticks(fontsize=ticksize)
    plt.ylim(0,max_data*1.2)

    # plt.title("Strong Scaling Performance", fontsize=titlesize)
    plt.legend(fontsize=legendsize,labelspacing=0.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    sns.despine()
    plt.savefig('/Users/hanayukino/strong_scaling.pdf', bbox_inches='tight')
    plt.show()

def draw_weak_scaling_bar():
    # 提取x轴和方法
    x = list(weak_scaling_data.keys())
    methods = list(next(iter(weak_scaling_data.values())).keys())
    
    # 归一化基准：P=8, T=1 的 1F1B
    # baseline = weak_scaling_data["GPU#=8"]["S-1F1B"]
    baseline = strong_scaling_data["8 GPUs"]["S-1F1B"]

    # 数据矩阵 (len(x) × len(methods))
    data = np.array([[weak_scaling_data[seq][method] / baseline for method in methods] for seq in x])
    max_data = max(max(data.tolist()))
    data_self = np.array([[weak_scaling_data[seq][method] / weak_scaling_data[seq]["S-1F1B"] for method in methods] for seq in x])

    # 设置柱状图位置
    x_pos = np.arange(len(x))

    plt.figure(figsize=(fig_width, fig_height))
    for i, method in enumerate(methods):
        y_vals = data[:, i]
        y_vals = data_self[:, i]
        bars = plt.bar(x_pos + i*bar_width, data[:, i], width=bar_width,
                label=method, hatch=hatches[method],edgecolor='black', color=colors.get(method))
        
        for bar, y in zip(bars, y_vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{y:.2f}x", ha="center", va="bottom", rotation=90, color="#000000", fontsize=ticksize)
    # 坐标轴和标签
    plt.xticks(x_pos + bar_width * (len(methods) - 1) / 2, x, fontsize=ticksize-1, rotation=0)
    # plt.xlabel("Configuration (GPU#, P, T)", fontsize=labelsize)
    plt.ylabel("Throughput (Normalized)", fontsize=labelsize)
    # plt.yticks(np.arange(0,max_data + 0.5, 0.5), fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.ylim(0,max_data*1.15)
    # plt.title("Weak Scaling Performance", fontsize=titlesize)
    plt.legend(fontsize=legendsize,labelspacing=0.25)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    sns.despine()
    plt.savefig('/Users/hanayukino/weak_scaling.pdf', bbox_inches='tight')
    plt.show()

draw_strong_scaling_bar()
draw_weak_scaling_bar()
