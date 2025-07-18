import numpy as np
import matplotlib.pyplot as plt
from e2e_data import *
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
import matplotlib.gridspec as gridspec
plt.rcParams['font.family'] = 'Helvetica'

fontsize=10
titlesize=22 + fontsize
labelsize= 20 + fontsize
ticksize= 16 + fontsize
legendsize= 20 + fontsize - 0.5
def bubble_comparison():
    def sinplot():
        # 方法名称
        methods = ['S-1F1B', 'I-1F1B', 'ZB', 'Mist', 'OctoPipe']
        colors = ['#ffde82','#4A90E2','#48BEA3','#E1D5E7','#E54C5E']
        # 每个方法在 S1 到 S4 和 Avg 的百分比
        # data = np.array([
        #     [45.95, 48.11, 48.11, 20.0, 40.5425],
        #     [35.48, 38.06, 38.06, 4.52, 29.03],
        #     [37.69, 40.19, 40.19, 7.79, 31.465],
        #     [27.04, 29.64, 29.64, 27.04, 28.34],
        #     # [2.82, 6.29, 6.29, 2.82, 4.55],
        # ])   
        data = np.array([
        # ['Ideal', 'LLaMA3', 'Gemma', 'DeepSeek']
            [27.55,   35.4025, 40.5425, 47.6925],
            [4.95,    20.61,   29.03,   22.2675],
            [11.52,   22.265,  31.465,  41.55],
            [27.55,   28.77,   28.34,   31.3425],
            [2.04, 5.9375, 7.175, 5.455],
        ])   
        # Ideal [27.55, 4.95, 11.52, 27.55]
        # GPT3 [29.785, 10.81, 19.2, 31.25, 6.2175]
        # llama3 [35.4025, 20.61, 22.265, 28.77]
        # gemma [40.5425, 29.03, 31.465, 28.34]
        # DeepSeek [47.6925, 22.2675, 41.55, 31.3425]

        # X轴标签（stage）
        labels = ['S1', 'S2', 'S3', 'S4', 'Avg.']
        labels = ['Ideal', 'LLaMA3', 'Gemma', 'DeepSeek']

        # labels = ['S1', 'S2', 'S3', 'S4']

        x = np.arange(len(labels))  # [0, 1, 2, 3, 4]
        bar_width = 0.15

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 每种方法一个 bar group
        for i in range(len(methods)):
            method = methods[i]
            ax.bar(x + i * bar_width, data[i], width=bar_width, color=colors[i], alpha=0.8, edgecolor='black', hatch=hatches[method], label=methods[i])

        ax.set_ylim(0, 59)
        ax.yaxis.set_major_locator(MultipleLocator(10))

        # 设置标签和刻度
        center_offset = (len(methods) - 1) * bar_width / 2
        ax.set_xticks(x + center_offset)
        ax.set_xticklabels(labels, fontsize=labelsize)
        ax.set_ylabel("Average Bubble Ratio (%)", fontsize=labelsize)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
        
        handle, label = ax.get_legend_handles_labels()
        # handle[-3], handle[-1] = handle[-1], handle[-3]
        # label[-3], label[-1] = label[-1], label[-3]
        # ax.set_title("Average Bubble Ratio on Different Models", fontsize=titlesize, y=1.15)
        
        # fig.legend(
        #     handles=handle,
        #     labels=label,
        #     fontsize=legendsize,
        #     loc='upper center',
        #     bbox_to_anchor=(0.54, 0.92),
        #     ncol=len(methods),
        #     frameon=False,
        #     handlelength=1.5,
        #     columnspacing=0.75,
        #     handletextpad=0.25,
        # )
        fig.legend(
            handles=handle,
            labels=label,
            fontsize=legendsize,
            loc='upper center',
            bbox_to_anchor=(0.54, 1.075),
            ncol=len(methods),
            frameon=False,
            handlelength=1.5,
            columnspacing=0.75,
            handletextpad=0.25,
        )
        # 显示百分比文字在柱子上方
        for i in range(len(methods)):
            for j in range(len(labels)):
                val = data[i][j]
                ax.text(x[j] + i * bar_width * 1.025, val+0.5, f"{val:.1f}", ha='center', rotation=90, va='bottom', fontsize=ticksize)
        
    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/bubble_comparison.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()


bubble_comparison()
