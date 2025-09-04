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
        methods = ['S-1F1B', 'I-1F1B', 'ZB', 'Mist']
        
        data = np.array([
        # small ['Ideal', 'Gemma', 'DeepSeek', 'Nemotron-H', 'OctoPipe']
            [17.97,   28.35, 34.3025, 31.135],
            [2.7,    20.4275,   27.60,   30.2575],
            [11.52,   26.7825,  33.355,  30.21],
            [6.82,   18.2,   22.315,   19.7475],
            # [0.93, 5.9375, 7.175, 9.71],
        ])   
        
        data = np.array([
        # medium ['Ideal', 'Gemma', 'DeepSeek', 'Nemotron-H', 'OctoPipe']
            [17.97,   36.211, 44.7475, 47.86375],
            [2.7,    30.3287,   39.4674,   51.13375],
            [11.52,   34.8662,  43.845,  47.19],
            [6.82,   20.7325,   25.57125,   41.48625],
            # [0.93, 5.9375, 7.175, 9.71],
        ])   
        
        # X轴标签（stage）
        labels = ['LLaMA-2', 'Gemma', 'DeepSeek', 'Nemotron-H']

        x = np.arange(len(labels))  # [0, 1, 2, 3, 4]
        bar_width = 0.2

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 5))

        # 每种方法一个 bar group
        for i in range(len(methods)):
            method = methods[i]
            ax.bar(x + i * bar_width, data[i], width=bar_width-0.00, color=colors[method], alpha=0.8, edgecolor='black', hatch=hatches[method], label=methods[i])

        ax.set_ylim(0, 62)
        ax.yaxis.set_major_locator(MultipleLocator(10))

        # 设置标签和刻度
        center_offset = (len(methods) - 1) * bar_width / 2
        ax.set_xticks(x + center_offset)
        ax.set_xticklabels(labels, fontsize=labelsize)
        ax.set_ylabel("Bubble Ratio (%)", fontsize=labelsize)
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
            bbox_to_anchor=(0.54, 1.095),
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
