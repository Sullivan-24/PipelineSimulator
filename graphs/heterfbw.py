import numpy as np
import matplotlib.pyplot as plt
from e2e_data import model_heter_4_stage
import seaborn as sns
from matplotlib.pyplot import MultipleLocator

fontsize=20
titlesize=22
labelsize=18
ticksize=12
legendsize=12
def heter_model_fbw():
    def sinplot():
        fig = plt.figure(figsize=(18, 6))
        bar_width = 0.2
        index = np.arange(4)  # 4个stage
        colors = ['#7ddfd7', '#f5b482']  # fwd，bwd
        plt.suptitle("Stage Fwd/Bwd Overhead on Different Models", y=1.04, fontsize=fontsize + titlesize)
        handles, labels = [], []
        global_max = 0
        for model in model_heter_4_stage.keys():
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            current_max = max(max(fwd_2k), max(bwd_2k))
            global_max = max(global_max, current_max)
            print(global_max)
        
        # 设置统一的y轴上限（增加10%余量）
        y_upper = global_max * 1.025

        for model_idx, model in enumerate(model_heter_4_stage.keys()):
            ax = plt.subplot(1, 3, model_idx + 1)
            
            # 获取数据
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            fwd_4k = model_heter_4_stage[model]["4K"]["fwd"]
            bwd_4k = model_heter_4_stage[model]["4K"]["bwd"]
            
            # 绘制分组柱状图
            bar1 = ax.bar(index - bar_width*1.5, fwd_2k, bar_width, 
                        color=colors[0], edgecolor='black', label='Fwd 2K')
            bar2 = ax.bar(index - bar_width/2, bwd_2k, bar_width,
                        color=colors[1], hatch='', edgecolor='black', label='Bwd 2K')
            bar1 = ax.bar(index + bar_width/2, fwd_4k, bar_width, 
                        color=colors[0], hatch='//', edgecolor='black', label='Fwd 4K')
            bar2 = ax.bar(index + bar_width*1.5, bwd_4k, bar_width,
                        color=colors[1], hatch='//', edgecolor='black', label='Bwd 4K')
            
            # 设置轴标签和刻度
            # ax.set_title(model, fontsize=fontsize+14)
            ax.set_xlabel(model, fontsize=fontsize+14)
            # ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_ylim(0, 0.18)

            ax.set_xticks(index)
            # ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], fontsize=fontsize)
            ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=fontsize+labelsize)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            y_major_locator=MultipleLocator(0.025)
            ax.yaxis.set_major_locator(y_major_locator)

            # 仅第一个子图显示y轴标签
            if model_idx == 0:
                ax.set_ylabel("Time (seconds)", fontsize=fontsize + labelsize)
                ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+ticksize)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', length=0, labelsize=0)

            # ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+6)
            
            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加图例（只加一次）
            # if model_idx == 0:
            #     ax.legend(loc="upper left", fontsize=fontsize+legendsize)
            if len(handles) == 0:
                handle, label = ax.get_legend_handles_labels()
                handles+=handle
                labels+=label
                print(handle,label)
        
        
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00),
            ncol=4, fontsize=fontsize + legendsize, frameon=False, columnspacing=1, handletextpad=1)

    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/hetermodelfbw.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()


heter_model_fbw()