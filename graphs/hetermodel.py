import numpy as np
import matplotlib.pyplot as plt
from e2e_data import model_heter_4_stage
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
plt.rcParams['font.family'] = 'Helvetica'

fontsize=24
titlesize=24
labelsize=20
ticksize=18
legendsize=14
def heter_model_comp():
    def sinplot():
        all = plt.figure(figsize=(18, 7.5))
        bar_width = 0.4
        index = np.arange(4)  # 4个stage
        colors = ['#4874CB', '#E54C5E']  # 2K蓝色，4K红色
        # plt.suptitle("Stage Computation Time on Different Models", y=0.98, fontsize=fontsize + titlesize)

        global_max = 0
        for model in model_heter_4_stage.keys():
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            fwd_4k = model_heter_4_stage[model]["4K"]["fwd"]
            bwd_4k = model_heter_4_stage[model]["4K"]["bwd"]
            current_max = max(max([fwd + bwd for fwd, bwd in zip(fwd_2k, bwd_2k)]), max([fwd + bwd for fwd, bwd in zip(fwd_4k, bwd_4k)]))
            global_max = max(global_max, current_max)
            print(global_max)
        
        # 设置统一的y轴上限（增加10%余量）
        y_upper = global_max * 1.025 * 1.25

        for model_idx, model in enumerate(model_heter_4_stage.keys()):
            ax = plt.subplot(1, 3, model_idx + 1)
            
            # 获取数据
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            total_2k = [fwd + bwd for fwd, bwd in zip(fwd_2k, bwd_2k)]
            standard = total_2k[0]
            total_2k = [t / standard for t in total_2k]
            
            fwd_4k = model_heter_4_stage[model]["4K"]["fwd"]
            bwd_4k = model_heter_4_stage[model]["4K"]["bwd"]
            total_4k = [fwd + bwd for fwd, bwd in zip(fwd_4k, bwd_4k)]
            standard = total_4k[0]
            total_4k = [t / standard for t in total_4k]
            
            # 绘制分组柱状图
            bar1 = ax.bar(index - bar_width/2, total_2k, bar_width, 
                        color=colors[0], edgecolor='black', label='2K Context')
            bar2 = ax.bar(index + bar_width/2, total_4k, bar_width,
                        color=colors[1], hatch='//', edgecolor='black', label='4K Context')
            
            x_offset = 0.025
            offset = 1.025
            # if model_idx == 0:
            #     offset = 0.5
            # elif model_idx == 1:
            #     offset = 0.4
            # else:
            #     offset = 0.3
            standard = 1
            for bar in bar1:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + x_offset,
                    height * 1.0 * offset,
                    f"{height/standard:.2f}x",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color="#1E3F8C",# #1E3F8C #2F57A8
                    # color="#FFFFFF",
                    fontsize=fontsize+ticksize  # 缩小字体
                )
            
            for bar in bar2:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + x_offset,
                    height * 1.0 * offset,
                    f"{height/standard:.2f}x",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color="#B03547", # #8A1F30 #B03547
                    # color="#FFFFFF",
                    fontsize=fontsize+ticksize  # 缩小字体
                )

            # 设置轴标签和刻度
            ax.set_title(model, fontsize=fontsize+labelsize, y=1.02)
            # ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_ylim(0, max(max(total_4k), max(total_2k)) * 1.4)
            # ax.set_ylim(0.8, 3)

            ax.set_xticks(index)
            # ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], fontsize=fontsize)
            ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=fontsize+labelsize)
            
            # ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            # y_major_locator=MultipleLocator(0.05)
            # ax.yaxis.set_major_locator(y_major_locator)

            # 仅第一个子图显示y轴标签
            if model_idx == 0:
                ax.set_ylabel("Time (Normalized)", fontsize=fontsize + labelsize)
                # ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+ticksize)
                ax.tick_params(labelleft=False, labelright=False)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', length=0, labelsize=0)

            # ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+6)
            
            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.25)
            
            # 添加图例（只加一次）
            if model_idx == 1:
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0.025, 0.999),  # 下移图例
                    labelspacing=0.25,   # 控制行间距（可调）
                    columnspacing=0.25,  # 控制列间距（可调）
                    handletextpad=0.25,  # 控制图例中图标与文字的间距
                    borderaxespad=0.01,  # 控制图例与图像边缘的距离
                    handlelength=1.5,      # 控制图例中图标的长度
                    fontsize=fontsize+legendsize
                )    

    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/hetermodelcomp.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

def heter_model_mem():
    def sinplot():
        plt.figure(figsize=(18, 6))
        bar_width = 0.35
        index = np.arange(4)  # 4个stage
        colors = ['#4874CB', '#E54C5E']  # 2K蓝色，4K红色
        plt.suptitle("Stage Memory on Different Models", y=1.0, fontsize=40)

        global_max = 0
        for model in model_heter_4_stage.keys():
            mem_2k = model_heter_4_stage[model]["2K"]["mem"]
            mem_4k = model_heter_4_stage[model]["4K"]["mem"]
            current_max = max(max(mem_2k), max(mem_4k))
            global_max = max(global_max, current_max)
            print(global_max)
        
        # 设置统一的y轴上限（增加10%余量）
        y_upper = global_max * 1.025

        for model_idx, model in enumerate(model_heter_4_stage.keys()):
            ax = plt.subplot(1, 3, model_idx + 1)
            
            # 获取数据
            fwd_2k = model_heter_4_stage[model]["2K"]["mem"]
            total_2k = fwd_2k
            
            fwd_4k = model_heter_4_stage[model]["4K"]["mem"]
            total_4k = fwd_4k
            
            # 绘制分组柱状图
            bar1 = ax.bar(index - bar_width/2, total_2k, bar_width, 
                        color=colors[0], edgecolor='black', label='2K Context')
            bar2 = ax.bar(index + bar_width/2, total_4k, bar_width,
                        color=colors[1], edgecolor='black', label='4K Context')
            
            # 设置轴标签和刻度
            ax.set_title(model, fontsize=fontsize+14)
            # ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_ylim(0, y_upper)

            ax.set_xticks(index)
            # ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], fontsize=fontsize)
            ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=fontsize+14)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            y_major_locator=MultipleLocator(10)
            ax.yaxis.set_major_locator(y_major_locator)

            # 仅第一个子图显示y轴标签
            if model_idx == 0:
                ax.set_ylabel("Memory (GB)", fontsize=fontsize + 14)
                ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+8)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', length=0, labelsize=0)

            # ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+6)
            
            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加图例（只加一次）
            if model_idx == 0:
                ax.legend(loc="upper left", fontsize=fontsize+10)

    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/hetermodelmem.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()


def heter_model_compmem():
    def sinplot():
        all = plt.figure(figsize=(18, 12))
        bar_width = 0.4
        index = np.arange(4)  # 4个stage
        colors = ['#4874CB', '#E54C5E']  # 2K蓝色，4K红色
        plt.suptitle("Stage Computation/Memory Overhead on Different Models", y=1.0, fontsize=fontsize + titlesize)

        global_max = 0
        for model in model_heter_4_stage.keys():
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            fwd_4k = model_heter_4_stage[model]["4K"]["fwd"]
            bwd_4k = model_heter_4_stage[model]["4K"]["bwd"]
            current_max = max(max([fwd + bwd for fwd, bwd in zip(fwd_2k, bwd_2k)]), max([fwd + bwd for fwd, bwd in zip(fwd_4k, bwd_4k)]))
            global_max = max(global_max, current_max)
            print(global_max)
        
        # 设置统一的y轴上限（增加10%余量）
        y_upper = global_max * 1.025

        for model_idx, model in enumerate(model_heter_4_stage.keys()):
            ax = plt.subplot(2, 3, model_idx + 1)
            
            # 获取数据
            fwd_2k = model_heter_4_stage[model]["2K"]["fwd"]
            bwd_2k = model_heter_4_stage[model]["2K"]["bwd"]
            total_2k = [fwd + bwd for fwd, bwd in zip(fwd_2k, bwd_2k)]
            
            fwd_4k = model_heter_4_stage[model]["4K"]["fwd"]
            bwd_4k = model_heter_4_stage[model]["4K"]["bwd"]
            total_4k = [fwd + bwd for fwd, bwd in zip(fwd_4k, bwd_4k)]
            
            # 绘制分组柱状图
            bar1 = ax.bar(index - bar_width/2, total_2k, bar_width, 
                        color=colors[0], edgecolor='black', label='2K Context')
            bar2 = ax.bar(index + bar_width/2, total_4k, bar_width,
                        color=colors[1], hatch='//', edgecolor='black', label='4K Context')
            standard = total_2k[0]
            for bar in bar1:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.0 + 0.01,
                    f"{height/standard:.2f}x",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color="#4874CB",
                    fontsize=fontsize+ticksize/2  # 缩小字体
                )
            
            standard = total_4k[0]
            for bar in bar2:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.0 + 0.01,
                    f"{height/standard:.2f}x",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    color="#E54C5E",
                    fontsize=fontsize+ticksize/2  # 缩小字体
                )

            # 设置轴标签和刻度
            ax.set_title(model, fontsize=fontsize+14)
            # ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_ylim(0, y_upper*1.25)

            ax.set_xticks(index)
            # ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], fontsize=fontsize)
            ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=fontsize+labelsize)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
            
            y_major_locator=MultipleLocator(0.05)
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
            if model_idx == 2:
                ax.legend(loc="upper left", fontsize=fontsize+legendsize)
        
        global_max = 0
        for model in model_heter_4_stage.keys():
            mem_2k = model_heter_4_stage[model]["2K"]["mem"]
            mem_4k = model_heter_4_stage[model]["4K"]["mem"]
            current_max = max(max(mem_2k), max(mem_4k))
            global_max = max(global_max, current_max)
            print(global_max)
        
        # 设置统一的y轴上限（增加10%余量）
        y_upper = global_max * 1.025

        for model_idx, model in enumerate(model_heter_4_stage.keys()):
            ax = plt.subplot(2, 3, 3 + model_idx + 1)
            
            # 获取数据
            fwd_2k = model_heter_4_stage[model]["2K"]["mem"]
            total_2k = fwd_2k
            
            fwd_4k = model_heter_4_stage[model]["4K"]["mem"]
            total_4k = fwd_4k
            
            # 绘制分组柱状图
            bar1 = ax.bar(index - bar_width/2, total_2k, bar_width, 
                        color=colors[0], edgecolor='black', label='2K Context')
            bar2 = ax.bar(index + bar_width/2, total_4k, bar_width,
                        color=colors[1], hatch='//', edgecolor='black', label='4K Context')
            
            # 设置轴标签和刻度
            ax.set_title(model, fontsize=fontsize+14)
            # ax.tick_params(axis='both', which='both', length=0, labelsize=0)
            ax.set_ylim(0, y_upper)

            ax.set_xticks(index)
            # ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'], fontsize=fontsize)
            ax.set_xticklabels(['S1', 'S2', 'S3', 'S4'], fontsize=fontsize+labelsize)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            
            y_major_locator=MultipleLocator(10)
            ax.yaxis.set_major_locator(y_major_locator)

            # 仅第一个子图显示y轴标签
            if model_idx == 0:
                ax.set_ylabel("Memory (GB)", fontsize=fontsize + labelsize)
                ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+ticksize)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', length=0, labelsize=0)

            # ax.tick_params(axis='y', which='both', length=5, labelsize=fontsize+6)
            
            # 添加网格线
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加图例（只加一次）
            if model_idx == 2:
                ax.legend(loc="upper left", fontsize=fontsize+legendsize)

    sinplot()
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/Users/hanayukino/hetermodelcompmem.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

heter_model_comp()