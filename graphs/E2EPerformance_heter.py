import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *

def drawhomo2row5col(arch="heter"):
    text_size = 10
    def sinplot():
        labels = heter_labels
        # 设置全局字体和样式
        plt.rcParams.update({
            'font.size': 7,
            'axes.titlesize': 8,
            'axes.labelsize': 11,
            'xtick.labelsize': 6,
            'ytick.labelsize': 11
        })

        # 创建2行5列的子图布局
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 1.5))  # 调整画布尺寸
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 同时调整水平和垂直间距

        # 定义所有配置（共10个）
        all_configs = [
            ("70B LLaMA", "pp4 tp8 dp1 mb16"),
            ("70B LLaMA", "pp8 tp4 dp1 mb32"),
            ("42B LLaMA", "pp4 tp8 dp1 mb16"),
            ("42B LLaMA", "pp8 tp4 dp1 mb32"),
            ("14B LLaMA", "pp4 tp2 dp1 mb16"),
            # ("14B LLaMA", "pp8 tp1 dp1 mb32"),
        ]

        bar_width = 0.6  # 柱宽
        temp_labels = list(reversed(list(labels.keys())))
        max_statistics = [0,0,0,0,0]
        statistics = [0,0,0,0,0,0]
        counts = [0,0,0,0,0,0]
        for i, (model, config) in enumerate(all_configs):
            # row = 0  # 计算行索引
            # col = i % 5   # 计算列索引
            idx = i
            ax = axes[i]
            config_data = data_h800_heter[model][config]
            
            methods = list(reversed(list(config_data.keys())))
            values = list(reversed(list(config_data.values())))
            colors_list = [colors[method] for method in methods]
            hatch_list = [hatches[method] for method in methods]

            # 绘制柱状图
            init_values = values
            values = [v if v!="OOM" else 0 for v in values]
            bars = ax.bar(methods, values, bar_width, hatch=hatch_list, color=colors_list, edgecolor='black', label=methods)
            
            standard = values[0]
            upp = values[-1]
            min_v = max(values)
            max_v = max(values)
            for i, v in enumerate(values):
                if v == 0: continue
                min_v = min(v, min_v)
                statistics[i] += upp / v
                max_statistics[i] = max(max_statistics[i] , upp / v)
                counts[i] += 1

            bottom_ratio = 0.9
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.03 if height !=0 else min_v*bottom_ratio * 1.03,
                    f"{height/standard:.2f}×" if height != 0 else "OOM",
                    ha='center',
                    # va='bottom',
                    rotation=90,
                    fontsize=text_size  # 缩小字体
                )

            # if not ((row == 1 and col == 2) or (row == 0 and col == 0) or (row == 0 and col == 2) or (row == 0 and col == 5)):
            #     for bar in bars:
            #         height = bar.get_height()
            #         ax.text(
            #             bar.get_x() + bar.get_width() / 2,
            #             height * 1.0 if height !=0 else min_v*0.98,
            #             f"{max_v/height:.3f}" if height != 0 else "OOM",
            #             ha='center',
            #             va='bottom',
            #             fontsize=7  # 缩小字体
            #         )

            # 设置子图标签
            pp_tp_dp = config[:-5].upper()
            pp, tp, dp = pp_tp_dp.split(' ')
            pp = eval(pp[2:])
            tp = eval(tp[2:])
            dp = eval(dp[2:])
            ax.set_title(f"{model[:3]} ({pp},{tp},{dp})", fontsize=11)  # 换行显示模型和配置
            # ax.set_xlabel(f"{model[:3]} {config[:-5].upper()}", fontsize=10, labelpad=5)  # 换行显示模型和配置
            if idx == 0:
                ax.set_ylabel("TGS", fontsize=14)
            ax.set_ylim(min_v*bottom_ratio, max(values) * 1.15)  # 统一Y轴范围
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度
            handles, ls = ax.get_legend_handles_labels()
            plt.setp(ax.get_xticklabels())
            plt.setp(ax.get_yticklabels())
        fig.legend(
            handles,
            ls,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.35),  # 下移图例
            ncol=len(temp_labels),
            frameon=False,
            prop={'size': 12}, 
        )
        for i in range(len(max_statistics)):
            print(round(max_statistics[i], 2), end=" ")
        print()

        for i in range(len(statistics)):
            if counts[i] == 0: continue
            print(round(statistics[i]/counts[i],2), end=" ")
        print()
        # 调整布局
        plt.tight_layout()


        plt.subplots_adjust(top=0.92, bottom=0.15)  # 给图例留出空间

    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig(f"/Users/hanayukino/e2e_performance_h800_1x5_{arch}.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

drawhomo2row5col(arch="heter")