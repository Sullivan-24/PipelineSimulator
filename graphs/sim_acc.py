import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *

def drawhomo1row6col():
    def sinplot():
        # 设置全局字体和样式
        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8})

        # 创建1行6列的子图布局
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(12, 3))  # 宽度加大到24，高度减小到3
        plt.subplots_adjust(wspace=0.4)  # 只调整水平间距

        # 定义所有6个配置（合并56B和70B的配置）
        all_configs = [
            ("70B Llama", "pp4 tp4 dp4 mb16"),
            ("70B Llama", "pp4 tp8 dp2 mb16"),
            ("70B Llama", "pp8 tp4 dp2 mb32"),
            ("56B Llama", "pp4 tp4 dp2 mb16"),
            ("56B Llama", "pp4 tp8 dp1 mb16"),
            ("56B Llama", "pp8 tp4 dp1 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度

        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = sim[model][config]["homo"]
            methods = list(config_data.keys())
            values = list(config_data.values())
            colors_list = [colors[method] for method in methods]

            # 绘制柱状图
            bars = ax.bar(methods, values, bar_width, color=colors_list, edgecolor='grey')
            
            # 添加数值标签（调整字体更小）
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.0,
                    f"{height}",
                    ha='center',
                    va='bottom',
                    fontsize=7  # 缩小字体
                )

            # 设置子图标签
            ax.set_xlabel(f"{model}\n{config}", fontsize=10, labelpad=5)  # 换行显示模型和配置
            if col_idx == 0:
                ax.set_ylabel("Theoretical Execution Time", fontsize=12)
            ax.set_ylim(min(values)*0.9, max(values) * 1.0)  # 统一Y轴范围
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度

        # 添加全局图例（调整位置到图像下方）
        legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in labels]
        fig.legend(
            legend_labels,
            labels.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0275),  # 下移图例
            ncol=6,
            frameon=False,
            fontsize=8
        )

        # 调整布局并显示
        plt.tight_layout()

    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/sim performance.pdf", format='pdf', dpi=200, bbox_inches="tight")
    plt.show()

def sim_acc():
    # 配置样式
    def sinplot():
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 10,
            'legend.fontsize': 8,
        })

        # 组织可比较的配置
        configs_to_compare = {
            "14B LLaMA": [
                # ("pp4 tp4 dp2 mb16", "pp4 tp4 dp2 mb16"),
                # ("pp4 tp8 dp1 mb16", "pp4 tp8 dp1 mb16"),
                # ("pp4 tp2 dp1 mb16", "pp4 tp2 dp1 mb16"),
                ("pp8 tp1 dp1 mb32", "pp8 tp1 dp1 mb32"),
            ],
            # "42B LLaMA": [
            #     ("pp4 tp4 dp2 mb16", "pp4 tp4 dp2 mb16"),
            #     ("pp8 tp4 dp1 mb32", "pp8 tp4 dp1 mb32"),
            # ],
            # "70B LLaMA": [
            #     ("pp4 tp8 dp1 mb16", "pp4 tp8 dp1 mb16"),
            #     # ("pp4 tp8 dp2 mb16", "pp4 tp8 dp2 mb16"),
            #     ("pp8 tp4 dp1 mb32", "pp8 tp4 dp1 mb32"),
            # ],
        }

        # 创建画布
        fig, axs = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
        fig.subplots_adjust(wspace=0.3, hspace=0.4)
        order = ["UnifiedPP", "Interleaved", "ZBV", "ZBH", "1F1B"]
        # 为每个配置绘制图表
        for model_idx, (model, configs) in enumerate(configs_to_compare.items()):
            for cfg_idx, (data_cfg, sim_cfg) in enumerate(configs):
                ax = axs[model_idx]
                
                # 获取原始数据
                data_homo = data_h800[model][data_cfg]["homo"]
                sim_homo = sim_h800[model][sim_cfg]["homo"]
                sim_homo = {k: 1/v for k, v in sim_homo.items()}

                # 计算归一化值
                data_base = data_homo["1F1B"]
                sim_base = sim_homo["1F1B"]
                
                
                data_norm = {k: v/data_base for k, v in data_homo.items()}
                sim_norm = {k: v/sim_base for k, v in sim_homo.items()}
                
                # 按顺序准备数据
                x = np.arange(len(order))
                data_values = [data_norm[k] for k in order]
                sim_values = [sim_norm[k] for k in order]
                color = [colors[o] for o in order]
                # 绘制折线
                # ax.bar(x, data_values, lw=1, label='Measured', color='#2c7bb6')
                bars = ax.bar(x, data_values, label='Measured', color=color)

                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    diff_ratio = (data_values[i] - sim_values[i])
                    if data_values[i] != 0:
                        diff_ratio /= data_values[i]
                    diff = diff_ratio
                    diff_text = f"+{diff*100:.2f}%" if diff > 0 else f"-{abs(diff)*100:.2f}%"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height * 1.00,
                        diff_text,
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )

                ax.plot(x, sim_values, '--', lw=1, marker='o', markersize=3, label='Simulated', color='#d7191c')
                
                # 装饰图表
                ax.set_xticks(x)
                ax.set_xlabel(f"{model} {data_cfg}", fontsize=10)
                ax.set_ylabel(f"Throughput", fontsize=10)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                # ax.set_xticklabels([labels[o] for o in order], rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0.75, 1.25)
                
                if model_idx == 1 and cfg_idx == 2:
                    ax.legend(ncol=2, loc='lower right')

                from matplotlib.lines import Line2D  # 导入 Line2D

                # 原始矩形图例
                legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in order]

                # 创建红色折线图例句柄（自定义线型/颜色/宽度）
                red_line = Line2D([0], [0], color='red', lw=1, linestyle='--')  # 实线样式
                legend_labels = [red_line] + legend_labels
                # 合并标签（原始标签 + 新标签）
                all_labels =  ["Simulator"] + order # 替换成你的实际名称

                fig.legend(
                    legend_labels,
                    all_labels,  # 使用合并后的标签
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.03),
                    ncol=7,
                    frameon=False,
                    fontsize=9
                )
                plt.tight_layout()

    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/sim acc.pdf", format='pdf', dpi=200, bbox_inches="tight")
    plt.show()

sim_acc()