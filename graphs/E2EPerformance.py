import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *

def drawhomo2row3col():
    def sinplot():
        # 设置全局字体和样式
        plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10})

        # 创建2行3列的子图布局
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # 定义每个模型的配置列表（选择每个模型的前3个配置）
        model_configs = {
            "56B Llama": [ "pp4 tp4 dp2 mb16", "pp4 tp8 dp1 mb16", "pp8 tp4 dp2 mb32"],
            "70B Llama": ["pp4 tp4 dp4 mb16", "pp4 tp8 dp2 mb16", "pp8 tp4 dp2 mb32"]
        }
        bar_width = 0.3

        # 遍历每个模型和配置绘制子图
        for row_idx, model in enumerate(["56B Llama", "70B Llama"]):
            configs = model_configs[model]
            for col_idx, config in enumerate(configs):
                ax = axes[row_idx, col_idx]
                config_data = data[model][config]["homo"]
                methods = list(config_data.keys())
                values = list(config_data.values())
                colors_list = [colors[method] for method in methods]

                # 绘制柱状图
                bars = ax.bar(methods, values, bar_width, color=colors_list)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height * 1.0,
                        f"{height}",
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )

                # 设置子图标题和标签
                # ax.set_title(f"{model}\n{config}", pad=12)
                ax.set_xlabel(f"{model} {config}")
                ax.set_ylabel("Tokens/GPU/second")
                ax.set_ylim(250, max(values) * 1.05)  # 统一Y轴范围
                ax.grid(axis='y', linestyle='--', alpha=0.6)
                # ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='x', rotation=45)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # 添加全局图例
        legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in labels]
        fig.legend(
            legend_labels,
            labels.keys(),
            # title="Methods",
            loc="upper center",
            # bbox_to_anchor=(0.5, 1.0175),
            ncol=6,
            frameon=False
        )
        # 调整布局并显示
        plt.tight_layout()

    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/e2e performance.pdf", format='pdf', dpi=200)

    plt.show()

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
            ("56B Llama", "pp8 tp4 dp2 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度

        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = data[model][config]["homo"]
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
                ax.set_ylabel("Tokens/GPU/second", fontsize=12)
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
    plt.savefig("/Users/hanayukino/e2e performance.pdf", format='pdf', dpi=200, bbox_inches="tight")
    plt.show()

def drawheter1row6col():
    def sinplot():
        # 设置全局字体和样式
        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8})

        # 创建1行6列的子图布局
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(12, 3))  # 宽度加大到24，高度减小到3
        plt.subplots_adjust(wspace=0.4)  # 只调整水平间距

        # 定义所有6个配置（合并56B和70B的配置）
        all_configs = [
            # ("70B Llama", "pp4 tp4 dp4 mb16"),
            # ("70B Llama", "pp4 tp8 dp2 mb16"),
            # ("70B Llama", "pp8 tp4 dp2 mb32"),
            ("56B Llama", "pp4 tp4 dp2 mb16"),
            ("56B Llama", "pp4 tp8 dp1 mb16"),
            ("56B Llama", "pp8 tp4 dp2 mb32"),
            ("56B Llama", "pp4 tp4 dp2 mb16"),
            ("56B Llama", "pp4 tp8 dp1 mb16"),
            ("56B Llama", "pp8 tp4 dp2 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度

        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = data[model][config]["heter"]
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
                ax.set_ylabel("Tokens/GPU/second", fontsize=12)
            if values:
                ax.set_ylim(0, max(values) * 1.0)  # 统一Y轴范围
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
    plt.savefig("/Users/hanayukino/e2e heter performance.pdf", format='pdf', dpi=200, bbox_inches="tight")
    plt.show()

drawheter1row6col()