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
            "56B LLaMA": [ "pp4 tp4 dp2 mb16", "pp4 tp8 dp1 mb16", "pp8 tp4 dp2 mb32"],
            "70B LLaMA": ["pp4 tp4 dp4 mb16", "pp4 tp8 dp2 mb16", "pp8 tp4 dp2 mb32"]
        }
        bar_width = 0.3

        # 遍历每个模型和配置绘制子图
        for row_idx, model in enumerate(["56B LLaMA", "70B LLaMA"]):
            configs = model_configs[model]
            for col_idx, config in enumerate(configs):
                ax = axes[row_idx, col_idx]
                config_data = comp_data[model][config]["homo"]
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
            ("70B LLaMA", "pp4 tp4 dp4 mb16"),
            ("70B LLaMA", "pp4 tp8 dp2 mb16"),
            ("70B LLaMA", "pp8 tp4 dp2 mb32"),
            ("56B LLaMA", "pp4 tp4 dp2 mb16"),
            ("56B LLaMA", "pp4 tp8 dp1 mb16"),
            ("56B LLaMA", "pp8 tp4 dp2 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度

        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = comp_data[model][config]["homo"]
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
            # ("70B LLaMA", "pp4 tp4 dp4 mb16"),
            # ("70B LLaMA", "pp4 tp8 dp2 mb16"),
            # ("70B LLaMA", "pp8 tp4 dp2 mb32"),
            ("56B LLaMA", "pp4 tp4 dp2 mb16"),
            ("56B LLaMA", "pp4 tp8 dp1 mb16"),
            ("56B LLaMA", "pp8 tp4 dp2 mb32"),
            ("56B LLaMA", "pp4 tp4 dp2 mb16"),
            ("56B LLaMA", "pp4 tp8 dp1 mb16"),
            ("56B LLaMA", "pp8 tp4 dp2 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度

        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = comp_data[model][config]["heter"]
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

def drawhomo1row6col():
    def sinplot():
        # 设置全局字体和样式
        plt.rcParams.update({'font.size': 8, 'axes.titlesize': 10, 'axes.labelsize': 8})

        # 创建1行6列的子图布局
        fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(12, 2))  # 宽度加大到24，高度减小到3
        plt.subplots_adjust(wspace=0.4)  # 只调整水平间距

        # 定义所有6个配置（合并56B和70B的配置）
        all_configs = [
            ("70B LLaMA", "pp4 tp4 dp2 mb16"),
            ("70B LLaMA", "pp4 tp8 dp1 mb16"),
            ("70B LLaMA", "pp8 tp4 dp1 mb32"),
            ("42B LLaMA", "pp4 tp4 dp2 mb16"),
            ("42B LLaMA", "pp4 tp8 dp1 mb16"),
            ("42B LLaMA", "pp8 tp4 dp1 mb32"),
            ("42B LLaMA", "pp4 tp4 dp1 mb16"),
            ("42B LLaMA", "pp8 tp2 dp1 mb32"),
            ("14B LLaMA", "pp4 tp2 dp1 mb16"),
            ("14B LLaMA", "pp8 tp1 dp1 mb32"),
        ]

        bar_width = 0.5  # 调宽柱子宽度
        temp_labels = list(labels.keys())
        temp_labels = temp_labels[0:1]+temp_labels[2:]
        # 遍历6个配置绘制子图
        for col_idx, (model, config) in enumerate(all_configs):
            ax = axes[col_idx]
            config_data = data_h800[model][config]["homo"]
            methods = list(config_data.keys())
            methods = methods[0:1]+methods[2:]
            values = list(config_data.values())
            values = values[0:1]+values[2:]
            colors_list = [colors[method] for method in methods]

            # 绘制柱状图
            init_values = values
            values = [v if v!="OOM" else 0 for v in values]
            bars = ax.bar(methods, values, bar_width, color=colors_list, edgecolor='grey')
            
            # 添加数值标签（调整字体更小）
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.0,
                    f"{height}" if height != 0 else "OOM",
                    ha='center',
                    va='bottom',
                    fontsize=7  # 缩小字体
                )

            # 设置子图标签
            ax.set_xlabel(f"{model}\n{config}", fontsize=10, labelpad=5)  # 换行显示模型和配置
            if col_idx == 0:
                ax.set_ylabel("Tokens/GPU/second", fontsize=11)
            ax.set_ylim(min(values)*0.98, max(values) * 1.0)  # 统一Y轴范围
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度

        # 添加全局图例（调整位置到图像下方）
        legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in temp_labels]
        fig.legend(
            legend_labels,
            temp_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0275),  # 下移图例
            ncol=len(temp_labels),
            frameon=False,
            fontsize=8
        )

        # 调整布局并显示
        plt.tight_layout()

    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/e2e performance h800.pdf", format='pdf', dpi=200, bbox_inches="tight")
    plt.show()

def drawhomo2row5col(arch="homo"):
    text_size = 9
    def sinplot():
        # 设置全局字体和样式
        plt.rcParams.update({
            'font.size': 7,
            'axes.titlesize': 8,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 7.5
        })

        # 创建2行5列的子图布局
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 3))  # 调整画布尺寸
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # 同时调整水平和垂直间距

        # 定义所有配置（共10个）
        all_configs = [
            ("70B LLaMA", "pp4 tp4 dp2 mb16"),
            ("70B LLaMA", "pp4 tp8 dp1 mb16"),
            ("70B LLaMA", "pp8 tp4 dp1 mb32"),
            ("42B LLaMA", "pp4 tp4 dp2 mb16"),
            ("42B LLaMA", "pp4 tp8 dp1 mb16"),
            ("42B LLaMA", "pp8 tp4 dp1 mb32"),
            ("42B LLaMA", "pp4 tp4 dp1 mb16"),
            ("42B LLaMA", "pp8 tp2 dp1 mb32"),
            ("14B LLaMA", "pp4 tp2 dp1 mb16"),
            ("14B LLaMA", "pp8 tp1 dp1 mb32"),
        ]

        bar_width = 0.6  # 柱宽
        temp_labels = list(reversed(list(labels.keys())))
        max_statistics = [0,0,0,0,0]
        statistics = [0,0,0,0,0,0]
        counts = [0,0,0,0,0,0]
        for i, (model, config) in enumerate(all_configs):
            row = i // 5  # 计算行索引
            col = i % 5   # 计算列索引
            ax = axes[row, col]
            config_data = data_h800[model][config][arch]
            methods = list(reversed(list(config_data.keys())))
            values = list(reversed(list(config_data.values())))
            print(values)
            if arch == "heter":
                values = [1/v if v else v for v in values]
            speedups = [round(v/values[0],2) for v in values]
            print(speedups)
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
            min_v = values[0]

            bottom_ratio = 0.98
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.0 if height >=min_v else min_v*bottom_ratio,
                    f"{height/standard:.2f}X" if height != 0 else "OOM",
                    ha='center',
                    va='bottom',
                    rotation=0,
                    fontsize=text_size  # 缩小字体
                )
            ax.set_xlabel(f"{model[:3]} {config[:-5].upper()}", fontsize=10, labelpad=5)  # 换行显示模型和配置
            if col == 0:
                ax.set_ylabel("TGS", fontsize=11)
            ax.set_ylim(min_v*bottom_ratio, max(values) * 1.01)  # 统一Y轴范围
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            if arch == "heter":
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度

            handles, ls = ax.get_legend_handles_labels()

        # legend_labels = [plt.Rectangle((0,0),1,1, hatch=hatches[method], color=colors[method], alpha=1) for method in temp_labels]
        fig.legend(
            handles,
            ls,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.025),  # 下移图例
            ncol=len(temp_labels),
            frameon=False,
            fontsize=12
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
    plt.savefig(f"/Users/hanayukino/e2e_performance_h800_2x5_{arch}.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

drawhomo2row5col(arch="homo")