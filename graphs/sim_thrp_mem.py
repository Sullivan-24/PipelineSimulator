import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from e2e_data import *
# 数据预处理
methods = ["OctoPipe", "Interleaved-L", "Interleaved", "ZBV", "ZBH", "Dapple"]
methods = ["OctoPipe", "Interleaved", "ZBV", "ZBH", "Dapple"]
methods = ["OctoPipe", "Interleaved", "ZBH", "Dapple"]
x = np.arange(8)
width = 0.6

text_offset = 3
text_font_size = 24
label_size = 24
xy_tick_size = 22
# 处理OOM数据
def process_data(data):
    return [float(v) if v != "OOM" else 0 for v in data]
ylabels = ['(a)', '(b)', '(c)', '(d)', '(e)']

def sinplot():

    # 创建子图
    fig, axs = plt.subplots(len(methods) + 1, 1, figsize=(12, 10), dpi=250)
    for i, ax in enumerate(axs):
        ax.text(1.02, 0.5, ylabels[i], transform=ax.transAxes,
                fontsize=28, va='center', ha='left')
    plt.subplots_adjust(hspace=0.10)
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
    order = ["OctoPipe", "Interleaved", "ZBV", "ZBH", "Dapple"]
    order = ["OctoPipe", "Interleaved", "ZBH", "Dapple"]
    # 为每个配置绘制图表
    for model_idx, (model, configs) in enumerate(configs_to_compare.items()):
        for cfg_idx, (data_cfg, sim_cfg) in enumerate(configs):
            ax = axs[0]
            
            # 获取原始数据
            data_homo = data_h800[model][data_cfg]["homo"]
            sim_homo = sim_h800[model][sim_cfg]["homo"]
            sim_homo = {k: 1/v for k, v in sim_homo.items()}

            # 计算归一化值
            data_base = data_homo["Dapple"]
            sim_base = sim_homo["Dapple"]
            
            
            data_norm = {k: v/data_base for k, v in data_homo.items()}
            sim_norm = {k: v/sim_base for k, v in sim_homo.items()}
            
            # 按顺序准备数据
            x = np.arange(len(order))
            data_values = [data_norm[k] for k in order]
            sim_values = [sim_norm[k] for k in order]
            color = [colors[o] for o in order]
            hatch_list = [hatches[k] for k in order]
            label_list = [k for k in order]
            # 绘制折线
            # ax.bar(x, data_values, lw=1, label='Measured', color='#2c7bb6')
            bars = ax.bar(x, data_values, label=label_list, edgecolor='black', hatch=hatch_list, color=color)

            for i, bar in enumerate(bars):
                height = bar.get_height()
                diff_ratio = (data_values[i] - sim_values[i])
                if data_values[i] != 0:
                    diff_ratio /= data_values[i]
                diff = diff_ratio
                diff_text = f"{diff*100:.2f}%" if diff > 0 else f"-{abs(diff)*100:.2f}%"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.00,
                    diff_text,
                    ha='center',
                    va='bottom',
                    fontsize=text_font_size,
                )

            ax.plot(x, sim_values, '--', lw=3, marker='o', markersize=6, label='Simulator', color='#d7191c')
            
            # 装饰图表
            ax.set_xticks(x)
            # ax.set_xlabel(f"{model} {data_cfg}", fontsize=label_size)
            ax.tick_params(axis='x', which='both', labelbottom=False)  # 隐藏X轴文字（labelbottom）
            ax.set_ylabel(f"Thr.", fontsize=label_size)
            # ax.set_xticklabels([labels[o] for o in order], rotation=45, ha='right')
            ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.8, 1.2)
            handles, ls = ax.get_legend_handles_labels()
            
    labels = []
    x = np.arange(8)
    for idx, method in enumerate(methods):
        ax = axs[idx + 1]
        # 提取数据
        pro_data = mem_pro_h800["14B LLaMA"]["pp8 tp1 dp1 mb32"]["homo"][method]
        sim_data = mem_sim_h800["14B LLaMA"]["pp8 tp1 dp1 mb32"]["homo"][method]
        labels.append(method)
        # 转换数据格式
        pro_values = process_data(pro_data)
        sim_values = process_data(sim_data)
        # 颜色配置
        bar_color = colors[method]
        line_color = 'red'

        # 绘制柱状图（Profile）
        bars = ax.bar(x, pro_values, width, color=bar_color, alpha=0.8, edgecolor='black', hatch=hatches[method], label='Profile')
        
        # 绘制折线图（Simulator）
        line, = ax.plot(x, sim_values, marker='o', markersize=6, 
                        linewidth=3, linestyle='--', color='#d7191c', label='Simulator')
        
        # 添加数据标签
        for i, v in enumerate(pro_values):
            if v != 0:
                diff = (sim_data[i] - v)/80*100
                if diff > 0:
                    ax.text(i, v+text_offset, f'{diff:.2f}%', ha='center', fontsize=text_font_size)
                else:
                    ax.text(i, v+text_offset, f'{diff:.2f}%', ha='center', fontsize=text_font_size)
            else:
                if i == 0:
                    ax.text(i, sim_data[i]-text_offset*7, f'OOM,\n{sim_data[i]:.2f}', ha='center', fontsize=text_font_size-2)
                else:
                    ax.text(i, sim_data[i]+text_offset, f'OOM,\n{sim_data[i]:.2f}', ha='center', fontsize=text_font_size-2)

        # 装饰子图
        # ax.set_title(f'{method}', fontsize=12, pad=10)
        ax.set_ylabel('Mem.', fontsize=label_size)
        ax.set_xticks(x)
        if idx == len(methods)-1:
            ax.set_xticklabels([f'GPU {i+1}' for i in range(8)], rotation=0, fontsize=label_size)
        else:
            # ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度（bottom）和文字（labelbottom）
            ax.tick_params(axis='x', which='both', labelbottom=False)  # 隐藏X轴文字（labelbottom）
        ax.grid(True, axis='y', linestyle='--', markersize=6, alpha=0.7)
        # ax.legend(loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
        
        # 统一y轴范围
        all_values = [v for v in pro_values + sim_values if not np.isnan(v)]
        ax.set_ylim(25, 80)
    
    from matplotlib.lines import Line2D  # 导入 Line2D

    # 原始矩形图例
    legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in labels]
    # 创建红色折线图例句柄（自定义线型/颜色/宽度）
    red_line = Line2D([0], [0], color='red', lw=1, linestyle='--')  # 实线样式
    legend_labels = [red_line] + legend_labels
    # 合并标签（原始标签 + 新标签）
    all_labels =  ["Simulator"] + labels # 替换成你的实际名称
    
    fig.legend(
        handles,
        ls,
        # legend_labels,
        # all_labels,  # 使用合并后的标签
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        frameon=False,
        prop={'size': text_font_size-1},
        # handlelength=1.5,     # 控制图例前的线条长度
        handletextpad=0.25,    # 控制图例图形和文字之间的间距
        columnspacing=0.8     # 控制每列之间的间距
    )
    plt.tight_layout()

sinplot()
sns.despine()
plt.savefig("/Users/hanayukino/sim_thro_mem.pdf", format='pdf', dpi=200, bbox_inches="tight")
plt.show()