import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from e2e_data import *
# 数据预处理
methods = ["UnifiedPP", "Interleaved-L", "Interleaved", "ZBV", "ZBH", "1F1B"]
methods = ["UnifiedPP", "Interleaved", "ZBV", "ZBH", "1F1B"]
x = np.arange(8)
width = 0.6

text_offset = 3
text_font_size = 12
label_size = 12
# 处理OOM数据
def process_data(data):
    return [float(v) if v != "OOM" else 0 for v in data]

def sinplot():

    # 创建子图
    fig, axs = plt.subplots(6, 1, figsize=(12, 9), dpi=250)
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
    order = ["UnifiedPP", "Interleaved", "ZBV", "ZBH", "1F1B"]
    # 为每个配置绘制图表
    for model_idx, (model, configs) in enumerate(configs_to_compare.items()):
        for cfg_idx, (data_cfg, sim_cfg) in enumerate(configs):
            ax = axs[0]
            
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
                    fontsize=text_font_size
                )

            ax.plot(x, sim_values, '--', lw=1, marker='o', markersize=3, label='Simulated', color='#d7191c')
            
            # 装饰图表
            ax.set_xticks(x)
            # ax.set_xlabel(f"{model} {data_cfg}", fontsize=label_size)
            ax.tick_params(axis='x', which='both', labelbottom=False)  # 隐藏X轴文字（labelbottom）
            ax.set_ylabel(f"Throughput", fontsize=label_size)
            # ax.set_xticklabels([labels[o] for o in order], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.75, 1.25)
            
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
        bars = ax.bar(x, pro_values, width, color=bar_color, alpha=0.8, label='Profile')
        
        # 绘制折线图（Simulator）
        line, = ax.plot(x, sim_values, color=line_color, marker='o', markersize=3, 
                        linewidth=1, linestyle='--', label='Simulator')
        
        # 添加数据标签
        for i, v in enumerate(pro_values):
            if v != 0:
                diff = (sim_data[i] - v)/80*100
                if diff > 0:
                    ax.text(i, v+text_offset, f'+{diff:.2f}%', ha='center', fontsize=text_font_size)
                else:
                    ax.text(i, v+text_offset, f'{diff:.2f}%', ha='center', fontsize=text_font_size)
            else:
                if i == 0:
                    ax.text(i, sim_data[i]-text_offset*3, f'OOM,{sim_data[i]:.2f}', ha='center', fontsize=text_font_size)
                else:
                    ax.text(i, sim_data[i]+text_offset, f'OOM,{sim_data[i]:.2f}', ha='center', fontsize=text_font_size)

        # 装饰子图
        # ax.set_title(f'{method}', fontsize=12, pad=10)
        ax.set_ylabel('Memory (GB)', fontsize=label_size)
        ax.set_xticks(x)
        if idx == 4:
            ax.set_xticklabels([f'Rank {i+1}' for i in range(8)], rotation=0, fontsize=label_size)
        else:
            # ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度（bottom）和文字（labelbottom）
            ax.tick_params(axis='x', which='both', labelbottom=False)  # 隐藏X轴文字（labelbottom）
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        # ax.legend(loc='upper right')
        
        # 统一y轴范围
        all_values = [v for v in pro_values + sim_values if not np.isnan(v)]
        ax.set_ylim(25, 81)
    
    from matplotlib.lines import Line2D  # 导入 Line2D

    # 原始矩形图例
    legend_labels = [plt.Rectangle((0,0),1,1, color=colors[method]) for method in labels]
    # 创建红色折线图例句柄（自定义线型/颜色/宽度）
    red_line = Line2D([0], [0], color='red', lw=1, linestyle='--')  # 实线样式
    legend_labels = [red_line] + legend_labels
    # 合并标签（原始标签 + 新标签）
    all_labels =  ["Simulator"] + labels # 替换成你的实际名称
    
    fig.legend(
        legend_labels,
        all_labels,  # 使用合并后的标签
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=len(all_labels),
        frameon=False,
        fontsize=12
    )
    plt.tight_layout()
sinplot()
sns.despine()
plt.savefig("/Users/hanayukino/sim_thro_mem.pdf", format='pdf', dpi=200, bbox_inches="tight")
plt.show()