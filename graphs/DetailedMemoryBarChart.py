import sys
sys.path.append(".")
from simulator.config import *
from simulator.PainterColor import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sinplot():
    # 样式配置
    title_size = 16
    xy_label_size = 16
    xy_tick_size = 9
    text_label_size = 12
    text_padding = 2
    bar_width = 0.3

    label_fmt = lambda x: f"x{x:.2f}"  # 标签格式化函数

    # 数据准备   
    # print(f"Seq={SEQ_LEN},Hid={HIDDEN_SIZE},Voc={VOCAB_SIZE},Model={MODEL_TYPE}.")
    # mem = [StateMemory.EMB, StateMemory.LAYER, StateMemory.HEAD, Activation.FULL, Activation.LOSS, Gradient.HEAD_INPUT, Gradient.HEAD_PARA, Gradient.INPUT, Gradient.PARAMETER]
    # print(mem)
    # input("WAIT")

    labels = [ "Transformer\nlayer", "Embedding\nlayer", "Head\nlayer", "Activation","Loss\ncomputation", 
             "Input/Parameter\ngradient (Embedding)", "Input/Parameter\ngradient (Head)", "Input/Parameter\ngradient"]
    act_opt_coe = 0.2
    Llama70B = [1.507843017578125, 1.95703125, 1.95703125, 6.0625 * act_opt_coe, 3.9140625, 1.95703125, 1.95703125, 1.507843017578125]
    
    Qwen72B = [2.750030517578125, 2.3203125, 2.3203125, 6.0625 * act_opt_coe, 4.640625, 2.3203125, 2.3203125, 2.750030517578125]

    normalizedLlama70B = [v / Llama70B[0] if i < 3 else v / Llama70B[3] for i,v in enumerate(Llama70B)]
    normalizedQwen72B = [v / Qwen72B[0] if i < 3 else v / Qwen72B[3] for i,v in enumerate(Qwen72B)]
    # 分割数据集
    param_labels = labels[:3]
    runtime_labels = labels[3:]

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})
    
    # 参数显存子图
    x = np.arange(len(param_labels))
    bars1 = ax1.bar(x - bar_width/2, Llama70B[:3], bar_width, label='Llama-70B', color=set_color(0,'f',0,False), edgecolor='black')
    bars2 = ax1.bar(x + bar_width/2, Qwen72B[:3], bar_width, label='Qwen-72B', color=set_color(0,'b',0,False), edgecolor='black')
    # ax1.set_title("Model Memory", fontsize=title_size, pad=0)
    ax1.set_xlabel('Model Memory', fontsize=xy_label_size)
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_labels, rotation=0, fontsize=xy_tick_size)
    ax1.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    def format_labels(norm_values, precision=2):
        return [f"{x:.{precision}f}x" for x in norm_values]
    ax1.bar_label(bars1,
                rotation=90,
                labels=format_labels(normalizedLlama70B[:3], 2),
                fontsize=text_label_size, padding=text_padding, color='darkorange')
    ax1.bar_label(bars2,
                rotation=90,
                labels=format_labels(normalizedQwen72B[:3], 2),
                fontsize=text_label_size, padding=text_padding, color='navy')
        
    # 运行时显存子图
    x = np.arange(len(runtime_labels))
    bars3 = ax2.bar(x - bar_width/2, Llama70B[3:], bar_width, label='Llama-70B', color=set_color(0,'f',0,False), edgecolor='black')
    bars4 = ax2.bar(x + bar_width/2, Qwen72B[3:], bar_width, label='Qwen-72B', color=set_color(0,'b',0,False), edgecolor='black')
    # ax2.set_title("Runtime Memory", fontsize=title_size, pad=0)
    ax2.set_xlabel('Runtime Memory', fontsize=xy_label_size)
    ax2.set_xticks(x)
    ax2.set_xticklabels(runtime_labels, rotation=0, fontsize=xy_tick_size-0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    ax2.bar_label(bars3, 
                rotation=90,
                labels=format_labels(normalizedLlama70B[3:], 2),
                fontsize=text_label_size, padding=text_padding, color='darkorange')
    ax2.bar_label(bars4,
                rotation=90,
                labels=format_labels(normalizedQwen72B[3:], 2),
                fontsize=text_label_size, padding=text_padding, color='navy')
        
    for ax in [ax1, ax2]:
        ax.tick_params(axis='y', labelsize=xy_tick_size)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax2.legend(fontsize=12)
    # 添加统一图例
    # handles, labels = ax1.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=xy_label_size+1, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间

# 设置样式
sns.set_style("white")
plt.tight_layout()
# 生成图形
sinplot()
sns.despine()

plt.savefig("/Users/hanayukino/detailed mem.pdf", format='pdf', dpi=200)
plt.show()