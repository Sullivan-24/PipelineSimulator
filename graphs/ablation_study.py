import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
# 数据预处理函数
def process_data(data):
    return [v if v != "OOM" else 0 for v in data]

def get_min_v(array):
    min_v = max(array)
    for v in array:
        if v == 0: continue
        min_v = min(min_v, v)
    return min_v

def draw_ablation_study():
    def sinplot():
        # 配置绘图参数
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9
        })

        # 创建画布
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
        plt.subplots_adjust(wspace=0.25)

        # 颜色配置
        colors = {
            "1F1B": "#FFF2CC",
            "+ Adaptive model placement": "#FFE6CC",
            "+ Adaptive workload schedule": "#D5E8D4",
            "+ Memory monitor": "#99CCFF",
            "+ Overlap-aware": "#F8CECC",
        }
        methods = ["1F1B", "+ Adaptive model placement", "+ Adaptive workload schedule", "+ Memory monitor", "+ Overlap-aware"]
        models = ["70B LLaMA","42B LLaMA","14B LLaMA"]
        configs = ["pp4 tp4 dp2 mb16","pp4 tp8 dp1 mb16","pp8 tp1 dp1 mb32"]
        text_size = 11
        label_size = 11
        xy_label_size = 13
        for i in range(len(models)):
            model = models[i]
            config = configs[i]
            model_70b = ablation_study_data[model][config]["homo"]
            values_70b = process_data([model_70b[m] for m in methods])

            bars_70b = axs[i].bar(methods, values_70b, color=[colors[m] for m in methods])
            axs[i].set_xlabel(f'{model} {config}', fontsize=xy_label_size)
            if i == 0:
                axs[i].set_ylabel('Tokens/GPU/second', fontsize=xy_label_size)
            axs[i].grid(axis='y', linestyle='--', alpha=0.7)
            min_v_70b = get_min_v(values_70b)
            axs[i].set_ylim(min_v_70b*0.9, max(values_70b)*1.0)
            axs[i].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            # 添加数值标签
            for bar in bars_70b:
                height = bar.get_height()
                if height > 0:
                    axs[i].text(bar.get_x() + bar.get_width()/2, height, f'x{height/min_v_70b:.3f}',
                            ha='center', va='bottom', fontsize=text_size)
                else:
                    axs[i].text(bar.get_x() + bar.get_width()/2, min_v_70b*0.9, f'OOM',
                            ha='center', va='bottom', fontsize=text_size)
                                
        # 添加图例
        legend_labels = [plt.Rectangle((0,0),1,1, color=colors[m]) for m in methods]
        fig.legend(legend_labels, 
                methods, 
                loc='lower center', 
                ncol=len(legend_labels), 
                bbox_to_anchor=(0.5, 0.95450),
                fontsize=label_size,
            )

        plt.tight_layout()
    sinplot()
    sns.despine()
    plt.savefig('/Users/hanayukino/ablation_study.pdf', bbox_inches='tight')
    plt.show()

draw_ablation_study()