import matplotlib.pyplot as plt
from e2e_data import *
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# 数据准备
comp_data = [
    # 1F1B
    # ("F-1F1B", 'F', (21.684+21.225+21.458+22.449)/4, (58.342+59.566+58.005+60.162 - (21.684+21.225+21.458+22.449))/4),
    # ("B-1F1B", 'B', (45.983+45.057+46.229+45.409)/4, (78.800+76.587+77.652+74.893 - (45.983+45.057+46.229+45.409))/4),
    # Backward-splitting
    ("F-ZBH", 'F', (20.998+20.882+20.518+21.359)/4/2, (56.983+57.393+57.195+57.078 - (20.998+20.882+20.518+21.359))/4),
    ("B-ZBH", 'B', (26.997+28.427+27.338+27.372)/4/2, (45.929+45.768+45.422+46.298 - (26.997+28.427+27.338+27.372))/4),
    ("W-ZBH", 'W', (18.869+18.340+18.879+18.341)/4/2, (33.906+32.943+33.697+33.066 - (18.869+18.340+18.879+18.341))/4),
]
memory_data = {
    "F (Activation)" : (1136) / 1024,
    "B (Input gradient)" : 0.3392646789550781,
    "B (Embedding layer)" : (1002) / 1024,
    "B (Head layer)" : (1002 + 128 + 64) / 1024,
    "W (Embedding layer)" : (2004)/1024,
    "W (Head layer)" : (2004 + 128 + 64)/1024,
    "W (Parameter gradient)" : (1416 * 2)/1024,
    "F (Loss)": (128 + 64 + 2004 * 2) / 1024,
}
width = 0.4  # 单边柱宽
xy_label_size = 26
legend_text_size = 23
text_size = 23
weight = ''
def imbalance():
    def sinplot():
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        
        # 配色方案
        # phase_colors = {
        #     'F': '#FFF2CC',  # 浅黄
        #     'B': '#DAE8FC',  # 浅蓝
        #     'W': '#D5E8D4',   # 浅绿
        #     'F (Activation)': '#FFF2CC', 
        #     'B (Input gradient)': '#DAE8FC',  
        #     'W (Parameter gradient)': '#D5E8D4',   
        #     'F (Loss)': '#FFF2CC', 
        #     'B (Embedding layer)': '#DAE8FC',  
        #     'B (Head layer)': '#99CCFF',   
        #     'W (Embedding layer)': '#D5E8D4',  
        #     'W (Head layer)': '#9AC7BF',  
        # }
        phase_colors = {
            'F': '#FFF2CC',  # 浅黄
            'B': '#DAE8FC',  # 浅蓝
            'W': '#D5E8D4',   # 浅绿
            'F (Activation)': '#FFF2CC', 
            'B (Input gradient)': '#DAE8FC',  
            'W (Parameter gradient)': '#D5E8D4',   
            'F (Loss)': '#FFF2CC', 
            'B (Embedding layer)': '#DAE8FC',  
            'B (Head layer)': '#DAE8FC',   
            'W (Embedding layer)': '#D5E8D4',  
            'W (Head layer)': '#D5E8D4',  
        }

        # Computation imbalance
        x = np.arange(len(comp_data))
        ax = axs[0]
        bars = []
        bars_non = []
        maxv = -1
        for i, (label, phase, t, nt) in enumerate(comp_data):
            # Transformer层（左半边）
            bars.append(
                ax.bar(i - width/2, t, width, 
                color=phase_colors[phase],
                edgecolor='k',
                linewidth=0.8,
                label='Trans.' if i==0 else "")
            )
            
            # Non-transformer层（右半边）
            bars_non.append(
                ax.bar(i + width/2, nt, width,
                    color=phase_colors[phase],
                    edgecolor='k',
                    linewidth=0.8,
                    hatch='/',
                    alpha=0.9,
                    label='Non-Trans.' if i==0 else "")
            )
            standard_v = comp_data[0][2] if "1F1B" in label else comp_data[2][2]
            # ax.text(i - width/2, t, f'x{t/standard_v:.2f}',
            #     ha='center', va='bottom', fontsize=text_size)
            # ax.text(i + width/2, nt, f'x{nt/standard_v:.2f}',
            #     ha='center', va='bottom', fontsize=text_size)
            maxv = max(t, maxv, nt)
            ax.text(i - width/2, t, f'{t/t:.2f}×',
                ha='center', va='bottom', rotation=90, fontsize=text_size)
            ax.text(i + width/2, nt, f'{nt/t:.2f}×',
                ha='center', va='bottom', rotation=90, fontsize=text_size)

        # 坐标轴设置
        ax.set_xticks(x)
        ax.set_ylim(0,maxv * 1.2)
        ax.set_xticklabels([d[0] for d in comp_data], rotation=0, fontsize=xy_label_size)
        ax.set_xlabel('(a) Computational imbalance', fontsize=xy_label_size-3, labelpad=15)
        ax.set_ylabel("Time (ms)", fontsize=xy_label_size)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度
        ax.tick_params(axis='y', labelsize=text_size)  # 隐藏X轴刻度
        # plt.setp(ax.get_xticklabels(), fontweight='bold')
        # plt.setp(ax.get_yticklabels(), fontweight='bold')
        # 自定义图例
        # legend_elements = [
        #     Patch(facecolor='#FFF2CC', edgecolor='k', label='F on Transformer'),
        #     Patch(facecolor='#DAE8FC', edgecolor='k', label='B on Transformer'),
        #     Patch(facecolor='#D5E8D4', edgecolor='k', label='W on Transformer'),
        #     Patch(facecolor='#FFF2CC', edgecolor='k', hatch='///', label='F on Non-Transformers'),
        #     Patch(facecolor='#DAE8FC', edgecolor='k', hatch='///', label='B on Non-Transformers'),
        #     Patch(facecolor='#D5E8D4', edgecolor='k', hatch='///', label='W on Non-Transformers'),
        # ]
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='k', label='Trans.'),
            Patch(facecolor='#FFFFFF', edgecolor='k', hatch='//', label='Non-Trans.'),
        ]
        ax.legend(handles=legend_elements, 
                ncol=1, 
                bbox_to_anchor=(1.05,0.97),
                loc='upper right', 
                # fontsize = legend_text_size
                prop={'size':legend_text_size-1},
        )
        maxv = -1
        # Memory imbalance
        ax = axs[1]
        x = np.arange(len(memory_data.keys()))
        for i,kv in enumerate(memory_data.items()):
            hatch = ""
            if "Emb" in kv[0]:
                hatch = "/"
            if "Head" in kv[0]:
                hatch='+'
            if "Loss" in kv[0]:
                hatch='\\'
            ax.bar(i, kv[1], width*2, 
                color=phase_colors[kv[0]],
                edgecolor='k',
                linewidth=0.8,
                hatch=hatch,
                label='Transformer' if i==0 else "")
            maxv = max(kv[1], maxv)
            ax.text(i, kv[1] + 0.01, f'{kv[1]/memory_data["F (Activation)"]:.2f}×',
                ha='center', va='bottom', rotation=90, fontsize=text_size)
        # print([d[0][0] for d in memory_data.items()])
        ax.set_ylim(0,maxv * 1.1)

        ax.set_xticklabels(['F', 'F', 'F-L', 'B', 'B-E', 'B-H', 'W', 'W-E', 'W-H'], rotation=0, fontsize=xy_label_size)
        # ax.set_xticklabels(['F', 'F', 'F-L', 'B', 'B-E', 'B-H', 'W', 'W-E', 'W-H'], rotation=0, fontsize=xy_label_size)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏X轴刻度
        ax.tick_params(axis='y', labelsize=text_size)  # 隐藏X轴刻度
        ax.set_ylabel("Peak memory (GB)", fontsize=xy_label_size)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        # legend_elements = [
        #     Patch(facecolor='#FFF2CC', edgecolor='k', label='F (Activation)'),
        #     Patch(facecolor='#FFF2CC', edgecolor='k', hatch='///', label='F (Loss)'),
        #     Patch(facecolor='#DAE8FC', edgecolor='k', hatch='///', label='B (Input gradient)'),
        #     Patch(facecolor='#DAE8FC', edgecolor='k', hatch='\\\\', label='B (Embedding layer)'),
        #     Patch(facecolor='#99CCFF', edgecolor='k', hatch='///', label='B (Head layer)'),
        #     Patch(facecolor='#D5E8D4', edgecolor='k', hatch='///', label='W (Parameter gradient)'),
        #     Patch(facecolor='#D5E8D4', edgecolor='k', hatch='\\\\', label='W (Embedding layer)'),
        #     Patch(facecolor='#9AC7BF', edgecolor='k', hatch='///', label='W (Head layer)'),
        # ]
        legend_elements = [
            Patch(facecolor='#FFFFFF', edgecolor='k', label='Trans.'),
            Patch(facecolor='#FFFFFF', edgecolor='k', hatch='\\\\', label='Loss'),
            Patch(facecolor='#FFFFFF', edgecolor='k', hatch='//', label='Emb.'),
            Patch(facecolor='#FFFFFF', edgecolor='k', hatch='+', label='Head'),
        ]
        ax.set_xlabel('(b) Memory imbalance', fontsize=xy_label_size-3, labelpad=15)

        ax.legend(handles=legend_elements, ncol=1, loc='upper left',
            bbox_to_anchor=(-0.03, 0.98),  # y 值从 1 调整到 0.95 表示向下移动
            # fontsize = legend_text_size,
            prop={'size':legend_text_size-1},
        )
        legend_elements = [
            Patch(facecolor='#FFF2CC', edgecolor='k', label='F'),
            Patch(facecolor='#DAE8FC', edgecolor='k', label='B'),
            Patch(facecolor='#D5E8D4', edgecolor='k', label='W'),
        ]
        fig.legend(
            handles=legend_elements,
            # fontsize = legend_text_size,
            prop={'size':legend_text_size},
            ncol=3,
            bbox_to_anchor=(0.5, 1.02),
            loc='upper center',
        )
        plt.tight_layout()

        # plt.setp(ax.get_xticklabels(), fontweight='bold')
        # plt.setp(ax.get_yticklabels(), fontweight='bold')
        # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/imbalance.pdf", format='pdf', dpi=200)
    plt.show()

imbalance()
