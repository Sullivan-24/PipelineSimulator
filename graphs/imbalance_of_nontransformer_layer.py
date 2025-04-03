import matplotlib.pyplot as plt
from e2e_data import *
import seaborn as sns
import numpy as np
from matplotlib.patches import Patch

# 数据准备
comp_data = [
    # 1F1B
    ("1F1B-F", 'F', (21.684+21.225+21.458+22.449)/4, (58.342+59.566+58.005+60.162)/4),
    ("1F1B-B", 'B', (45.983+45.057+46.229+45.409)/4, (78.800+76.587+77.652+74.893)/4),
    # Backward-splitting
    ("ZBH-F", 'F', (20.998+20.882+20.518+21.359)/4, (56.983+57.393+57.195+57.078)/4),
    ("ZBH-B", 'B', (26.997+28.427+27.338+27.372)/4, (45.929+45.768+45.422+46.298)/4),
    ("ZBH-W", 'W', (18.869+18.340+18.879+18.341)/4, (33.906+32.943+33.697+33.066)/4)
]
memory_data = {
    "F (Activation)" : (1136) / 1024,
    "F (+ Loss)": (1136 + 128 + 64 + 2004 * 2) / 1024,
    "B (Input gradient)" : 0.3392646789550781,
    "B (+ Embedding layer)" : (0.3392646789550781 * 1024 + 1002) / 1024,
    "B (+ Head layer)" : (0.3392646789550781 * 1024 + 1002 + 128 + 64) / 1024,
    "W (Parameter gradient)" : (1416 * 2)/1024,
    "W (+ Embedding layer)" : (1416 * 2 + 2004)/1024,
    "W (+ Head layer)" : (1416 * 2 + 2004 + 128 + 64)/1024,
}
width = 0.4  # 单边柱宽
xy_label_size = 18
text_size = 11
def imbalance():
    def sinplot():
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # 配色方案
        phase_colors = {
            'F': '#FFF2CC',  # 浅黄
            'B': '#DAE8FC',  # 浅蓝
            'W': '#D5E8D4',   # 浅绿
            'F (Activation)': '#FFF2CC', 
            'B (Input gradient)': '#DAE8FC',  
            'W (Parameter gradient)': '#D5E8D4',   
            'F (+ Loss)': '#FFF2CC', 
            'B (+ Embedding layer)': '#DAE8FC',  
            'B (+ Head layer)': '#99CCFF',   
            'W (+ Embedding layer)': '#D5E8D4',  
            'W (+ Head layer)': '#9AC7BF',  
        }

        # Computation imbalance
        x = np.arange(len(comp_data))
        ax = axs[0]
        bars = []
        bars_non = []
        for i, (label, phase, t, nt) in enumerate(comp_data):
            # Transformer层（左半边）
            bars.append(
                ax.bar(i - width/2, t, width, 
                color=phase_colors[phase],
                edgecolor='k',
                linewidth=0.8,
                label='Transformer' if i==0 else "")
            )
            
            # Non-transformer层（右半边）
            bars_non.append(
                ax.bar(i + width/2, nt, width,
                    color=phase_colors[phase],
                    edgecolor='k',
                    linewidth=0.8,
                    hatch='/',
                    alpha=0.9,
                    label='Non-Transformer' if i==0 else "")
            )
            ax.text(i - width/2, t, f'x{t/t:.2f}',
                ha='center', va='bottom', fontsize=text_size)
            ax.text(i + width/2, nt, f'x{nt/t:.2f}',
                ha='center', va='bottom', fontsize=text_size)

        # 坐标轴设置
        ax.set_xticks(x)
        ax.set_xticklabels([d[0] for d in comp_data], rotation=0, fontsize=xy_label_size)
        ax.set_title("Computation imbalance of (F, B, W)", fontsize=xy_label_size)
        ax.set_ylabel("Time (ms)", fontsize=xy_label_size)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
        # 自定义图例
        legend_elements = [
            Patch(facecolor='#FFF2CC', edgecolor='k', label='F'),
            Patch(facecolor='#DAE8FC', edgecolor='k', label='B'),
            Patch(facecolor='#D5E8D4', edgecolor='k', label='W'),
            Patch(facecolor='#FFF2CC', edgecolor='k', hatch='///', label='F + Non-Transformer'),
            Patch(facecolor='#DAE8FC', edgecolor='k', hatch='///', label='B + Non-Transformer'),
            Patch(facecolor='#D5E8D4', edgecolor='k', hatch='///', label='W + Non-Transformer'),
        ]
        ax.legend(handles=legend_elements, ncol=2, loc='upper right', fontsize = 8.5)
        
        # Memory imbalance
        ax = axs[1]
        x = np.arange(len(memory_data.keys()))
        for i,kv in enumerate(memory_data.items()):
            hatch = False
            if "(+ " in kv[0]:
                hatch = True
            ax.bar(i, kv[1], width*2, 
                color=phase_colors[kv[0]],
                edgecolor='k',
                linewidth=0.8,
                hatch="/" if hatch else "",
                label='Transformer' if i==0 else "")
            ax.text(i, kv[1], f'x{kv[1]/memory_data["F (Activation)"]:.2f}',
                ha='center', va='bottom', fontsize=text_size)
        # print([d[0][0] for d in memory_data.items()])
        ax.set_xticklabels(['F', 'F', 'F', 'B', 'B', 'B', 'W', 'W', 'W'], rotation=0, fontsize=xy_label_size)
        ax.set_ylabel("Peak memory overhead (GB)", fontsize=xy_label_size)
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        legend_elements = [
            Patch(facecolor='#FFF2CC', edgecolor='k', label='F (Activation)'),
            Patch(facecolor='#FFF2CC', edgecolor='k', hatch='///', label='F (+ Loss)'),
            Patch(facecolor='#DAE8FC', edgecolor='k', label='B (Input gradient)'),
            Patch(facecolor='#DAE8FC', edgecolor='k', hatch='///', label='B (+ Embedding layer)'),
            Patch(facecolor='#99CCFF', edgecolor='k', hatch='///', label='B (+ Head layer)'),
            Patch(facecolor='#D5E8D4', edgecolor='k', label='W (Parameter gradient)'),
            Patch(facecolor='#D5E8D4', edgecolor='k', hatch='///', label='W (+ Embedding layer)'),
            Patch(facecolor='#9AC7BF', edgecolor='k', hatch='///', label='W (+ Head layer)'),
        ]
        ax.set_title("Memory imbalance of (F, B, W)", fontsize=xy_label_size)
        ax.legend(handles=legend_elements, ncol=1, loc='upper center', fontsize = 8.5)
        plt.tight_layout()
        # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/imbalance.pdf", format='pdf', dpi=200)
    plt.show()

imbalance()
