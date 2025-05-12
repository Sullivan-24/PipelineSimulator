import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
xy_label_size = 28
xy_tick_size = 28
def varyseq(arch="heter"):
    import matplotlib.pyplot as plt

    # 数据
    vary_seq = {
        "OctoPipe": [1752, 2345, 2582, 2630, 2596],
        "ZBH": [1772, 2254, 2439, 2480, 2450],
        "Dapple": [1730, 2203, 2401, 2395, 2380],
    }
    sequence_lengths = ["2K", "4K", "8K", "12K", "16K"]
    colors = {'OctoPipe':'#F53255', 
              'ZBH':'#F46920', 
              'Dapple':'#FFAF00',
    }

    # 画图
    plt.figure(figsize=(12, 4))
    for label, values in vary_seq.items():
        if label == 'ZBH':
            plt.plot(sequence_lengths, values, marker='*', lw=3, markersize=12, linestyle='--', label=label,color=colors[label])
        elif label == 'Dapple':
            plt.plot(sequence_lengths, values, marker='o', lw=3, markersize=12, label=label,color=colors[label])
        else:
            plt.plot(sequence_lengths, values, marker='^', lw=3, markersize=12, label=label,color=colors[label])

    plt.xlabel("Sequence Length", fontsize = xy_label_size)
    plt.ylabel("TGS", fontsize=xy_label_size)
    # plt.title("TGS vs. Sequence Length", fontsize=xy_label_size)
    plt.legend(loc='lower right', prop={'size':xy_tick_size-1})
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.savefig(f"/Users/hanayukino/e2e_performance_varyseq_{arch}.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

varyseq()