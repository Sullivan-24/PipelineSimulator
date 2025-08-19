import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
fontsize=6
titlesize=20 + fontsize
labelsize= 20 + fontsize
ticksize= 16 + fontsize
legendsize= 20 + fontsize - 1
def varyseq():

    x = list(vary_length_seqence.keys())
    methods = list(next(iter(vary_length_seqence.values())).keys())

    # 画折线图
    plt.figure(figsize=(10, 6))
    for method in methods:
        y = [vary_length_seqence[seq][method] for seq in x]
        plt.plot(x, y, marker=markers[method],markersize=14, linewidth=2.5, color=line_colors[method], label=method)

    plt.xlabel("Sequence Length",fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.ylabel("Throughput (TGS)",fontsize=labelsize)
    plt.yticks(fontsize=ticksize)
    plt.title("Throughput across varying sequence lengths",fontsize=titlesize)
    plt.legend(fontsize=legendsize, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    sns.despine()

    plt.savefig(f"/Users/hanayukino/e2e_performance_varyseq.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

varyseq()