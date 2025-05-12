def stirling_second_kind(L, S):
    dp = [[0] * (S + 1) for _ in range(L + 1)]
    dp[0][0] = 1
    
    for l in range(1, L + 1):
        for s in range(1, S + 1):
            dp[l][s] = s * dp[l-1][s] + dp[l-1][s-1]
    
    return dp[L][S]

layers = [24, 28, 32, 36, 48, 64, 80]
pps = [4, 8, 16]
cases = [[], [], []]
for i, pp in enumerate(pps):
    for layer in layers:
        cases[i].append(stirling_second_kind(layer, pp))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
xy_label_size = 16
xy_tick_size = 16
def varyseq(arch="heter"):
    import matplotlib.pyplot as plt

    # 数据
    case_with_pp = {
        4: cases[0],
        8: cases[1],
        16: cases[2],
    }
    sequence_lengths = ["24", "28", "32", "36", "48", "64", "80"]
    colors = {4:'#60A917', 
              8:'#FF8000', 
              16:'#1BA1E2',
    }
    markers = {4:'.', 
              8:'^', 
              16:'*',
    }

    # 画图
    plt.figure(figsize=(4, 4))
    for label, values in case_with_pp.items():
        plt.plot(sequence_lengths, values, marker=markers[label], lw=3, markersize=16, label=f"#GPU={label}",color=colors[label])

    plt.xlabel("#Layers", fontsize = xy_label_size)
    plt.ylabel("#MP policies (log scale)", fontsize = xy_label_size)
    plt.yscale("log")
    # plt.title("TGS vs. Sequence Length", fontsize=xy_label_size)
    # plt.legend(loc='lower right', prop={'size':xy_tick_size-1})
    plt.legend(prop={'size':xy_tick_size + 0})
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.savefig(f"/Users/hanayukino/cases.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

varyseq()