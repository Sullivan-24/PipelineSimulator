import math

def count_valid_fbw_permutations(N):
    total = math.factorial(3 * N) // (6 ** N)
    return total

def count_valid_fb_permutations(N):
    total = math.factorial(2 * N) // (2 ** N)
    return total

nmbs = [2, 4, 8, 12, 16, 20, 24, 32]
cases = [[], []]
for i in range(2):
    for nmb in nmbs:
        if i == 0:
            cn = count_valid_fb_permutations(nmb)
        else:
            cn = count_valid_fbw_permutations(nmb)
        cases[i].append(cn)
print(cases)
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
        "F+B": cases[0],
        "F+I+W": cases[1],
    }
    sequence_lengths = [str(nmb) for nmb in nmbs]
    colors = {"F+B":'#60A917', 
              "F+I+W":'#FF8000', 
              "FIBFIB":'#1BA1E2',
    }
    markers = {"F+B":'.', 
              "F+I+W":'^', 
              "VVV":'*',
    }

    # 画图
    plt.figure(figsize=(4, 4))
    for label, values in case_with_pp.items():
        plt.plot(sequence_lengths, values, marker=markers[label], lw=3, markersize=16, label=f"{label}",color=colors[label])

    plt.xlabel("#Micro-batches", fontsize = xy_label_size)
    plt.ylabel("#WS policies (log scale)", fontsize = xy_label_size)
    plt.yscale("log")
    # plt.title("TGS vs. Sequence Length", fontsize=xy_label_size)
    # plt.legend(loc='lower right', prop={'size':xy_tick_size-1})
    plt.legend(prop={'size':xy_tick_size + 4})
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=xy_tick_size)
    plt.yticks(fontsize=xy_tick_size)
    plt.savefig(f"/Users/hanayukino/ws_cases.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

varyseq()