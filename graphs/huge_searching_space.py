
import matplotlib.pyplot as plt
xy_label_size = 18
xy_tick_size = 16
def varyseq():
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
    fig, axs = plt.subplots(1,2, figsize=(8, 4), dpi=300)
    for label, values in case_with_pp.items():
        axs[0].plot(sequence_lengths, values, marker=markers[label], lw=3, markersize=16, label=f"#GPU={label}",color=colors[label])

    axs[0].set_xlabel("#Layers\n(a)", fontsize = xy_label_size)
    axs[0].set_ylabel("#MP (log scale)", fontsize = xy_label_size)
    axs[0].set_yscale("log")
    axs[0].legend(prop={'size':xy_label_size - 4})
    axs[0].grid(True)
    axs[0].tick_params(axis='x', labelsize=xy_tick_size)
    axs[0].tick_params(axis='y', labelsize=xy_tick_size)
    
    import math

    def count_valid_fbw_permutations(N):
        total = math.log10(math.factorial(3 * N) // (6 ** N))
        return total

    def count_valid_fb_permutations(N):
        total = math.log10(math.factorial(2 * N) // (2 ** N))
        return total

    nmbs = [2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 128]
    cases = [[], []]
    for i in range(2):
        for nmb in nmbs:
            if i == 0:
                cn = count_valid_fb_permutations(nmb)
            else:
                cn = count_valid_fbw_permutations(nmb)
            cases[i].append(cn)

    # 数据
    case_with_pp = {
        "F+B": cases[0],
        "F+I+W": cases[1],
    }
    sequence_lengths = [str(nmb) for nmb in nmbs]
    colors = {"F+B":'#FF8000', 
              "F+I+W":'#60A917', 
              "FIBFIB":'#1BA1E2',
    }
    markers = {"F+B":'.', 
              "F+I+W":'^', 
              "VVV":'*',
    }

    for label, values in case_with_pp.items():
        axs[1].plot(sequence_lengths, values, marker=markers[label], lw=3, markersize=16, label=f"{label}",color=colors[label])

    axs[1].set_xlabel("#Micro-batches\n(b)", fontsize = xy_label_size)
    axs[1].set_ylabel("#WS (log scale)", fontsize = xy_label_size)
    axs[1].set_yscale("log")
    axs[1].legend(prop={'size':xy_label_size - 4})
    axs[1].grid(True)
    axs[1].tick_params(axis='x', labelsize=xy_tick_size)
    axs[1].tick_params(axis='y', labelsize=xy_tick_size)
    plt.tight_layout()

    plt.savefig(f"/Users/hanayukino/mp_ws.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
    plt.show()

varyseq()