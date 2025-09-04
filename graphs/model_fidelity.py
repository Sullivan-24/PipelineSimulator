import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from e2e_data import *

model   = "Nemotron-H"
seq_key = "Seq_4k"          # 也可换成 Seq_2k
confs   = ["Small", "Medium", "Large"]
methods = ["S-1F1B", "I-1F1B", "ZB", "Mist", "OctoPipe"]

fontsize=6
titlesize=20 + fontsize
labelsize=20 + fontsize
ticksize=16 + fontsize
legendsize=16 + fontsize
fig_width = 10
fig_height = 5.5
def model_fidelity():
    # ---------- 归一化 ----------
    def build_df(data_dict):
        rows = []
        for conf in confs:
            base = data_dict[model][seq_key][conf]["S-1F1B"]
            for m in methods:
                val = data_dict[model][seq_key][conf][m]
                rows.append({"conf": conf, "method": m, "norm": val / base})
        return pd.DataFrame(rows)

    df_bar = build_df(h200_real_data)          # throughput
    df_sim = build_df({k: {sk: {c: {m: 1/v for m,v in inner.items()}
                                for c,inner in outer.items()}
                        for sk,outer in inner2.items()}
                    for k,inner2 in h200_sim_data.items()})   # 1/time
    errors = []
    for m in methods:
        real = df_bar[df_bar.method==m]["norm"].to_list()
        sim  = df_sim[df_sim.method==m]["norm"].to_list()
        print(m)
        for r,s in zip(real, sim):
            error = r-s
            print(round(error,4), end=" ")
            errors.append(error)
        print()
    print(f"Avg. Error: {sum(errors)/(len(errors)-3)} .")


    # ---------- 画图 ----------
    x_group = np.arange(len(confs))            # 0,1,2
    width   = 0.15
    offsets = (np.arange(len(methods)) - (len(methods)-1)/2) * width   # -0.3~+0.3

    fig, ax = plt.subplots(figsize=(10,5.5))

    # 1) 柱状图：3 组 * 5 方法
    for m, off in zip(methods, offsets):
        ax.bar(x_group + off,
            df_bar[df_bar.method==m]["norm"],
            # width, label=f"{m} (real)", color=colors[m], alpha=0.7, edgecolor='black', hatch=hatches[m],)
            width, label=f"{m} (real)", color=colors[m], alpha=0.7)

    # 2) 三条折线：每一条对应一种 conf，穿过该组 5 根柱子中心
    for i, conf in enumerate(confs):
        # x 坐标：该组 5 根柱子的中心
        xs = x_group[i] + offsets
        ys = df_sim[df_sim.conf==conf]["norm"].values
        if i == 0:
            ax.plot(xs, ys,
                    marker='D', markersize=10, c=line_colors[methods[-1]], linestyle='--', lw=3, label=f"Simulated")
        else:
            ax.plot(xs, ys,
                    marker='D', markersize=10, c=line_colors[methods[-1]], linestyle='--', lw=3,)
    ax.set_ylim(0.5,1.6)
    ax.set_xticks(x_group)
    ax.set_xticklabels(confs, fontsize=labelsize)
    ax.tick_params(axis='y', which='both', labelsize=ticksize)
    ax.set_ylabel("Throughput (Normalized)", fontsize=labelsize)
    # ax.set_title(f"Performance Model Fidelity", fontsize=titlesize, y=1.035)
    handle, label = ax.get_legend_handles_labels()
    handle[-3], handle[-1] = handle[-1], handle[-3]
    label[-3], label[-1] = label[-1], label[-3]
    ax.legend(
        handles=handle,
        labels=label,
        fontsize=legendsize,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.75,
        handletextpad=0.25,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    sns.despine()
    plt.savefig("/Users/hanayukino/model_fidelity.pdf", format='pdf', dpi=200, bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    model_fidelity()