import matplotlib.pyplot as plt
from e2e_data import h200_data as data
from e2e_data import *
import seaborn as sns
fontsize=16
titlesize=22 + fontsize
labelsize= 20 + fontsize
ticksize= 16 + fontsize
legendsize= 20 + fontsize - 0.5
def draw_h200_e2e_seq(seqlen=2):
    def sinplot():
        fig, axes = plt.subplots(1, 3, figsize=(32, 8))
        standard_method = "S-1F1B"
        nomalized = True
        for idx, (ax, kv) in enumerate(zip(axes, data.items())):
            model, seq_data = kv
            seq_2k = seq_data[f"Seq_{seqlen}k"]
            configs = list(seq_2k.keys())
            methods = list(next(iter(seq_2k.values())).keys())
            configs_sorted = sorted(configs, key=lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))
            x = range(len(configs_sorted))
            bar_width = 0.15

            for i, method in enumerate(methods):
                values = [seq_2k[config][method] / seq_2k[config][standard_method] if nomalized else seq_2k[config][method] for config in configs_sorted]
                bars = ax.bar([pos + i * bar_width + bar_width/2 for pos in x], values, edgecolor='black', color=colors[method], hatch=hatches[method], width=bar_width, label=method)
                offset = 1.01
                for v_idx, v in enumerate(values):
                    ax.text(
                        bars[v_idx].get_x() + bars[v_idx].get_width() / 2,
                        v * 1.0 * offset,
                        f"{v:.2f}x",
                        ha='center',
                        va='bottom',
                        rotation=90,
                        color="#000000",
                        fontsize=ticksize  # 缩小字体
                    )

            ax.set_title(" ", color="#FFFFFF", fontsize=titlesize, y=1.025)
            ax.text(0.5, -0.15, model, fontsize=titlesize, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([pos + bar_width * (len(methods) / 2) for pos in x])
            ax.set_xticklabels(configs_sorted, fontsize=labelsize)
            if idx == 0:
                ax.set_ylabel("Throughput (Normalized)", fontsize=labelsize)
            ax.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
            ax.set_ylim(0.2, 1.65)
            ax.grid(axis='y', linestyle='--', alpha=0.25)
            handle, label = ax.get_legend_handles_labels()
        fig.legend(
            handles=handle,
            labels=label,
            fontsize=legendsize,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(methods),
            frameon=False,
        )

        plt.tight_layout()

    sinplot()
    sns.despine()
    plt.savefig(f"/Users/hanayukino/e2e performance {seqlen}k.pdf", format='pdf', dpi=200)

    plt.show()

def draw_h200_e2e():
    def sinplot():
        fig, axes = plt.subplots(2, 3, figsize=(32, 14))  # Increased height to accommodate both rows
        standard_method = "S-1F1B"
        normalized = True
        models = list(data.keys())
        # First row: Seq_2k
        for col, model in enumerate(models):
            ax = axes[0, col]
            seq_data = data[model]
            seq_2k = seq_data["Seq_2k"]
            configs = list(seq_2k.keys())
            methods = list(next(iter(seq_2k.values())).keys())
            # configs_sorted = sorted(configs, key=lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))
            configs_sorted = configs
            x = range(len(configs_sorted))
            bar_width = 0.15

            for i, method in enumerate(methods):
                values = [seq_2k[config][method] / seq_2k[config][standard_method] if normalized else seq_2k[config][method] 
                          for config in configs_sorted]
                bars = ax.bar([pos + i * bar_width + bar_width/2 for pos in x], values, 
                             edgecolor='black', color=colors[method], hatch=hatches[method], 
                             width=bar_width, label=method)
                
                offset = 1.01
                for v_idx, v in enumerate(values):
                    ax.text(
                        bars[v_idx].get_x() + bars[v_idx].get_width() / 2,
                        v * 1.0 * offset,
                        f"{v:.2f}x",
                        ha='center',
                        va='bottom',
                        rotation=90,
                        color="#000000",
                        fontsize=ticksize
                    )
            # ax.set_title(f"{model} (Seqlen = 2k)", fontsize=titlesize, y=1.025)
            ax.set_title(" ", color="#FFFFFF", fontsize=titlesize, y=1.025)
            ax.text(0.5, -0.175, f"{model} (Seqlen = 2k)", fontsize=titlesize, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([pos + bar_width * (len(methods) / 2) for pos in x])
            ax.set_xticklabels(configs_sorted, fontsize=labelsize)
            if col == 0:
                ax.set_ylabel("Throughput (Normalized)", fontsize=labelsize)
            ax.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
            ax.set_ylim(0.2, 1.65)
            ax.grid(axis='y', linestyle='--', alpha=0.25)

        # Second row: Seq_4k
        for col, model in enumerate(models):
            ax = axes[1, col]
            seq_data = data[model]
            seq_4k = seq_data["Seq_4k"]
            configs = list(seq_4k.keys())
            methods = list(next(iter(seq_4k.values())).keys())
            # configs_sorted = sorted(configs, key=lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))
            configs_sorted = configs
            x = range(len(configs_sorted))
            bar_width = 0.15

            for i, method in enumerate(methods):
                values = [seq_4k[config][method] / seq_4k[config][standard_method] if normalized else seq_4k[config][method] 
                          for config in configs_sorted]
                bars = ax.bar([pos + i * bar_width + bar_width/2 for pos in x], values, 
                             edgecolor='black', color=colors[method], hatch=hatches[method], 
                             width=bar_width, label=method)
                
                offset = 1.01
                for v_idx, v in enumerate(values):
                    ax.text(
                        bars[v_idx].get_x() + bars[v_idx].get_width() / 2,
                        v * 1.0 * offset,
                        f"{v:.2f}x",
                        ha='center',
                        va='bottom',
                        rotation=90,
                        color="#000000",
                        fontsize=ticksize
                    )

            # ax.set_title(f"{model} (Seqlen = 4k)", fontsize=titlesize, y=1.025)
            ax.text(0.5, -0.175, f"{model} (Seqlen = 4k)", fontsize=titlesize, ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([pos + bar_width * (len(methods) / 2) for pos in x])
            ax.set_xticklabels(configs_sorted, fontsize=labelsize)
            if col == 0:
                ax.set_ylabel("Throughput (Normalized)", fontsize=labelsize)
            ax.tick_params(axis='y', which='both', length=5, labelsize=ticksize)
            ax.set_ylim(0.2, 1.65)
            ax.grid(axis='y', linestyle='--', alpha=0.25)

        # Create a single legend for all subplots
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles=handles,
            labels=labels,
            fontsize=legendsize,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.025),
            ncol=len(methods),
            frameon=False,
        )

        plt.tight_layout()

    sinplot()
    sns.despine()
    plt.savefig(f"/Users/hanayukino/e2e performance.pdf", format='pdf', dpi=200)
    plt.show()

if __name__ == "__main__":
    draw_h200_e2e()