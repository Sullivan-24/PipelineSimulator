import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
import matplotlib.colors as mcolors
from regression import update_solving_time
from matplotlib.patches import Rectangle

fontsize=6
titlesize=20 + fontsize
labelsize= 20 + fontsize
ticksize= 16 + fontsize
legendsize= 20 + fontsize - 0.5

solving_time = update_solving_time()

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import seaborn as sns

def draw_solving_time():
    def sinplot():
        plt.figure(figsize=(10, 6))

        # Define colors and markers for different configurations
        styles = {
            "S": {"ILP Solver (Simple)": (line_colors["Mist"], 'o'), "ILP Solver": (line_colors["ZB"], 'o'), "OctoPipe": (line_colors["OctoPipe"], 'o')},
            "M": {"ILP Solver (Simple)": (line_colors["Mist"], '^'), "ILP Solver": (line_colors["ZB"], '^'), "OctoPipe": (line_colors["OctoPipe"], '^')},
            "L": {"ILP Solver (Simple)": (line_colors["Mist"], 'D'), "ILP Solver": (line_colors["ZB"], 'D'), "OctoPipe": (line_colors["OctoPipe"], 'D')},
        }

        # Plot each configuration in the main plot
        for model in solving_time:
            for config in solving_time[model]:
                data = solving_time[model][config]
                x = list(data.keys())
                y = list(data.values())
                if config in styles[model].keys():
                    color, marker = styles[model][config]
                else:
                    continue
                label = f"{model} - {config}"
                if "OctoPipe" in config:
                    linestyle = '-'
                elif "ILP Solver (Simple)" in config:
                    linestyle = '-.'
                else:
                    linestyle = '--'
                plt.semilogy(np.log2(x), y, marker=marker, linestyle=linestyle, color=color,
                             markersize=12, linewidth=2.5, label=label, alpha=1.0)

        # Customize the main plot
        x_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        plt.xticks(np.log2(x_ticks), x_ticks, fontsize=ticksize)
        plt.xlabel("Micro-batch#", fontsize=labelsize)
        plt.yticks(fontsize=ticksize)
        plt.ylabel("Time (seconds, log scale)", fontsize=labelsize)
        plt.title("Pipeline Generation Time Across Configurations", fontsize=titlesize, y=1.00)
        plt.grid(True, which="both", ls="--", alpha=0.25)

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        group = len(handles) // 3
        order = []
        for i in range(group):
            order.extend([i, i + group, i + group * 2])
        plt.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc='upper left', fontsize=legendsize - 5, frameon=False,
            labelspacing=0.5,
        )
        sns.despine()

        # Create an inset plot for the dense region
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='center right',
                              bbox_to_anchor=(0.5725, 0.0, 0.4, 0.8), bbox_transform=ax.figure.transFigure)
        for model in solving_time:
            for config in solving_time[model]:
                data = solving_time[model][config]
                x = list(data.keys())
                y = list(data.values())
                if config in styles[model].keys():
                    color, marker = styles[model][config]
                else:
                    continue
                label = f"{model} - {config}"
                linestyle = '-' if "OctoPipe" in config else '-'
                ax_inset.semilogy(np.log2(x), y, marker=marker, linestyle=linestyle, color=color,
                                  markersize=12, linewidth=2, label=label, alpha=1.0)
        
        ax_inset.set_xlim(np.log2(64)-0.25, np.log2(256)+0.25)
        ax_inset.set_ylim(1e0-0.5, 1e2+100)
        # ax_inset.set_ylim(-1e3, 1e5+100)

        # 设置y轴刻度为 1e3, 1e2, 1e1
        yticks = [1e2, 1e1, 1e0]
        ax_inset.set_yticks(yticks)
        ax_inset.yaxis.set_minor_locator(plt.NullLocator())
        # ax_inset.set_yticklabels(['1000', '100', '10'])

        # 只保留这些刻度的网格线
        ax_inset.grid(True, which='major', ls='--', alpha=0.75)
        ax_inset.grid(False, which='minor')

        ax_inset.tick_params(axis='both', labelsize=ticksize)
        ax_inset.set_xticklabels([])

        # Add a mark to indicate the zoomed region in the main plot
        mark_inset(ax, ax_inset, loc1=3, loc2=4, fc="lightgray", ec="0.25", lw=2, ls="--")

        plt.tight_layout()

    sinplot()
    plt.savefig('/Users/hanayukino/solving_time.pdf', bbox_inches='tight')
    plt.show()

draw_solving_time()