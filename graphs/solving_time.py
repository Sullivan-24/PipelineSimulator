import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from e2e_data import *
import matplotlib.colors as mcolors
fontsize=18
titlesize=22 + fontsize
labelsize= 20 + fontsize
ticksize= 16 + fontsize
legendsize= 20 + fontsize - 0.5

def draw_solving_time():
    
    def sinplot():
        plt.figure(figsize=(12, 8))

        # Define colors and markers for different configurations
        styles = {
            "pp4layer32": {"1 ILP Solver": ('b', 'o'), "1+3 ILP Solver": ('b', 'o'), "1+2 ILP Solver": ('b', 'o'), "1+2+3 ILP Solver": ('b', 'o'), "1+2+3 OctoPipe": ('g', 's')},
            "pp8layer64": {"1 ILP Solver": ('b', 'o'), "1+3 ILP Solver": ('b', 'o'), "1+2 ILP Solver": ('b', 'o'), "1+2+3 ILP Solver": ('b', 'o'), "1+2+3 OctoPipe": ('m', 'D')}
        }

        # Plot each configuration
        for model in solving_time:
            for config in solving_time[model]:
                data = solving_time[model][config]
                x = list(data.keys())
                y = list(data.values())
                color, marker = styles[model][config]
                label = f"{model} - {config}"
                
                # Plot main line
                # plt.semilogy(x, y, marker=marker, linestyle='-', color=color, 
                #             markersize=8, linewidth=2, label=label)
                plt.semilogy(np.log2(x), y, marker=marker, linestyle='-',
                    markersize=8, linewidth=2, label=label, alpha=0.8)
                
                # Annotate points with time > 1 second
                # for i, (xi, yi) in enumerate(zip(x, y)):
                #     if i == len(y)-1:
                #         plt.annotate(f"{yi:.2e}", (xi, yi), textcoords="offset points", 
                #                     xytext=(0,10), ha='center', fontsize=9)

        # Customize the plot
        x_ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        plt.xticks(np.log2(x_ticks), x_ticks)
        plt.xlabel("Input Size (n)", fontsize=12)
        plt.ylabel("Time (seconds, log scale)", fontsize=12)
        plt.title("Solving Time Comparison Across Configurations", fontsize=14, pad=20)
        plt.grid(True, which="both", ls="--", alpha=0.5)

        # Use a custom x-axis to show all important points
        # all_x_ticks = sorted({x for model in solving_time for config in solving_time[model] 
        #                     for x in solving_time[model][config]})
        # plt.xticks(all_x_ticks, rotation=45)

        # Add legend outside the plot
        plt.legend(loc='upper left', borderaxespad=0.)
        plt.tight_layout()

    sinplot()
    sns.despine()
    plt.savefig('/Users/hanayukino/solving_time.pdf', bbox_inches='tight')
    plt.show()

draw_solving_time()