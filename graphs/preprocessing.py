import matplotlib.pyplot as plt
from e2e_data import *
import seaborn as sns
xy_lable_size = 23
xy_tick_size = 15
def pre():
    def sinplot():
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        x = [4, 8, 10, 12, 16, 20, 24, 40]  # pp值
        colors = ['#FF6B6B', '#45B7D1', '#FFCE30']  # 颜色方案
        colors = ['#FF6B6B', '#45B7D1', '#45B7D1']  # 颜色方案
        # colors = ['#F8CECC', '#99CCFF', '#FFF2CC']  # 颜色方案
        ax = axes
        # 绘制每条折线
        for idx, (model, pp_time) in enumerate(preprocessing.items()):
            # print(model, times)
            pps = [eval(k[2:]) for k,v in pp_time.items()]
            times = [v['single'] for k,v in pp_time.items()]
            original_times = times
            
            layer_num = 80
            if model == "42B":
                layer_num = 48
            elif model == "14B":
                layer_num = 16
            if model == "42B":
                continue
            max_chunks = [layer_num//pp for pp in pps]
            placement_num = [2**(chunk-1) for chunk in max_chunks]
            stimes = [max(original_times[i], times[i] * placement_num[i]) for i in range(len(times))]
            ax.plot(pps, stimes, 
                    marker='o', 
                    markersize=8,
                    linewidth=2,
                    color=colors[idx],
                    label=f"{model}, single thread")
            
            mtimes = [max(original_times[i] * 1.05, times[i] * placement_num[i] / 96) for i in range(len(times))]

            ax.plot(pps, mtimes, 
                    marker='*', 
                    markersize=8,
                    linewidth=2,
                    linestyle='--',
                    color=colors[idx],
                    label=f"{model}, 128 threads")
            print(stimes)
            print(mtimes)
            
        # ax.set_title("Single thread", fontsize=xy_lable_size)
        ax.set_xlabel('Pipeline parallelism size', fontsize=xy_lable_size)
        ax.set_ylabel('Time (second)', fontsize=xy_lable_size-2)
        ax.set_xticks(x)
        ax.tick_params(axis='both', which='major', labelsize=xy_tick_size)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=xy_lable_size-4)  # 隐藏X轴刻度

        handles, labels = plt.gca().get_legend_handles_labels()
        # print(handles)
        # print(labels)
        # ax.legend([handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]], [labels[0], labels[2], labels[4], labels[1], labels[3], labels[5]], fontsize=xy_lable_size - 9, ncol=2)
        ax.legend(ncol=2, 
                prop={'size': xy_lable_size - 7.5},  # 字号和加粗
        )
        plt.tight_layout()
        # plt.setp(ax.get_xticklabels(), fontweight='bold')
        # plt.setp(ax.get_yticklabels(), fontweight='bold')
    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/preprocessing.pdf", format='pdf', dpi=200)

    plt.show()

pre()