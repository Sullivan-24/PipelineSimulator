import matplotlib.pyplot as plt
from e2e_data import *
import seaborn as sns

def pre():
    def sinplot():
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
        x = [4, 8, 10, 12, 16, 20, 24, 40]  # pp值
        colors = ['#FF6B6B', '#45B7D1', '#FFCE30']  # 颜色方案
        # colors = ['#F8CECC', '#99CCFF', '#FFF2CC']  # 颜色方案

        # 绘制每条折线
        for idx, (model, pp_time) in enumerate(preprocessing.items()):
            # print(model, times)
            pps = [eval(k[2:]) for k,v in pp_time.items()]
            times = [v['single'] for k,v in pp_time.items()]
            original_times = times

            layer_num = 80
            if model == "42B LLaMA":
                layer_num = 48
            elif model == "14B LLaMA":
                layer_num = 16
            max_chunks = [layer_num//pp for pp in pps]
            placement_num = [2**(chunk-1) for chunk in max_chunks]
            times = [max(original_times[i]+1, times[i] * placement_num[i]) for i in range(len(times))]
            axes[0].plot(pps, times, 
                    marker='o', 
                    markersize=4,
                    linewidth=1.5,
                    color=colors[idx],
                    label=model)
        axes[0].set_title("Time of pipeline schedule generation (single thread)", fontsize=14)
        axes[0].set_xlabel('Pipeline Parallelism Degree', fontsize=12)
        axes[0].set_ylabel('Time (seconds)', fontsize=14)
        axes[0].set_xticks(x)
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend(title="Model Size", fontsize=10)

        # 添加数据标签
        # for model in preprocessing:
        #     for pp in x:
        #         value = preprocessing[model][f'pp{pp}']['single']
        #         plt.text(pp, value+0.5, 
        #                 f'{value:.2f}s', 
        #                 ha='center', 
        #                 va='bottom',
        #                 fontsize=8)

        for idx, (model, pp_time) in enumerate(preprocessing.items()):
            # print(model, times)
            pps = [eval(k[2:]) for k,v in pp_time.items()]

            times = [v['single'] for k,v in pp_time.items()]
            original_times = times

            layer_num = 80
            if model == "42B LLaMA":
                layer_num = 48
            elif model == "14B LLaMA":
                layer_num = 16
            max_chunks = [layer_num//pp for pp in pps]
            placement_num = [2**(chunk-1) for chunk in max_chunks]
            times = [max(original_times[i]+1, times[i] * placement_num[i] / 96) for i in range(len(times))]
            axes[1].plot(pps, times, 
                    marker='o', 
                    markersize=4,
                    linewidth=1.5,
                    color=colors[idx],
                    label=model)
        axes[1].set_title("Time of pipeline schedule generation (128 threads)", fontsize=14)
        axes[1].set_xlabel('Pipeline Parallelism Degree', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].set_xticks(x)
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend(title="Model Size", fontsize=10)

        plt.tight_layout()
    # 生成图形
    sinplot()
    sns.despine()
    plt.savefig("/Users/hanayukino/preprocessing.pdf", format='pdf', dpi=200)

    plt.show()

pre()