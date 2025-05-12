import matplotlib.pyplot as plt

# 数据
labels = [
    '+ IAM',
    '+ Adaptive WS',
    '+ Adaptive MP',
    'Baseline'
]
xlabels = [
    "(a) Layer Imbalance",
    "(b) Layer + Device Imbalance",
]
speedups = [
    [1.12, 1.09, 1.05, 1.00],
    [1.69, 1.51, 1.32, 1.00],
]
colors = ['#FFB570'] * (len(labels) - 1) + ['#6e6e6e']
# colors = {
#     "OctoPipe": "#F8CECC",
#     "Interleaved": "#99CCFF",
#     "ZBV": "#D5E8D4",
#     "ZBH": "#FFE6CC",
#     "Dapple": "#FFF2CC",
#     "Dapple": "#FFF2CC",
#     "Metis": "#E1D5E7",
#     "Alpa": "#B0E3E6",
# }
colors = ["#F8CECC", "#D5E8D4", '#99CCFF', "#FFF2CC"]
baseline = [
    [1.06, 'Interleaved'],
    [1.21, 'Alpa']
]
nr = 2
nc = 1
# 创建图形
fig, axs = plt.subplots(nrows=nr, ncols=nc, figsize=(8, 4))
label_size = 16
text_size = 14
legend_size = 16
min_v = 0.8
for idx in range(nr * nc):
    ax = axs[idx]
    if idx == 0:
        min_v = 0.9
    else:
        min_v = 0.7
    speedup = speedups[idx]
    # 横向柱状图
    bars = ax.barh(range(len(labels)), speedup, color=colors, edgecolor='black')
    offset = 0.1
    if idx == 0:
        offset = 0.025
    # 添加每个 bar 的右侧文字
    y_offset = -0.06
    for i, (bar, val) in enumerate(zip(bars, speedup)):
        ax.text(val - offset, bar.get_y() + bar.get_height()/2 + y_offset,
                f'{val:.2f}x', va='center', ha='left', fontsize=text_size, weight='bold')
        if not labels[i] == labels[-1]:
            ax.text(min_v, bar.get_y() + bar.get_height()/2 + y_offset,
                    f'{labels[i]}', va='center', ha='left', fontsize=text_size, weight='bold')
        else:
            ax.text(min_v, bar.get_y() + bar.get_height()/2 + y_offset,
                    # f'{labels[i]}', va='center', ha='left', fontsize=text_size, color="white", weight='bold')
                    f'{labels[i]}', va='center', ha='left', fontsize=text_size, weight='bold')

    # Y轴设置
    ax.set_yticks(range(len(labels)))
    # ax.set_yticklabels(labels, fontsize=11)

    # X轴设置
    ax.set_xlabel(f'{xlabels[idx]}', fontsize=label_size)
    ax.set_xlim(min_v, max(speedup) * 1.05)
    ax.axvline(baseline[idx][0], color='black', linestyle='--', lw=3,  label=baseline[idx][1])

    # 图例
    ax.legend(loc='upper right', prop={'size': legend_size},)
    ax.tick_params(axis='both', labelsize=label_size)  # 隐藏X轴刻度
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏y轴刻度
    
    # plt.setp(ax.get_xticklabels())
    # plt.setp(ax.get_yticklabels())
    # # 去掉顶部和右侧边框
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"/Users/hanayukino/speedup_breakdown.pdf", 
              format='pdf', 
              dpi=300,
              bbox_inches="tight")
plt.show()
