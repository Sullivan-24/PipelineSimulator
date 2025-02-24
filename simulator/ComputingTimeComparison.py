# import matplotlib.pyplot as plt
# import numpy as np

# ccolors = {
#     "F_1st":'#3299FF',
#     "B_1st":'#00FFFF',
#     "F_last":'#3299FF',
#     "B_last":'#00FFFF',
#     "emb":'#FF0000',
#     "head":'#4C0099',
#     "ce":'#FF8000',
# }

# # 数据
# x = ["1k", "2k", "4k", "8k", "16k"]
# F_1st = [3.142, 5.694, 12.441, 37.468, 115.097]
# B_1st = [5.667, 10.031, 21.238, 71.181, 223.768]
# F_last = [8.939, 12.068, 22.496, 58.208, 159.809]
# B_last = [8.444, 16.524, 41.99, 101.699, 287.042]
# emb = [0.114, 0.270, 0.521, 1.054, 2.083]
# head = [1.586, 3.078, 5.982, 12.011, 26.635]
# ce = [1.365, 2.653, 5.279, 10.774, 21.368]

# # 归一化函数
# def normalize(data, ref):
#     return [d / r for d, r in zip(data, ref)]

# # 归一化数据
# F_1st_norm = normalize(F_1st, F_1st)
# B_1st_norm = normalize(B_1st, F_1st)
# F_last_norm = normalize(F_last, F_1st)
# B_last_norm = normalize(B_last, F_1st)

# # 设置柱状图的宽度
# bar_width = 0.15

# # 设置x轴的位置
# index = np.arange(len(x))
# plt.figure(figsize=(5, 4))

# # 绘制柱状图
# bars1 = plt.bar(index + 1.5 * bar_width, F_1st, bar_width, color=ccolors["F_1st"], label='F w/o Head+CE')
# bars2 = plt.bar(index + 2.5 * bar_width, F_last, bar_width, color=ccolors["F_last"], label='F w/ Head+CE', edgecolor="black", hatch='//')
# bars3 = plt.bar(index + 3.5 * bar_width, B_1st, bar_width, color=ccolors["B_1st"], label='B w/o Head+CE')
# bars4 = plt.bar(index + 4.5 * bar_width, B_last, bar_width, color=ccolors["B_last"], label='B w/ Head+CE', edgecolor="black", hatch='\\\\')

# # 在每个柱状图上添加归一化后的数值
# for bars, norm_values in zip([bars1, bars2, bars3, bars4], [F_1st_norm, F_last_norm, B_1st_norm, B_last_norm]):
#     for bar, norm_value in zip(bars, norm_values):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height, f'x{norm_value:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)

# title = 'Computing times (ms) of F and B'
# # 添加x轴标签和标题
# plt.xlabel('Sequence Length')
# plt.ylabel('Time (ms)')
# plt.yscale('log')
# plt.ylim((0,600))
# plt.title(title)
# plt.xticks(index + 3 * bar_width, x)

# # 添加图例
# plt.legend()
# plt.grid()

# # 显示图形
# plt.savefig("/Users/hanayukino/{}.svg".format(title), format='svg',dpi=200)
# plt.show()


# F_1st = [9.808,17.703,38.785,92.837]
# B_1st = [16.245,32.145,71.478,165.481]
# F_last = [17.559,43.762,96.225,156.866]
# B_last = [36.12,60.184,126.098,277.441]

# F_1st_norm = normalize(F_1st, F_1st)
# B_1st_norm = normalize(B_1st, F_1st)
# F_last_norm = normalize(F_last, F_1st)
# B_last_norm = normalize(B_last, F_1st)

import matplotlib.pyplot as plt
import numpy as np
title_size = 16
xy_label_size = 16
xy_tick_size = 16
data_value_font_size = 12
# 颜色定义
ccolors = {
    "F_1st": '#3299FF',
    "B_1st": '#00FFFF',
    "F_last": '#3299FF',
    "B_last": '#00FFFF',
    "emb": '#FF0000',
    "head": '#4C0099',
    "ce": '#FF8000',
}

# 原始数据
# x = ["1k", "2k", "4k", "8k", "16k"]
# F_1st = [3.142, 5.694, 12.441, 37.468, 115.097]
# B_1st = [5.667, 10.031, 21.238, 71.181, 223.768]
# F_last = [8.939, 12.068, 22.496, 58.208, 159.809]
# B_last = [8.444, 16.524, 41.99, 101.699, 287.042]
x = ["1k", "2k", "4k", "8k"]
F_1st = [3.142, 5.694, 12.441, 37.468]
B_1st = [5.667, 10.031, 21.238, 71.181]
F_last = [8.939, 12.068, 22.496, 58.208]
B_last = [8.444, 16.524, 41.99, 101.699]

# 新增的 hiddensize 为 8K 的数据
x_hidden = ["1k", "2k", "4k", "8k"]
F_1st_hidden = [9.808, 17.703, 38.785, 92.837]
B_1st_hidden = [16.245, 32.145, 71.478, 165.481]
F_last_hidden = [17.559, 43.762, 96.225, 156.866]
B_last_hidden = [36.12, 60.184, 126.098, 277.441]

# 归一化函数
def normalize(data, ref):
    return [d / r for d, r in zip(data, ref)]

# 归一化原始数据
F_1st_norm = normalize(F_1st, F_1st)
B_1st_norm = normalize(B_1st, F_1st)
F_last_norm = normalize(F_last, F_1st)
B_last_norm = normalize(B_last, F_1st)

# 归一化 hiddensize 为 8K 的数据
F_1st_hidden_norm = normalize(F_1st_hidden, F_1st_hidden)
B_1st_hidden_norm = normalize(B_1st_hidden, F_1st_hidden)
F_last_hidden_norm = normalize(F_last_hidden, F_1st_hidden)
B_last_hidden_norm = normalize(B_last_hidden, F_1st_hidden)

# 设置柱状图的宽度
bar_width = 0.15

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 绘制原始数据的柱状图
index = np.arange(len(x))
bars1 = ax1.bar(index + 1.5 * bar_width, F_1st, bar_width, color=ccolors["F_1st"], label='F w/o Head+CE')
bars2 = ax1.bar(index + 2.5 * bar_width, F_last, bar_width, color=ccolors["F_last"], label='F w/ Head+CE', edgecolor="black", hatch='//')
bars3 = ax1.bar(index + 3.5 * bar_width, B_1st, bar_width, color=ccolors["B_1st"], label='B w/o Head+CE')
bars4 = ax1.bar(index + 4.5 * bar_width, B_last, bar_width, color=ccolors["B_last"], label='B w/ Head+CE', edgecolor="black", hatch='\\\\')

# 在原始数据的柱状图上添加归一化后的数值
for bars, norm_values in zip([bars1, bars2, bars3, bars4], [F_1st_norm, F_last_norm, B_1st_norm, B_last_norm]):
    for bar, norm_value in zip(bars, norm_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height, f'x{norm_value:.2f}', ha='center', va='bottom', rotation=90, fontsize=data_value_font_size)

# 设置第一个子图的标题和标签
ax1.set_xlabel('Sequence Length', fontsize=xy_label_size)
ax1.set_ylabel('Time (ms)', fontsize=xy_label_size)
ax1.set_yscale('log')
ax1.set_ylim((0, 200))
ax1.set_title('Hidden Size = 4K', fontsize=title_size)
ax1.set_xticks(index + 3 * bar_width)
# ax1.tick_params(axis='x', labelsize=xy_tick_size-3)
# ax1.tick_params(axis='y', labelsize=xy_tick_size) 
ax1.set_xticklabels(x)
ax1.legend()
ax1.grid()

# 绘制 hiddensize 为 8K 的数据的柱状图
index_hidden = np.arange(len(x_hidden))
bars1_hidden = ax2.bar(index_hidden + 1.5 * bar_width, F_1st_hidden, bar_width, color=ccolors["F_1st"], label='F w/o Head+CE')
bars2_hidden = ax2.bar(index_hidden + 2.5 * bar_width, F_last_hidden, bar_width, color=ccolors["F_last"], label='F w/ Head+CE', edgecolor="black", hatch='//')
bars3_hidden = ax2.bar(index_hidden + 3.5 * bar_width, B_1st_hidden, bar_width, color=ccolors["B_1st"], label='B w/o Head+CE')
bars4_hidden = ax2.bar(index_hidden + 4.5 * bar_width, B_last_hidden, bar_width, color=ccolors["B_last"], label='B w/ Head+CE', edgecolor="black", hatch='\\\\')

# 在 hiddensize 为 8K 的数据的柱状图上添加归一化后的数值
for bars, norm_values in zip([bars1_hidden, bars2_hidden, bars3_hidden, bars4_hidden], [F_1st_hidden_norm, F_last_hidden_norm, B_1st_hidden_norm, B_last_hidden_norm]):
    for bar, norm_value in zip(bars, norm_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height, f'x{norm_value:.2f}', ha='center', va='bottom', rotation=90, fontsize=data_value_font_size)

# 设置第二个子图的标题和标签
ax2.set_xlabel('Sequence Length', fontsize=xy_label_size)
# ax2.set_ylabel('Time (ms)', fontsize=xy_label_size)
ax2.set_yscale('log')
ax2.set_ylim((0, 450))
ax2.set_title('Hidden Size = 8K', fontsize=title_size)
ax2.set_xticks(index_hidden + 3 * bar_width)
# ax2.tick_params(axis='x', labelsize=xy_tick_size-3)
# ax2.tick_params(axis='y', labelsize=xy_tick_size) 
ax2.set_xticklabels(x_hidden)
ax2.legend()
ax2.grid()
title = 'Computing times (ms) of F and B'
# plt.title(title)

# 调整布局并保存图像
plt.tight_layout()
plt.savefig("/Users/hanayukino/{}.svg".format(title), format='svg',dpi=200)
plt.show()