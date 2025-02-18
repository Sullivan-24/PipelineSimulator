import matplotlib.pyplot as plt
import numpy as np

ccolors = {
    "F_1st":'#3299FF',
    "B_1st":'#00FFFF',
    "F_last":'#3299FF',
    "B_last":'#00FFFF',
    "emb":'#FF0000',
    "head":'#4C0099',
    "ce":'#FF8000',
}

# 数据
x = ["1k", "2k", "4k", "8k", "16k"]
F_1st = [3.142, 5.694, 12.441, 37.468, 115.097]
B_1st = [5.667, 10.031, 21.238, 71.181, 223.768]
F_last = [8.939, 12.068, 22.496, 58.208, 159.809]
B_last = [8.444, 16.524, 41.99, 101.699, 287.042]
emb = [0.114, 0.270, 0.521, 1.054, 2.083]
head = [1.586, 3.078, 5.982, 12.011, 26.635]
ce = [1.365, 2.653, 5.279, 10.774, 21.368]

# 归一化函数
def normalize(data, ref):
    return [d / r for d, r in zip(data, ref)]

# 归一化数据
F_1st_norm = normalize(F_1st, F_1st)
B_1st_norm = normalize(B_1st, F_1st)
F_last_norm = normalize(F_last, F_1st)
B_last_norm = normalize(B_last, F_1st)

# 设置柱状图的宽度
bar_width = 0.15

# 设置x轴的位置
index = np.arange(len(x))
plt.figure(figsize=(5, 4))

# 绘制柱状图
bars1 = plt.bar(index + 1.5 * bar_width, F_1st, bar_width, color=ccolors["F_1st"], label='F w/o Head+CE')
bars2 = plt.bar(index + 2.5 * bar_width, F_last, bar_width, color=ccolors["F_last"], label='F w/ Head+CE', edgecolor="black", hatch='//')
bars3 = plt.bar(index + 3.5 * bar_width, B_1st, bar_width, color=ccolors["B_1st"], label='B w/o Head+CE')
bars4 = plt.bar(index + 4.5 * bar_width, B_last, bar_width, color=ccolors["B_last"], label='B w/ Head+CE', edgecolor="black", hatch='\\\\')

# 在每个柱状图上添加归一化后的数值
for bars, norm_values in zip([bars1, bars2, bars3, bars4], [F_1st_norm, F_last_norm, B_1st_norm, B_last_norm]):
    for bar, norm_value in zip(bars, norm_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'x{norm_value:.2f}', ha='center', va='bottom', rotation=90, fontsize=8)

title = 'Computing times (ms) of F and B'
# 添加x轴标签和标题
plt.xlabel('Sequence Length')
plt.ylabel('Time (ms)')
plt.yscale('log')
plt.ylim((0,600))
plt.title(title)
plt.xticks(index + 3 * bar_width, x)

# 添加图例
plt.legend()
plt.grid()

# 显示图形
plt.savefig("/Users/hanayukino/{}.svg".format(title), format='svg',dpi=200)
plt.show()