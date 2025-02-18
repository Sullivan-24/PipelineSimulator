ofob = [
    [60.061859130859375, 47.936859130859375, 35.811859130859375, 27.217132568359375],
    [238.12371826171875 - (49-0.5) * 3, 189.62371826171875 - (49-0.5) * 2, 141.12371826171875 - (49-0.5) * 1, 99.68426513671875]
]
afab = [
    [108.56185913085938, 108.56185913085938, 108.56185913085938, 131.86166381835938],
    [432.12371826171875 - (49-0.5) * 7, 432.12371826171875 - (49-0.5) * 7, 432.12371826171875 - (49-0.5) * 7, 478.72332763671875 - (49-0.5) * 7],
]
zbv = [
    [64.34233093261719, 60.81205749511719, 60.81205749511719, 66.06344604492188], # 4+4+wo
    [106.43505859375, 99.62451171875, 99.87451171875, 100.12451171875], # 8+8+w
]
upp = [
    [78.54985809326172, 79.3746566772461, 79.3746566772461, 79.3746566772461], # 4+4+wo
    [75.28622436523438, 74.12411499023438, 73.99911499023438, 74.12411499023438], # 8+8+w
]
import matplotlib.pyplot as plt
import numpy as np

# 数据
ofob4kseq4khid = [60.061859130859375, 47.936859130859375, 35.811859130859375, 27.217132568359375]
ofob8kseq8khid = [238.12371826171875 - (49-0.5) * 3, 189.62371826171875 - (49-0.5) * 2, 141.12371826171875 - (49-0.5) * 1, 99.68426513671875]

afab4kseq4khid = [108.56185913085938, 108.56185913085938, 108.56185913085938, 131.86166381835938]
afab8kseq8khid = [432.12371826171875 - (49-0.5) * 7, 432.12371826171875 - (49-0.5) * 7, 432.12371826171875 - (49-0.5) * 7, 478.72332763671875 - (49-0.5) * 7]

zbv4kseq4khid = [64.34233093261719, 60.81205749511719, 60.81205749511719, 66.06344604492188]
zbv8kseq8khid = [106.43505859375, 99.62451171875, 99.87451171875, 100.12451171875]

upp4kseq4khid = [78.54985809326172, 79.3746566772461, 79.3746566772461, 79.3746566772461]
upp8kseq8khid = [75.28622436523438, 74.12411499023438, 73.99911499023438, 74.12411499023438]

# 颜色配置
colors = {
    "UPP": '#D62728',
    "1F1B": '#1f77b4',
    "OFOB": '#1f77b4',
    "AFAB": 'grey',
    "I1F1B": '#ff7f0e',
    "ZBV": '#2ca02c',
    "ZBH": '#e377c2',
}
title_size = 24
xy_label_size = 16.5
xy_tick_size = 12
value_size = 12
# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
x_labels = [f"Device {i+1}" for i in range(len(upp8kseq8khid))]

# 设置全局标题
title = 'Required Peak Memory of Different Schedules'
# fig.suptitle(title, fontsize=title_size)

# 绘制 4kseq 4khid 数据
ax1.plot(ofob4kseq4khid, label='1F1B', color=colors['OFOB'], marker='o')
ax1.plot(afab4kseq4khid, label='AFAB', color=colors['AFAB'], marker='s')
ax1.plot(zbv4kseq4khid, label='ZBV', color=colors['ZBV'], marker='^')
ax1.plot(upp4kseq4khid, label='UPP (Ours)', color=colors['UPP'], marker='*')
# ax1.set_title('SEQ_LEN=4K HID_SIZE=4K W/O Recomp.', fontsize=xy_label_size)
ax1.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
ax1.set_xticks(np.arange(4))
ax1.set_xticklabels(x_labels, fontsize=xy_tick_size)
ax1.set_xlabel("SEQ_LEN=4K HID_SIZE=4K W/O Recomp.", fontsize=xy_label_size)

ax1.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='y', labelsize=xy_tick_size)
# 添加横线表示 GPU_MAX_MEMORY
ax1.axhline(y=80, color='r', linestyle='--', label='GPU Max Memory')
# ax1.legend(fontsize=12)
ax1.set_yticks(range(20,140,10))

for data in [ofob4kseq4khid, afab4kseq4khid, zbv4kseq4khid, upp4kseq4khid]:
    for i, v in enumerate(data):
        t = f'{v:.2f}'
        if v > 80:
            t += f'\nOOM'
        if 0 < i < 3:
            ax1.text(i, v + 1, t, ha='center', fontsize=value_size)
        elif i == 0:
            ax1.text(i, v + 1, t, ha='left', fontsize=value_size)
        else:
            ax1.text(i, v - 4, t, ha='right', fontsize=value_size)

# 绘制 8kseq 8khid 数据
ax2.plot(ofob8kseq8khid, label='1F1B', color=colors['OFOB'], marker='o')
ax2.plot(afab8kseq8khid, label='AFAB', color=colors['AFAB'], marker='s')
ax2.plot(zbv8kseq8khid, label='ZBV', color=colors['ZBV'], marker='^')
ax2.plot(upp8kseq8khid, label='UPP (Ours)', color=colors['UPP'], marker='*')
# ax2.set_title('SEQ_LEN=8K HID_SIZE=8K W/ Recomp.', fontsize=xy_label_size)
ax2.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
ax2.set_xticks(np.arange(4))
ax2.set_xticklabels(x_labels, fontsize=xy_tick_size)
ax2.set_xlabel("SEQ_LEN=8K HID_SIZE=8K W/ Recomp.", fontsize=xy_label_size)

ax2.grid(True, linestyle='--', alpha=0.6)
ax2.tick_params(axis='y', labelsize=xy_tick_size)
# 添加横线表示 GPU_MAX_MEMORY
ax2.axhline(y=80, color='r', linestyle='--', label='GPU Max Memory')
ax2.legend(fontsize=12)
ax2.set_yticks(range(70,141,10))


for did, data in enumerate([ofob8kseq8khid, afab8kseq8khid, zbv8kseq8khid, upp8kseq8khid]):
    for i, v in enumerate(data):
        t = f'{v:.2f}'
        if v > 80:
            t += f'\nOOM'
        if 0 < i < 3:
            ax2.text(i, v + 1, t, ha='center', fontsize=value_size)
        elif i == 0:
            ax2.text(i, v + 1, t, ha='left', fontsize=value_size)
        else:
            down_value = 0
            if did == 1:
                down_value = 4
            elif did == 0:
                down_value = 4
            elif did == 2:
                down_value = -2
            ax2.text(i, v - down_value, t, ha='right', fontsize=value_size)

# 调整布局
plt.tight_layout()
plt.savefig(f"/Users/hanayukino/{title}.svg",format='svg',dpi=200)
# 显示图表
plt.show()

