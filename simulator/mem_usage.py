pp4mb16_1f1b = [29.810272216796875, 23.747772216796875, 17.685272216796875, 14.446990966796875]
pp4mb16_Interleaved1f1b = [38.904022216796875, 32.841522216796875, 26.779022216796875, 23.540740966796875]
pp4mb16_zbv = [33.00959014892578, 30.18537139892578, 30.18537139892578, 32.811065673828125]
pp4mb16_upp = [38.65802764892578, 42.31037139892578, 45.34162139892578, 45.34162139892578]

import matplotlib.pyplot as plt
import numpy as np

# 数据
pp4mb16_1f1b = [29.810272216796875, 23.747772216796875, 17.685272216796875, 14.446990966796875]
pp4mb16_Interleaved1f1b = [38.904022216796875, 32.841522216796875, 26.779022216796875, 23.540740966796875]
pp4mb16_zbv = [33.00959014892578, 30.18537139892578, 30.18537139892578, 32.811065673828125]
pp4mb16_upp = [38.65802764892578, 42.31037139892578, 45.34162139892578, 45.34162139892578]
pp4mb16_upp_layerwise = [79.49858093261719, 64.62396240234375, 49.123069763183594, 50.998565673828125]  # 新增数据

# 设备数量
devices = np.arange(1, 5)

# 创建画布
plt.figure(figsize=(12, 7))

# 绘制折线图
plt.plot(devices, pp4mb16_1f1b, marker='o', label='1F1B')
plt.plot(devices, pp4mb16_Interleaved1f1b, marker='s', label='Interleaved1F1B')
plt.plot(devices, pp4mb16_zbv, marker='^', label='ZBV')
plt.plot(devices, pp4mb16_upp, marker='d', label='UPP')
plt.plot(devices, pp4mb16_upp_layerwise, marker='*', label='UPP Layerwise')  # 新增数据绘图

# 添加数据标签
for i, (a, b, c, d, e) in enumerate(zip(pp4mb16_1f1b, pp4mb16_Interleaved1f1b, pp4mb16_zbv, pp4mb16_upp, pp4mb16_upp_layerwise)):
    plt.text(i + 1, a, f'{a:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + 1, b, f'{b:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + 1, c, f'{c:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + 1, d, f'{d:.2f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + 1, e, f'{e:.2f}', ha='center', va='bottom', fontsize=9)  # 新增数据标签

# 设置标题和标签
plt.title('Memory Utilization Comparison of Different Schedule Algorithms', fontsize=14)
plt.xlabel('Device ID', fontsize=12)
plt.ylabel('Memory Overhead (GB)', fontsize=12)

# 设置x轴刻度
plt.xticks(devices)

# 添加图例
plt.legend()

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图表
plt.show()