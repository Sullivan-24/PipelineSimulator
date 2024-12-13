import matplotlib.pyplot as plt

def plot_line_chart(x_data, y_data, title='折线图', x_label='X轴', y_label='Y轴'):
    plt.figure(figsize=(10, 6))  # 图形大小
    plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label='Time')  # 绘制折线
    plt.title(title)  # 图表标题
    plt.xlabel(x_label)  # X轴标签
    plt.ylabel(y_label)  # Y轴标签
    plt.grid(True)  # 添加网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图表

x_data = [1, 2, 3, 4, 5]  # X轴数据
y_data = [2, 3, 5, 7, 11]  # Y轴数据
[
    [808, 808, 808, 808],  # 1F1B
    [798, 698, 660, 665],     # Interleaved-1F1B
    [656, 640, 628, 615],     # UnifiedPP-I1F1B
    [732, 714, 702, 678],     # ZBH1
    [658, 634, 634, 630],     # ZBV
    [618, 615, 615],     # UnifiedPP-2   Vshape
    [609, 597, 597],     # UnifiedPP-4   Wavelike
    [625, 599, 592],     # UnifiedPP-6   Wavelike
    [595, 589],     # UnifiedPP-12  Wavelike
]
[
    [],[],[],[],[],[],[],[]
]
# 绘制折线图
plot_line_chart(x_data, y_data, title='', x_label='', y_label='')