import matplotlib.pyplot as plt

def plot_line_chart(x_data, y_data, labels, title='Comparison of Different Schedules (pp=4, mb=8)', x_label='Max Activation Counts (Times of mb)', y_label='Theoretical Time Cost'):
    plt.figure(figsize=(10, 6))  # 图形大小
    for x, y, label in zip(x_data, y_data, labels):
        linestyle = '--' if 'UnifiedPP' in str(label) else '-'
        marker = 's' if 'UnifiedPP' in str(label) else 'o'
        plt.plot(x, y, marker=marker, linestyle=linestyle, label=label)  # 绘制折线
    
    plt.axhline(615, color='red',linewidth=0.5, ls='--')  # 添加水平线
    plt.text(x=x[0], y=615, s=f'Theoretical Minimal Time={615} with Chunk=2', color='red', fontsize=7.5, verticalalignment='bottom', horizontalalignment='left')
    
    plt.title(title)  # 图表标题
    plt.xlabel(x_label)  # X轴标签
    plt.ylabel(y_label)  # Y轴标签
    plt.grid(True)  # 添加网格
    plt.legend()  # 显示图例
    plt.show()  # 显示图表

# x_data = [1, 2, 3, 4, 5]  # X轴数据
# y_data = [2, 3, 5, 7, 11]  # Y轴数据
colors = [
    ["1F1B"],
    ["Interleaved-1F1B"],
    ["UnifiedPP-I1F1B"],
    ["ZBH1"],
    ["ZBV"],
    ["UnifiedPP-Chunk-2"],
    ["UnifiedPP-Chunk-4"],
    ["UnifiedPP-Chunk-6"],
    ["UnifiedPP-Chunk-12"],
]
labels = [
    ["1F1B"],
    ["Interleaved-1F1B"],
    ["UnifiedPP-I1F1B"],
    ["ZBH1"],
    ["ZBV"],
    ["UnifiedPP-Chunk-2"],
    ["UnifiedPP-Chunk-4"],
    ["UnifiedPP-Chunk-6"],
    ["UnifiedPP-Chunk-12"],
]
y_data = [
    [808, 808, 808, 808],  # 1F1B
    [798, 698, 660, 665],     # Interleaved-1F1B
    [656, 640, 628, 615],     # UnifiedPP-I1F1B
    [732, 714, 702, 678],     # ZBH1
    [658, 634, 630],     # ZBV
    [618, 615, 615],     # UnifiedPP-2   Vshape
    [609, 597, 597],     # UnifiedPP-4   Wavelike
    [625, 599, 592],     # UnifiedPP-6   Wavelike
    [595, 589],     # UnifiedPP-12  Wavelike
]
x_data = [
    [1,1.25,1.5,2],
    [1,1.25,1.5,2],
    [1,1.25,1.5,2],
    [1,1.25,1.5,2],
    [1,1.5,2],
    [1.25,1.5,2],
    [1.25,1.5,2],
    [1.25,1.5,2],
    [1.5,2],
]
# 绘制折线图
plot_line_chart(x_data, y_data, labels=labels)