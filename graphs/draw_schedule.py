import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

# 模拟参数（你需要根据真实值填入）
canvas_width = 1000
canvas_height = 400
device_size = 4
pp_align = 0
pp_height = 40

# 模拟输入数据结构
data = {
    "f_0_0_0": 50,
    "r_1_1_0": 200,
    "b_2_0_1": 300,
    "w_3_2_1": 450,
    # 更多数据...
}

# 模拟 block 长度（假设每个阶段固定）
forward_length = [50, 50, 50, 50]
backward_b_length = [30, 30, 30, 30]
backward_w_length = [20, 20, 20, 20]

# 控制是否显示文字
SHOW_WORKLOAD_TEXT = True

# 显示颜色（可替换为你原来的 set_color 函数）
def set_color(pid, workload_type, layer_num):
    color_map = {
        'f': '#87CEFA',   # Light Blue
        'r': '#00CED1',   # Dark Cyan
        'b': '#FFA07A',   # Light Salmon
        'w': '#CD5C5C'    # Indian Red
    }
    return color_map.get(workload_type, '#D3D3D3')

# 解析 microbatch key
def parse_microbatch_key(key):
    k, mid, pid, did = key.split("_")
    return k, int(pid), int(mid), int(did)

# 开始绘图
fig, ax = plt.subplots(figsize=(12, 6))

# 画每个 device 的 timeline 背景
for pid in range(device_size):
    y0 = (pp_height + pp_align) * pid + 5
    y1 = y0 + pp_height
    rect = patches.Rectangle((pp_align, y0), canvas_width - 2 * pp_align, pp_height,
                             linewidth=1, edgecolor='black', facecolor='white')
    ax.add_patch(rect)

# 画每个 microbatch block
for microbatch_key, offset in data.items():
    k, pid, mid, did = parse_microbatch_key(microbatch_key)

    x0 = pp_align + offset
    y0 = (pp_height + pp_align) * did + 5

    if k == 'f' or k == 'r':
        width = forward_length[pid]
    elif k == 'b':
        width = backward_b_length[pid]
    elif k == 'w':
        width = backward_w_length[pid]
    else:
        width = 10

    x1 = x0 + width
    y1 = y0 + pp_height

    color = set_color(pid, k, device_size)
    block = patches.Rectangle((x0, y0), width, pp_height,
                              linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(block)

    if SHOW_WORKLOAD_TEXT:
        ax.text(x0 + width / 2, y0 + pp_height / 2,
                str(mid),
                ha='center', va='center', fontsize=8)

# 图形美化
ax.set_xlim(0, canvas_width)
ax.set_ylim(0, canvas_height)
ax.axis('off')

# 保存为 PDF
with PdfPages('/Users/hanayukino/pipeline_schedule.pdf') as pdf:
    pdf.savefig(fig, bbox_inches='tight')

plt.show()

plt.close(fig)
print("Saved to pipeline_schedule.pdf")
