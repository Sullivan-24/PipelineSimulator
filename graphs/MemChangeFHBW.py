import sys
sys.path.append(".")
from simulator.config import *
import matplotlib.pyplot as plt
import seaborn as sns

def sinplot():
    title_size = 24
    xy_label_size = 16
    xy_tick_size = 16
    # 数据
    ywo = [0, Activation.FULL, Activation.FULL + Activation.LOSS, Activation.FULL, Activation.FULL + Gradient.INPUT, Activation.FULL + Gradient.INPUT + Gradient.PARAMETER, 0]
    yw = [0, Activation.INPUT, Activation.INPUT + Activation.LOSS, Activation.INPUT, Activation.FULL + Gradient.INPUT, Activation.FULL + Gradient.INPUT + Gradient.PARAMETER, 0]
    x = range(len(ywo))
    x = ["F0","F1-H0","H","H1-B0","B1-W0","W","W1"]
    # x = ["F0","F1-H0","H0.5","H1-B0","B1-W0","W0.5","W1"]
    awo = [0, 0, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, 0]
    xawo = [0, 1, 1, 2, 3, 4, 5, 6, 6]
    ao = [0, 0, 0, 0, Activation.FULL, Activation.FULL, Activation.FULL, Activation.FULL, 0]
    xao = [0, 1, 2, 3, 3, 4, 5, 6, 6]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    title = "Memory Usage Changes While Computing F-H-B-W"
    # fig.suptitle(title, fontsize=title_size, y=0.950)

    # 第一张子图：Memory Usage
    ax1.plot(x, ywo, marker='^', linestyle='dashdot', c='blue', label="Layer-wise w/o Recomp.")
    ax1.plot(x, yw, marker='*', linestyle='--', c='red', label="Layer-wise w/ Recomp.")
    ax1.plot(xawo, awo, marker='^', linestyle='dashdot', c='#006600', alpha=0.8, label="Stage-wise w/o Recomp.")
    ax1.plot(xao, ao, marker='*', linestyle='dashed', c='#FF8000', alpha=0.8, label="Stage-wise w/ Recomp.")
    ax1.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
    ax1.set_title("(Sequence len., Hidden size) = (4K, 4K)", fontsize=xy_label_size)
    # ax1.set_xlabel("Seq = 4K, Hid = 4K", fontsize=xy_label_size)
    # ax1.legend(fontsize=12)
    # ax1.grid(True)
    # ax1.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])
    ax1.tick_params(axis='x', labelsize=xy_tick_size-3)
    ax1.tick_params(axis='y', labelsize=xy_tick_size) 

    # ax1.set_xticks([]) 

    s = 4*K
    h = 8*K

    full = (34*b*s*h + 5*b*s*s*a)/G
    loss = (2*2*b*s*v)/G
    act_input = (2*b*s*h)/G
    gradient_input = DATA_TYPE * (12 * h * h + 13 * h) / G
    gradient_para = DATA_TYPE * (12 * h * h + 13 * h) / G
    head_para = DATA_TYPE * v * h / G

    ywo = [0, full, full + loss, full, full + gradient_input, full + gradient_input + gradient_para, 0]
    yw = [0, act_input, act_input + loss, act_input, full + gradient_input, full + gradient_input + gradient_para, 0]
    x = range(len(ywo))
    x = ["F0","F1-H0","H","H1-B0","B1-W0","W","W1"]

    awo = [0, 0, full, full, full, full, full, full, 0]
    xawo = [0, 1, 1, 2, 3, 4, 5, 6, 6]
    ao = [0, 0, 0, 0, full, full, full, full, 0]
    xao = [0, 1, 2, 3, 3, 4, 5, 6, 6]
    ax2.plot(x, ywo, marker='^', linestyle='dashdot', c='blue', label="Layer-wise w/o Recomp.")
    ax2.plot(x, yw, marker='*', linestyle='--', c='red', label="Layer-wise w/ Recomp.")
    ax2.plot(xawo, awo, marker='^', linestyle='dashdot', c='#006600', alpha=0.8, label="Stage-wise w/o Recomp.")
    ax2.plot(xao, ao, marker='*', linestyle='dashed', c='#FF8000', alpha=0.8, label="Stage-wise w/ Recomp.")
    ax2.set_ylabel("Memory Usage (GB)", fontsize=xy_label_size)
    ax2.set_title("(Sequence len., Hidden size) = (4K, 8K)", fontsize=xy_label_size)
    # ax2.set_xlabel("Seq = 8K, Hid = 8K", fontsize=xy_label_size)
    ax2.legend(fontsize=10)
    # ax2.set_yticks([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7])
    # ax2.set_yticks([0,1.0,2,3,4,5,6,7])
    # ax2.grid(True)

    ax2.tick_params(axis='x', labelsize=xy_tick_size-3)
    ax2.tick_params(axis='y', labelsize=xy_tick_size) 
# 调整布局
plt.tight_layout()


sns.set_style("white")
sinplot() # 默认无参数状态，就是删除上方和右方的边框
sns.despine()

# plt.savefig(f"/Users/hanayukino/{title}.svg",format='svg',dpi=200)
# 显示图形
plt.show()