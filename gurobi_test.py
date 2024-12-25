from gurobipy import Model, GRB

# 创建模型
model = Model()

# 定义变量
n = 10  # 数组长度
x = model.addVars(n, vtype=GRB.BINARY, name='x')

# 添加约束：确保数组中有 5 个 1
model.addConstr(sum(x[i] for i in range(n)) == 5, "num_ones")

# 添加连续性约束
# 使用单一的起始点和结束点来定义连续区间
for start in range(n - 4):  # start 必须保证有足够的空间放 5 个 1
    model.addConstrs((x[i] == 1 for i in range(start, start + 5)), f"segment_{start}")

# 目标函数：最大化 1 的个数（这里固定为 5，所以目标函数可以是任意的）
model.setObjective(sum(x[i] for i in range(n)), GRB.MAXIMIZE)

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    for i in range(n):
        print(f"x[{i}] = {x[i].X}")
else:
    print("无解")