#!/bin/bash
#SBATCH --job-name=distributed_scheduling
#SBATCH --nodes=4                # 总节点数
#SBATCH --ntasks-per-node=1      # 每个节点1个任务
#SBATCH --cpus-per-task=32       # 每个任务使用32线程
#SBATCH --time=02:00:00          # 最大运行时间

# 加载Python环境
module load python/3.9

# 运行计算任务
srun python UnifiedSchedule.py \
    --node-id $SLURM_NODEID \
    --total-nodes $SLURM_JOB_NUM_NODES

# 聚合结果（只在最后一个节点执行）
if [ $SLURM_NODEID -eq $((SLURM_JOB_NUM_NODES-1)) ]; then
    python aggregate_results.py
fi