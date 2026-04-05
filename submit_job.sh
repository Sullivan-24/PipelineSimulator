#!/bin/bash
#SBATCH --partition=llm_s
#SBATCH --job-name=distributed_scheduling
#SBATCH --nodes=1                # 总节点数
#SBATCH --ntasks-per-node=1      # 每个节点1个任务
#SBATCH --cpus-per-task=32       # 每个任务使用32线程
#SBATCH --gpus-per-task=0        # 不使用GPU
#SBATCH --time=02:00:00          # 最大运行时间
#SBATCH --output=/mnt/petrelfs/guojihu/AutoPipelineGenerate/sbatch_output/output_%j.log  # 输出文件路径

# # 加载必要环境
# module purge
# module load python/3.9 gcc/9.3.0

# 创建虚拟环境（按需启用）
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt

# 运行分布式计算
echo "Starting node $SLURM_NODEID"
srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    python UnifiedSchedule.py \
    --node-id $SLURM_NODEID \
    --total-nodes $SLURM_JOB_NUM_NODES

# 在最后一个节点聚合结果
if [ "$SLURM_NODEID" -eq "$((SLURM_JOB_NUM_NODES-1))" ]; then
    echo "Aggregating results on node $SLURM_NODEID"
    python aggregate_results.py
fi

# 清理虚拟环境（按需启用）
# deactivate