import os
import heapq
import glob
import pickle

def load_and_validate(filepath: str) -> list:
    """加载并验证临时文件"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            return [
                (-t, uid, res, pl) 
                for t, uid, res, pl in data.get('results', [])
            ]
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return []
    
# def load_and_validate(filepath: str) -> list:
#     """加载并验证临时文件"""
#     if not os.path.exists(filepath):
#         return []
    
#     try:
#         with open(filepath, 'rb') as f:
#             data = pickle.load(f)
#             return [(-t, res, pl) for t, res, pl in data.get('results', [])]
#     except Exception as e:
#         print(f"Error loading {filepath}: {str(e)}")
#         return []

def main(output_file: str = "global_top10.txt"):
    # 加载所有节点结果
    all_results = []
    for fpath in glob.glob("temp_node_*.pkl"):
        all_results.extend(load_and_validate(fpath))
    
    if not all_results:
        print("No valid results found!")
        return
    
    # # 提取实际时间成本并排序
    # valid_results = []
    # for neg_t, res, pl in all_results:
    #     try:
    #         valid_results.append((-neg_t, res, pl))
    #     except:
    #         continue
    
    # # 获取全局前10
    # top10 = heapq.nsmallest(10, valid_results, key=lambda x: x[0])
    # 提取实际时间成本并排序（忽略UID）
    valid_results = []
    for t, uid, res, pl in all_results:
        try:
            valid_results.append( (t, res, pl) )  # 丢弃UID
        except:
            continue
    
    # 按时间成本排序
    top10 = sorted(valid_results, key=lambda x: x[0])[:10]

    # 保存结果
    with open(output_file, 'w') as f:
        for idx, (t, res, pl) in enumerate(top10, 1):
            f.write(f"Rank {idx} | Time: {t:.4f}\n")
            f.write(f"Placement: {pl}\n")
            f.write(f"Results: {res}\n{'='*40}\n")
    
    # 清理临时文件
    for fpath in glob.glob("temp_node_*.pkl"):
        try:
            os.remove(fpath)
        except:
            pass
    print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    main()