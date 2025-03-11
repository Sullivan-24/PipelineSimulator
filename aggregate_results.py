import os
import heapq
import glob
import pickle

def load_temp_files(pattern: str = "temp_node_*.bin"):
    """加载所有临时结果文件"""
    all_results = []
    
    for filename in glob.glob(pattern):
        with open(filename, 'rb') as f:
            node_results = pickle.load(f)
            # 转换回正时间成本并扩展
            for neg_t, res, pl in node_results:
                all_results.append((-neg_t, res, pl))
    
    return all_results

def save_final_top10(results: list, output_file: str = "global_top10.txt"):
    """保存最终结果"""
    top10 = heapq.nsmallest(10, results, key=lambda x: x[0])
    
    with open(output_file, 'w') as f:
        for t, res, pl in top10:
            f.write(f"Time: {t}\nResult: {res}\nPlacement: {pl}\n\n")
    print(f"Saved top10 results to {output_file}")

if __name__ == "__main__":
    # 聚合结果
    combined = load_temp_files()
    
    # 保存最终结果
    save_final_top10(combined)
    
    # 清理临时文件
    for f in glob.glob("temp_node_*.bin"):
        os.remove(f)
    print("Cleaned temporary files")