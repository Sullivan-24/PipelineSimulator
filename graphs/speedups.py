from e2e_data import h200_data
def calculate_avg_speedup(data):
    results = {}
    for model in data:
        model_results = {}
        for seq in ["Seq_2k", "Seq_4k"]:
            seq_data = data[model][seq]
            methods = ["S-1F1B", "I-1F1B", "ZB", "Mist"]
            speedups = {method: [] for method in methods}
            
            for config in seq_data:
                octopipe = seq_data[config]["OctoPipe"]
                # octopipe = seq_data[config]["Mist"]
                for method in methods:
                    if method in seq_data[config]:
                        speedups[method].append(octopipe / seq_data[config][method])
            print(seq, speedups)
            avg_speedup = {method: sum(values)/len(values) for method, values in speedups.items()}
            model_results[seq] = avg_speedup
        results[model] = model_results
    return results

# 输出结果
for model, seq_data in calculate_avg_speedup(h200_data).items():
    print(f"\n{model}模型:")
    for seq, methods in seq_data.items():
        print(f"{seq}:")
        for method, ratio in methods.items():
            print(f"  vs {method}: {ratio:.2f}")