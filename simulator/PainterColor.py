from .abstract.mutils import *
def set_color(sid, workload_type, layer_num):
    color = None
    if workload_type == 'f':    #颜色设置，加上w的情况
        color = "#00AFFF"
    elif workload_type == 'b':
        color = "#00FFFF" 
    else:
        color = "#00FF6F"

    if RUN_MODE == RunMode.LAYERWISE_GUROBI_SOLVE or LAYERWISE:
        if sid == 0:
            color = "#FF0000" # 红色: #FF0000
        elif sid == layer_num - 2:
            if workload_type == 'f':
                color = "#800080" # 紫色: #800080
            elif workload_type == 'b':
                color = "#EE82EE"
            else:
                color = "#FF00FF"
        elif sid == layer_num - 1:
            if workload_type == 'f':
                color = "#FFA500" # 橙色: #FFA500
            elif workload_type == 'b':
                color = "#FF4500"
            else:
                color = "#FF7F50"
    return color