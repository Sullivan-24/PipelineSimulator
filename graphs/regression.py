import numpy as np
from scipy.optimize import curve_fit
from e2e_data import solving_time

def exp_func(x, a, b):
    return a * np.exp(b * x)

def update_solving_time(): 
    for k,v in solving_time.items():

        for method, values in solving_time[k].items():
            x = list(solving_time[k][method].keys())
            y = list(solving_time[k][method].values())
            if len(x) == 0 or len(y) == 0 or len(x) != len(y):
                continue
            params, _ = curve_fit(exp_func, x, y, p0=(0.01, 0.5))
            a, b = params
            for i in [8, 16, 32, 64]:
                if i not in x:
                    yi = exp_func(i, a, b)
                    solving_time[k][method][i] = float(yi)
    return solving_time