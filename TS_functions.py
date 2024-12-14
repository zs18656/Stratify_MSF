import numpy as np
from functools import reduce
import numpy as np
import json

def load_basicts_data(dataset, idx = -1, folder = 'BasicTS_Data/'):
    folder = 'BasicTS_Data/'
    dataset += '/'
    with open(folder+dataset+'desc.json', 'r') as file:
        data_desc = json.load(file)
    print(data_desc['feature_description'])
    data = np.memmap(folder+dataset+'data.dat', dtype='float32', mode='r', shape=tuple([x for x in data_desc['shape']]))
    if idx == -1:
        return data
    else:
        print(f"selected {data_desc['feature_description'][idx]}")
        return data[:, :, idx] 

def create_sliding_windows(data, lag, horizon):
    X, Y = [], []
    
    for i in range(len(data) - lag - horizon + 1):
        # Create the lagged window for X
        X_window = data[i:(i + lag)]
        
        # Create the H-ahead target for Y
        Y_window = data[(i + lag):(i + lag + horizon)]
        
        X.append(X_window)
        Y.append(Y_window)
    
    return np.array(X), np.array(Y)

def factors(n):    
    return sorted(list(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))))

def getMackey(N1, noise = 0):
    N = N1 + 100
    b   = 0.1
    c   = 0.2
    tau = 17
    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
         1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17,N+99):
        y.append(y[n] - b*y[n] + c*y[n-tau]/(1+y[n-tau]**10))
    y = y[100:] 

    return 10*np.diff(y[:N1]) + np.random.normal(0, noise, np.diff(y[:N1]).shape)


def moving_window(lst, window_size, h):
    result_window = []
    for i in range(len(lst) - window_size - h):
        result_window.append(lst[i:i+window_size])
    return np.array(result_window)

def shiftData(time_series, h):
    original = time_series[:, 0]
    t_list, t_plus_h_list = [], []
    for t in range(len(time_series) - 2 * h):
        t_list.append(time_series[t])
        t_plus_h_list.append(original[t+h:t+2*h])
    return np.array(t_list), np.array(t_plus_h_list)
