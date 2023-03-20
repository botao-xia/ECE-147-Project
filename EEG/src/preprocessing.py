import numpy as np

def mean_max_subsampling(X, y, trim_len=500, sub_sample=2, noise=True):
    """
    Preprocessing the data by
    1. Trim the time series to (sample, 22, trim_len)
    2. Concate the augmented time series
        - max pool with stride and kernel_size = sub_sample
        - average pool with stride and kernel_size = sub_sample
        - subsample of the time series
    
    """
    total_X = None
    total_y = None

    X = X[:,:,0:trim_len]

    X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
    total_X = X_max
    total_y = y

    X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, sub_sample),axis=3)
    X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)  
    total_X = np.vstack((total_X, X_average))
    total_y = np.hstack((total_y, y))

    print(total_X.shape)

    for i in range(sub_sample):

        X_subsample = X[:, :, i::sub_sample] + \
                            (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
            
        total_X = np.vstack((total_X, X_subsample))
        total_y = np.hstack((total_y, y))
    
    return total_X, total_y