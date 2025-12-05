import numpy as np

def hampel_filter(input_series, K=3, n_sigmas=3):
    """
    Hampel Filter for outlier removal.
    K: Window size (radius). Total window width is 2*K+1.
    n_sigmas: Number of standard deviations for threshold.
    """
    n = len(input_series)
    new_series = list(input_series) # Make a copy
    k_const = 1.4826 # scale factor for Gaussian distribution
    
    # For each point in the series (handling boundaries simply by skipping or clamping)
    # Here we skip the first K and last K points for simplicity
    for i in range(K, n - K):
        window = input_series[(i - K):(i + K + 1)]
        x0 = np.median(window)
        S0 = k_const * np.median(np.abs(np.array(window) - x0))
        
        if np.abs(input_series[i] - x0) > n_sigmas * S0:
            new_series[i] = x0
            
    return new_series

def normalize_csi(csi_data):
    """
    Min-Max normalization to [0, 1]
    """
    csi_np = np.array(csi_data)
    min_val = np.min(csi_np)
    max_val = np.max(csi_np)
    if max_val - min_val == 0:
        return csi_np.tolist()
    return ((csi_np - min_val) / (max_val - min_val)).tolist()
