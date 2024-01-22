import numpy as np

def rmse(predictions, targets):
   
    pred = np.array(predictions)
    tar = np.array(targets)
    
    # Calculate the squared differences
    squared_diff = (pred - tar) ** 2
    
    # Calculate the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Calculate the square root of the mean squared differences
    rmse = np.sqrt(mean_squared_diff)
    
    return rmse
