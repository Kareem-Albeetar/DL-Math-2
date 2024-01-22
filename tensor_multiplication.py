import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    dimensions is a tuple of integers.
    """
    return torch.ones(dimensions) * val

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    Note that the dimensions of A and B should be the same.
    """
    return A * B

def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. (sum {x_i * w_i})
    """
    return torch.matmul(X, W.T)

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. (sum {x_i * w_i}) and add the bias.
    """
    return torch.matmul(X, W.T) + b

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    """
    return torch.heaviside(sum_total, torch.tensor(0.0))

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    return calculate_activation(sum_total)
