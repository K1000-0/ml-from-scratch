import numpy as np

def make_linear_data(n = 100, noise = 5.0):
    x = 2 * np.random.rand(n, 1) # create a tuple of size (n, 1) that contains random number from 0 to 2
    y = 4 + 3 * x[:, 0] + noise  * np.random.randn(n)
    return x,y 

