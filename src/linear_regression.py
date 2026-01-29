import numpy as np

class LinearRegressionClosedForm: # class of linear regresssion that use the normal equation to minimize the weight
    def fit(self, x, y):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        self.w = np.linalg.inv(np.transpose(x_b) @ x_b) @ np.transpose(x_b) @ y

    def predict(self, x):
        x_b = np.c_[np.ones((x.shape[0], 1)), x]
        return x_b @ self.w