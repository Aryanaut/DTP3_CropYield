import numpy as np
from matplotlib import pyplot as plt

def normalize_array(array):
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    return (array - mean) / std, mean, std

class MVF:

    # ind: independant variable matrix, np array, size = no. of data points * no. of dependant variables
    # dep: depdendant variable matrix, size = no. of data points * 1

    def __init__(self, ind, dep):
    
        self.lr = 0.01
        self.iterations = 1000
        self.ind = ind
        self.dep = dep
        self.ind_mean, self.ind_std, self.dep_mean, self.dep_std = None, None, None, None

        self.weights = np.zeros((self.ind.shape[1], 1))
        
        self.feature_names = []
        self.target_names = None
    
        self.costs = []
        self.predictions = None

    def normalize_data(self):
        self.ind, self.ind_mean, self.ind_std = normalize_array(self.ind)
        self.dep, self.dep_mean, self.dep_std = normalize_array(self.dep)

    def _compute_cost(self, ind, weights, lambda_reg=0.1):
        m = len(self.dep)
        predictions = np.dot(ind, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - self.dep) ** 2)
        reg_term = (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2) 
        return cost + reg_term

    @property
    def get_biased_ind(self):
        return np.c_[np.ones(self.ind.shape[0]), self.ind]

    @property
    def get_dep_mean_std(self):
        return self.dep_mean, self.dep_std

    @property
    def get_ind_mean_std(self):
        return self.ind_mean, self.ind_std

    @property
    def get_ind(self):
        return self.ind

    @property
    def get_dep(self):
        return self.dep
    
    def gradient_descent(self, lambda_reg=0.1, tol=1e-6):
        ind = np.c_[np.ones(self.ind.shape[0]), self.ind]
        weights = np.zeros((ind.shape[1], 1))
        m = len(self.dep)
        
        for _ in range(self.iterations):
            predictions = np.dot(ind, weights)
            errors = predictions - self.dep
            gradient = (1 / m) * np.dot(ind.T, errors)
            weights -= self.lr * gradient
            cost = self._compute_cost(ind, weights)
            self.costs.append(cost)

            if len(self.costs) > 1 and abs(costs[-1] - costs[-2]) < tol: 
                print("Convergence found")
                break

        self.weights = weights
        self.predictions = predictions * self.dep_std + self.dep_mean