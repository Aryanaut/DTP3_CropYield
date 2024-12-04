import numpy as np
from matplotlib import pyplot as plt

def normalize_z(array: np.ndarray, columns_means=None,
                columns_stds=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if columns_means is None:
        columns_means=np.mean(array,axis=0)
    if columns_stds is None:
        columns_stds=np.std(array,axis=0)
        
    out = (array-columns_means)/columns_stds
    
    return out, columns_means, columns_stds

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
        self.IND = ind
        self.ind = ind
        self.dep = dep
        self.ind_mean, self.ind_std, self.dep_mean, self.dep_std = None, None, None, None

        self.weights = np.zeros((self.ind.shape[1], 1))
    
        self.costs = []
        self.predictions = None
    
    def normalize_z(self, array: np.ndarray, columns_means=None,
                columns_stds=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if columns_means is None:
            columns_means=np.mean(array,axis=0)
        if columns_stds is None:
            columns_stds=np.std(array,axis=0)
            
        out = (array-columns_means)/columns_stds
        
        return out, columns_means, columns_stds

    def normalize_data(self):
        self.ind, self.ind_mean, self.ind_std = normalize_array(self.ind)
        self.dep, self.dep_mean, self.dep_std = normalize_array(self.dep)

    def _compute_cost(self, ind, weights):
        m = len(self.dep)
        predictions = np.dot(ind, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - self.dep) ** 2)
        return cost

    @property
    def get_biased_ind(self):
        return np.c_[np.ones(self.ind.shape[0]), self.ind]

    @property
    def get_dep_mean_std(self):
        return self.dep_mean, self.dep_std

    @property
    def get_ind_mean_std(self):
        return self.ind_mean, self.ind_std
    
    def gradient_descent(self):
        ind = np.c_[np.ones(self.ind.shape[0]), self.ind]
        weights = np.zeros((self.ind.shape[1], 1))
        m = len(self.dep)
        for _ in range(self.iterations):
            predictions = np.dot(ind, weights)
            errors = predictions - self.dep
            gradient = (1 / m) * np.dot(ind.T, errors)
            weights -= self.lr * gradient
            cost = self._compute_cost(ind, weights)
            self.costs.append(cost)

        self.weights = weights[1:4]
        self.predictions = predictions * self.dep_std + self.dep_mean