import numpy as np

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
    return (array - mean) / std

class MVF:

    # ind: independant variable matrix, np array, size = no. of data points * no. of dependant variables
    # dep: depdendant variable matrix, size = no. of data points * 1

    def __init__(self, ind, dep):
    
        self.lr = 0.01
        self.iterations = 1000
        self.ind = ind
        self.dep = dep
        self.weights = np.zeros((self.ind.shape[1], 1))
    
        self.costs = []
    
    def normalize_z(self, array: np.ndarray, columns_means=None,
                columns_stds=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if columns_means is None:
            columns_means=np.mean(array,axis=0)
        if columns_stds is None:
            columns_stds=np.std(array,axis=0)
            
        out = (array-columns_means)/columns_stds
        
        return out, columns_means, columns_stds

    def normalize_data(self):
        self.ind = normalize_array(self.ind)
        self.dep = normalize_array(self.dep)

    def _compute_cost(self, ind, weights):
        m = len(self.dep)
        predictions = np.dot(ind, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - self.dep) ** 2)
        return cost
    
    def gradient_descent(self):
        ind = self.ind = np.c_[np.ones(self.ind.shape[0]), self.ind]
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