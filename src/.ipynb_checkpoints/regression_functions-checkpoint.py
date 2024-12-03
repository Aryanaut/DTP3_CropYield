import numpy as np

def normalize_z(array: np.ndarray, columns_means,
                columns_stds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if columns_means is None:
        columns_means=np.mean(array,axis=0)
    if columns_stds is None:
        columns_stds=np.std(array,axis=0)
        
    out = (array-columns_means)/columns_stds
    
    return out, columns_means, columns_stds

def normalize_minmax(array_in: np.ndarray, columns_mins, 
                     columns_maxs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out=array_in.copy()
    columns_mins = np.min(out, axis=0, keepdims=True) if columns_mins is None else columns_mins
    columns_maxs = np.max(out, axis=0, keepdims=True) if columns_maxs is None else columns_maxs
    out = (out-columns_mins)/(columns_maxs-columns_mins)
    return out, columns_mins, columns_maxs

class MVF:

    # ind: independant variable matrix, np array, size = no. of data points * no. of dependant variables
    # dep: depdendant variable matrix, size = no. of data points * 1

    def __init__(self, ind, dep):
    
        self.lr = 0.01
        self.iterations = 1000
        self.ind = ind
        self.dep = dep
        self.weights = np.zeros(self.dep.shape[1])
    
        self.costs = []

    def normalize_data():
        noramlize_z(self.ind)
        normalize_z(self.dep)

    def _compute_cost(self):
        m = len(self.dep)
        predictions = np.dot(self.ind, weights)
        cost = (1 / (2 * m)) * np.sum((predictions - self.dep) ** 2)
        return cost
    
    def gradient_descent(self, X, y, weights, lr, iterations):
        m = len(self.dep)
        for _ in range(self.iterations):
            predictions = np.dot(self.ind, self.weights)
            errors = predictions - self.dep
            gradient = (1 / m) * np.dot(self.ind.T, errors)
            weights -= self.lr * gradient
            cost = self._compute_cost()
            self.costs.append(cost)
        return weights