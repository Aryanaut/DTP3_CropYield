import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

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
        self.finaliterations = 0
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

    def gradient_descent(self):
        ind = np.c_[np.ones(self.ind.shape[0]), self.ind]
        weights = np.zeros((ind.shape[1], 1))
        m = len(self.dep)

        self.costs = []

        self.finaliterations = 0
        
        for _ in range(self.iterations):
            predictions = np.dot(ind, weights)
            errors = predictions - self.dep
            gradient = (1 / m) * np.dot(ind.T, errors)
            weights -= self.lr * gradient
            cost = self._compute_cost(ind, weights)
            self.costs.append(cost)
            self.finaliterations += 1

        self.weights = weights
        self.predictions = predictions * self.dep_std + self.dep_mean

    
    def gradient_descent_improved(self, lambda_reg=0.1, tol=1e-6):
        ind = np.c_[np.ones(self.ind.shape[0]), self.ind]
        weights = np.zeros((ind.shape[1], 1))
        m = len(self.dep)

        self.costs = []

        self.finaliterations = 0
        
        for _ in range(self.iterations):
            predictions = np.dot(ind, weights)
            errors = predictions - self.dep
            gradient = (1 / m) * np.dot(ind.T, errors)
            weights -= self.lr * gradient
            cost = self._compute_cost(ind, weights)
            self.costs.append(cost)
            self.finaliterations += 1

            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < tol: 
                print("Convergence found")
                break

            if len(self.costs) > 1 and self.costs[-1] > self.costs[-2]:
                lr *= lr_decay
                print(f"Learning rate decreased to {lr} at iteration {i + 1}.")

        self.weights = weights
        self.predictions = predictions * self.dep_std + self.dep_mean


class MVFSplit(MVF):

    def __init__(self, ind, dep):
        super().__init__(ind, dep)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(ind, dep, test_size=0.2, random_state=42)

        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Validation set size: {self.X_val.shape[0]}")

    def _compute_cost(self, X, y, weights, lambda_reg=0.1):
        m = len(y)  # Number of examples

        # Predicted values
        predictions = np.dot(X, weights)
        mse_cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        reg_term = (lambda_reg / (2 * m)) * np.sum(weights[1:] ** 2)  # Exclude weights[0]
        
        # Total cost
        total_cost = mse_cost + reg_term
        return total_cost

    def gradient_descent(self, lambda_reg=0.1, tol=1e-6):

        # Normalize features using the training set's statistics
        X_train, X_mean, X_std = normalize_array(self.X_train)
        X_val = (self.X_val - X_mean) / X_std 
        y_train, y_mean, y_std = normalize_array(self.y_train)
        y_val = (self.y_val - y_mean) / y_std 

        # Add the bias term
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        X_val = np.c_[np.ones(X_val.shape[0]), X_val]

        m_train = len(self.y_train)
        self.costs = []
        self.val_costs = []

        self.weights = np.zeros((X_train.shape[1], 1))
        
        for i in range(self.iterations):
            # Training set gradient descent
            # print(X_train.shape)
            predictions_train = np.dot(X_train, self.weights)
            errors_train = predictions_train - y_train
            
            gradient = (1 / m_train) * np.dot(X_train.T, errors_train) + (lambda_reg / m_train) * np.vstack(([0], self.weights[1:]))
            self.weights -= self.lr * gradient
            
            # Compute training cost
            cost = self._compute_cost(X_train, y_train, self.weights, lambda_reg)
            self.costs.append(cost)
            
            # Compute validation cost
            val_cost = self._compute_cost(X_val, y_val, self.weights, lambda_reg)
            self.val_costs.append(val_cost)
            
            # Early stopping based on training cost
            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < tol:
                print(f"Convergence reached at iteration {i + 1}!")
                break

            if len(self.costs) > 1 and self.costs[-1] > self.costs[-2]:
                lr *= lr_decay
                print(f"Learning rate decreased to {lr} at iteration {i + 1}.")

        # return weights, costs, val_costs
