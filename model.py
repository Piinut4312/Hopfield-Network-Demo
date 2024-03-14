import numpy as np
from dataset import Dataset

class HopfieldNetwork:

    def __init__(self, size: int=5):
        self.size = size
        self.W = np.zeros(shape=(size, 1))
        self.thr = np.zeros(shape=(size, 1))


    def train(self, train_data: Dataset):
        I = train_data.num_data*np.eye(self.size, self.size)/self.size
        self.W = np.sum([np.outer(data, data) for data in train_data], axis=0)/self.size-I
        # self.thr = np.sum(self.W, axis=0)
        self.thr = np.zeros(shape=(self.size, 1))
        

    def predict(self, data: np.ndarray, max_iter: int=50):
        result = np.copy(data)
        hist = []
        for _ in range(max_iter):
            changed = False
            for i in range(self.size):
                x = np.dot(self.W[i,], result)-self.thr[i]
                if x*result[i] < 0:
                    result[i] *= -1
                    changed = True
                hist.append(np.copy(result))
            if not changed:
                return hist
        return hist