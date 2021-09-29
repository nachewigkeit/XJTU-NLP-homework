import numpy as np


class NaiveBayes:
    def __init__(self, featureNum, clsNum):
        self.clsNum = clsNum
        self.weight = np.zeros((featureNum, clsNum))

    def fit(self, x, y):
        for cls in range(self.clsNum):
            clsWeight = np.sum(x[y == cls, :], axis=0)
            clsWeight = clsWeight + 1
            clsWeight /= np.sum(clsWeight)
            self.weight[:, cls] = clsWeight

    def predict(self, x):
        prob = np.matmul(x, np.log2(self.weight))
        return np.argmax(prob, axis=1)
