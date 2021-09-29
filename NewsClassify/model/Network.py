import numpy as np
from random import shuffle


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-6)


class Network:
    def __init__(self, featureNum, clsNum, rate):
        self.featureNum = featureNum
        self.clsNum = clsNum
        self.rate = rate

        self.w = np.random.uniform(-np.sqrt(6 / (featureNum + clsNum)), np.sqrt(6 / (featureNum + clsNum)),
                                   (featureNum, clsNum))
        self.b = np.zeros((1, clsNum))

    def forward(self, x):
        return softmax(np.matmul(x, self.w) + self.b)

    def loss(self, x, y):
        p = self.forward(x)
        p = -np.log(p + 1e-6)
        sum = 0
        for i in range(x.shape[0]):
            sum += p[i, int(y[i])]
        return sum / x.shape[0]

    def update(self, x, y):
        bPrime = self.forward(x)
        for i in range(x.shape[0]):
            bPrime[i, int(y[i])] -= 1
        wPrime = np.matmul(x.T, bPrime) / x.shape[0]
        bPrime = np.sum(bPrime, axis=0) / x.shape[0]

        self.w -= wPrime * self.rate
        self.b -= bPrime * self.rate

    def fit(self, x, y, batch):
        pos = list((range(x.shape[0])))
        shuffle(pos)

        for i in range(int(x.shape[0] / batch)):
            batchPos = pos[i * batch:(i + 1) * batch]
            self.update(x[batchPos, :], y[batchPos])

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
