import numpy as np
import pickle
import sys

sys.path.append("..")
import config
import utils
from model.Network import Network

print("Load")
x = np.load(r'../data/x.npy')
y = np.load(r'../data/y.npy')
x /= np.max(x)

print("Train")
pos = utils.validFold(x.shape[0], config.fold)
for i in range(config.fold):
    clf = Network(x.shape[1], 20, 10)
    for epoch in range(500):
        clf.fit(x[pos[i][0], :], y[pos[i][0]], 1000)
    print("train loss", clf.loss(x[pos[i][0], :], y[pos[i][0]]))
    print("test loss", clf.loss(x[pos[i][1], :], y[pos[i][1]]))
    with open(r'../weight/NN/NN' + str(i) + '.p', "wb") as f:
        pickle.dump(clf, f)
