import numpy as np
import pickle
import sys

sys.path.append("..")
import config
import utils
from model.NaiveBayes import NaiveBayes

print("Load")
c = np.load(r'../data/c.npy')
y = np.load(r'../data/y.npy')

print("Train")
pos = utils.validFold(c.shape[0], config.fold)
for i in range(config.fold):
    clf = NaiveBayes(c.shape[1], 20)
    clf.fit(c[pos[i][0], :], y[pos[i][0]])
    with open(r'../weight/NB/NB' + str(i) + '.p', "wb") as f:
        pickle.dump(clf, f)
