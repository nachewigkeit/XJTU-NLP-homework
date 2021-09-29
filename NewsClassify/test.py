import numpy as np
import pickle
import utils
import config


def printPRF(path, x, y):
    pTrain = []
    rTrain = []
    fTrain = []
    pTest = []
    rTest = []
    fTest = []
    for i in range(config.fold):
        with open(path + str(i) + '.p', "rb") as file:
            clf = pickle.load(file)
        yPred = clf.predict(x)
        p, r, f = utils.evaluate(y[pos[i][0]], yPred[pos[i][0]], 20)
        pTrain.append(p)
        rTrain.append(r)
        fTrain.append(f)

        p, r, f = utils.evaluate(y[pos[i][1]], yPred[pos[i][1]], 20)
        pTest.append(p)
        rTest.append(r)
        fTest.append(f)

    print("p")
    print(np.mean(pTrain))
    print(np.mean(pTest))
    print("r")
    print(np.mean(rTrain))
    print(np.mean(rTest))
    print("f")
    print(np.mean(fTrain))
    print(np.mean(fTest))


print("Load")
x = np.load(r'data/x.npy')
c = np.load(r'data/c.npy')
y = np.load(r'data/y.npy')
x /= np.max(x)

print("Test")
pos = utils.validFold(c.shape[0], config.fold)

print("NB")
printPRF("weight/NB/NB", c, y)

print("NN")
printPRF("weight/NN/NN", x, y)
