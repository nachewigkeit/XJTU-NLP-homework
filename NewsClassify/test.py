import numpy as np
import pickle
import utils
import config
import matplotlib.pyplot as plt


def performance(path, x, y, clsNum):
    cTrain = np.zeros((clsNum, clsNum))
    cTest = np.zeros((clsNum, clsNum))
    pTrain = 0
    rTrain = 0
    fTrain = 0
    pTest = 0
    rTest = 0
    fTest = 0
    for i in range(config.fold):
        with open(path + str(i) + '.p', "rb") as file:
            clf = pickle.load(file)
        yPred = clf.predict(x)
        c, p, r, f = utils.evaluate(y[pos[i][0]], yPred[pos[i][0]], clsNum)
        cTrain += c
        pTrain += p / config.fold
        rTrain += r / config.fold
        fTrain += f / config.fold

        c, p, r, f = utils.evaluate(y[pos[i][1]], yPred[pos[i][1]], clsNum)
        cTest += c
        pTest += p / config.fold
        rTest += r / config.fold
        fTest += f / config.fold

    plt.figure()
    plt.title("Train:" + str(fTrain))
    plt.xticks(list(range(20)))
    plt.yticks(list(range(20)))
    plt.imshow(cTrain, cmap="Blues")
    plt.colorbar()
    plt.figure()
    plt.title("Test:" + str(fTest))
    plt.xticks(list(range(20)))
    plt.yticks(list(range(20)))
    plt.imshow(cTest, cmap="Blues")
    plt.colorbar()
    plt.show()
    print("precision")
    print(pTrain)
    print(pTest)
    print("recall")
    print(rTrain)
    print(rTest)
    print("F1")
    print(fTrain)
    print(fTest)


print("Load")
x = np.load(r'data/x.npy')
c = np.load(r'data/c.npy')
y = np.load(r'data/y.npy')
x /= np.max(x)

print("Test")
pos = utils.validFold(c.shape[0], config.fold)

print("NB")
performance("weight/NB/NB", c, y, 20)

print("NN")
performance("weight/NN/NN", x, y, 20)
