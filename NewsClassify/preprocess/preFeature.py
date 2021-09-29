import os
import sys
import pickle
import numpy as np

sys.path.append('..')
import utils
import config

if __name__ == "__main__":
    with open(r"../data/idfCount.p", "rb") as f:
        idfCount = pickle.load(f)

    idfCountThres = {}
    thres = config.numThres
    for key, value in idfCount.items():
        if value >= thres:
            idfCountThres[key] = value
    featureNum = len(idfCountThres)
    fileNum = config.fileNum
    x = np.zeros((fileNum, featureNum))
    c = np.zeros((fileNum, featureNum))
    y = np.zeros(fileNum)

    dataPath = config.dataPath
    i = 0
    clsDict = config.clsDict
    for root, dirs, files in os.walk(dataPath):
        for file in files:
            y[i] = clsDict[root.split('\\')[-1]]

            filePath = os.path.join(root, file)
            fileCount, fileSet = utils.wordCount(filePath)
            total = sum(fileCount.values())

            j = 0
            for key, value in sorted(idfCountThres.items(), key=lambda x: x[1]):
                if key in fileCount:
                    x[i, j] = (fileCount[key] / total) * np.log10(fileNum / value)
                    c[i, j] = fileCount[key]
                j += 1
            i += 1
        print(i)

    np.save(r'../data/x.npy', x)
    np.save(r'../data/c.npy', c)
    np.save(r'../data/y.npy', y)
