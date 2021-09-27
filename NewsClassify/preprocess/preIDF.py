import os
import sys
from collections import Counter
from tqdm import tqdm
import pickle

sys.path.append('..')
import utils
import config

if __name__ == "__main__":
    idfCount = Counter()
    dataPath = config.dataPath
    for root, dirs, files in os.walk(dataPath):
        for file in tqdm(files):
            filePath = os.path.join(root, file)
            fileCount, fileSet = utils.wordCount(filePath)
            idfCount.update(fileSet)

    with open(r"../data/idfCount.p", "wb") as f:
        pickle.dump(idfCount, f)
