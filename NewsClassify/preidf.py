import os
from collections import Counter
from tqdm import tqdm
import pickle
import utils

if __name__ == "__main__":
    idfCount = Counter()
    dataPath = r"E:\dataset\20_newsgroups"
    for root, dirs, files in os.walk(dataPath):
        for file in tqdm(files):
            filePath = os.path.join(root, file)
            fileCount, fileSet = utils.wordCount(filePath)
            idfCount.update(fileSet)

    with open(r"data/idfCount.p", "wb") as f:
        pickle.dump(idfCount, f)
