import numpy as np
import os
import utils
import config
from tqdm import tqdm
import pickle

text = []
vec = []
for root, dirs, files in os.walk(config.oldTextPath):
    for file in tqdm(files):
        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
            lines = f.readlines()

        oneText, oneVec = utils.lines2data(lines)
        text += oneText
        vec.append(oneVec)

vec = np.vstack(vec)
print(vec.shape)
database = {
    "text": text,
    "vec": vec
}
with open(config.databasePath, "wb") as f:
    pickle.dump(database, f)
