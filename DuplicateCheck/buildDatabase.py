import numpy as np
import os
import utils
import config
from tqdm import tqdm
import pickle

id = []
pos = []
text = []
vec = []
for root, dirs, files in os.walk(config.oldTextPath):
    for file in tqdm(files):
        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
            lines = f.readlines()

        onePos, oneText, oneVec = utils.lines2data(lines)
        id += [int(file.split('.')[0])] * len(oneText)
        pos += onePos
        text += oneText
        vec.append(oneVec)

vec = np.vstack(vec)
print(vec.shape)
database = {
    "id": np.array(id).astype('int'),
    "pos": pos,
    "text": text,
    "vec": vec
}
with open(config.databasePath, "wb") as f:
    pickle.dump(database, f)
