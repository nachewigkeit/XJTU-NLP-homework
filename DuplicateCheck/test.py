import numpy as np

import config
import utils

with open(config.newTextPath, "r", encoding="utf-8") as f:
    lines = f.readlines()

newText, newVec = utils.lines2data(lines)
print(newVec.shape)

scores = np.dot(utils.database['vec'], newVec.T)
pos = np.argmax(scores, axis=0)
for i in range(len(pos)):
    print(newText[i])
    print(utils.database['text'][pos[i]])
    print(scores[pos[i], i])
