import utils
import config
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

with open(os.path.join(config.datasetPath, "train.tsv"), "r", encoding='utf8') as f:
    lines = f.readlines()
lines = [line.strip().split('\t') for line in lines]

scores = []
label = []
text = []
for line in tqdm(lines[1:10000]):
    vec0 = utils.sent2vec(line[0])
    vec1 = utils.sent2vec(line[1])
    score = np.dot(vec0, vec1.T)[0, 0]
    scores.append(score)
    label.append(int(line[2]))
    text.append((line[0], line[1]))

scores = np.array(scores)
label = np.array(label).astype(int)

plt.hist(scores[label == 0])
plt.show()
plt.hist(scores[label == 1])
plt.show()
