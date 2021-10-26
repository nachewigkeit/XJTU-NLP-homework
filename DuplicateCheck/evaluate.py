import utils
import config
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve

with open(os.path.join(config.datasetPath, "train.tsv"), "r", encoding='utf8') as f:
    lines = f.readlines()
lines = [line.strip().split('\t') for line in lines]

scores = []
label = []
text = []
length = []
for line in tqdm(lines[1:2000]):
    vec0 = utils.sent2vec(line[0])
    vec1 = utils.sent2vec(line[1])
    score = np.dot(vec0, vec1.T)[0, 0]
    scores.append(score)
    label.append(int(line[2]))
    text.append((line[0], line[1]))
    length.append((len(line[0]) + len(line[1])) / 2)

scores = np.array(scores)
label = np.array(label).astype(int)

fpr, tpr, thresholds = roc_curve(label, scores)
print(auc(fpr, tpr))
plt.plot(fpr, tpr)
plt.show()

'''
plt.scatter(length, abs(scores - label))
plt.show()
'''
