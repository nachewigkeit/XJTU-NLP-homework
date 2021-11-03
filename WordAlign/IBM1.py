import pickle
from tqdm import tqdm
import numpy as np
import utils

epoch = 10

with open(r"data/sents", "rb") as f:
    sents = pickle.load(f)
with open(r"data/translation", "rb") as f:
    translation = pickle.load(f)
text = []
for k in sents:
    text.append(np.zeros((epoch, len(k[1])), dtype=np.int))

for s in tqdm(range(epoch)):
    # count
    for k in sents:
        cnSent, enSent = k

        for i in enSent:
            sum = 0
            for j in cnSent:
                sum += translation[j][i][1]

            for j in cnSent:
                delta = translation[j][i][1] / sum
                translation[j][i][0] += delta

    # update
    for cn in translation.values():
        sum = 0
        for en in cn.values():
            sum += en[0]
        for en in cn.values():
            en[1] = en[0] / sum
            en[0] = 0

    # visualize
    for k in range(len(sents)):
        cnSent, enSent = sents[k]
        for i in range(len(enSent)):
            maxProb = -1
            for j in range(len(cnSent)):
                if translation[cnSent[j]][enSent[i]][1] > maxProb:
                    maxProb = translation[cnSent[j]][enSent[i]][1]
                    maxWord = j
            text[k][s, i] = maxWord

utils.visualize(sents, text, "IBM1")
