import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import sklearn.preprocessing as prepro
import config
import pickle
from time import time

start = time()
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
model = model.cuda()
print("model load time:", time() - start)
start = time()
if os.path.exists(config.databasePath):
    with open(config.databasePath, "rb") as f:
        database = pickle.load(f)
print("database load time:", time() - start)


def splitPara(para, thres=20):
    para = re.sub('([，。！？；])([^：])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^：])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^：])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([：:][“‘])([^：:])', r'\1\n\2', para)
    para = re.sub('“|‘|”|’', '', para)
    para = re.sub('（[一二三四五六七八九十]+）|[一二三四五六七八九十]+、|\([一二三四五六七八九十]+\)', '', para)  # 删除序号
    para = re.sub('\n+', '\n', para)  # 删除多余的换行符
    para = re.sub('　|——', '', para)  # 删除中文的宽空格、连字符
    para = para.rstrip()  # 删除段尾的换行符
    texts = para.split("\n")

    idx = 0
    while idx < len(texts):
        if len(texts[idx]) < thres and idx != len(texts) - 1:
            texts[idx] = texts[idx] + texts[idx + 1]
            del texts[idx + 1]
        else:
            idx += 1
    return texts


def sent2vec(sent):
    vec = model.encode(sent)
    vec = prepro.normalize(vec.reshape(1, -1))
    return vec


def lines2data(lines):
    pos = []
    text = []
    vec = []

    for i in range(len(lines)):
        line = lines[i].strip()
        if len(line) > 0:
            sents = splitPara(line)
            for sent in sents:
                text.append(sent)
                vec.append(sent2vec(sent))
            pos += [i] * len(sents)

    vec = np.vstack(vec)
    return pos, text, vec


def duplicateCheck(lines, tars, thres, k):
    queryPos, queryText, queryVec = lines2data(lines)
    answer = []

    docId = []
    docPos = []
    docText = []
    docVec = []

    for tar in tars:
        index = np.where(database['id'] == tar)[0].tolist()
        docId += [tar] * len(index)
        docPos +=[database['pos'][i] for i in index]
        docText += [database['text'][i] for i in index]
        docVec.append(database['vec'][index])

    docVec = np.vstack(docVec)
    scores = np.dot(docVec, queryVec.T)

    for j in range(scores.shape[1]):
        index = np.where(scores[:, j] > thres)[0].tolist()
        if len(index) > k:
            index = np.argsort(-scores[:, j])[:k].tolist()
        for i in index:
            answer.append((docId[i], docPos[i], scores[i, j], docText[i], queryPos[j], queryText[j]))

    return answer


if __name__ == "__main__":
    text = splitPara("苹果（一种水果）")
    text = "\n".join(text[:])
    print(text)
    print(sent2vec(text).shape)
