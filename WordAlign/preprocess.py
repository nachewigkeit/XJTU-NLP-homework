import os
import re
import numpy as np
import nltk
import pickle
from tqdm import tqdm

path = r"E:\AI\dataset\NLP\translation"

with open(os.path.join(path, 'cn.txt'), "r", encoding='utf-8') as f:
    cnLines = f.readlines()
with open(os.path.join(path, 'en.txt'), "r") as f:
    enLines = f.readlines()
lemmatizer = nltk.stem.WordNetLemmatizer()
enWordSet = set()
translation = {}
distortion = {}
sents = []
for i in tqdm(range(len(cnLines))):
    # 中文分词
    cnWord = re.findall(u'[\u4E00-\u9FA5]+', cnLines[i])
    cnWord.insert(0, "NULL")

    # 英文分词与词形变化
    enLine = enLines[i].lower()
    enWord = re.findall(u'[a-z]+\'?[a-z]+', enLine)
    enWord = [lemmatizer.lemmatize(i) for i in enWord]
    enWordSet.update(enWord)

    sents.append([cnWord, enWord])

    # 建翻译稀疏表
    for cn in cnWord:
        if cn not in translation.keys():
            translation[cn] = {}
        for en in enWord:
            if en not in translation[cn].keys():
                translation[cn][en] = [0, 0]
            translation[cn][en][1] += 1

    # 建扭曲表
    shape = (len(cnWord), len(enWord))
    if shape not in distortion.keys():
        distortion[shape] = np.zeros((*shape, 2))
        distortion[shape][:, :, 1] = 1 / len(cnWord)

print("中文词表大小", len(translation))
print("英文词表大小", len(enWordSet))
print("训练集句数", len(sents))

sumLen = 0
for i in translation.values():
    sumLen += len(i)
    sumTrans = 0
    for j in i.values():
        sumTrans += j[1]
    for j in i.values():
        j[1] /= sumTrans

print("平均相关词数", sumLen / len(translation))

with open(r"data/sents", "wb") as f:
    pickle.dump(sents, f)
with open(r"data/translation", "wb") as f:
    pickle.dump(translation, f)
with open(r"data/distortion", "wb") as f:
    pickle.dump(distortion, f)
