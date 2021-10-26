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


def splitPara(para):
    para = re.sub('([。！？；])([^：])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^：])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^：])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([：:][“‘])([^：:])', r'\1\n\2', para)  # 冒号+引号时，应该分句
    para = re.sub('“|‘|”|’', '', para)  # 中文引号删除
    para = re.sub('\"|\'', '', para)  # 英文引号删除
    para = re.sub('（[一二三四五六七八九十]+）|[一二三四五六七八九十]+、', '', para)  # 删除汉字序号
    para = re.sub('\d+\.(?!\d+)', '', para)  # 删除数字序号
    para = re.sub('\n+', '\n', para)  # 删除多余的换行符
    para = re.sub('　|——', '', para)  # 删除中文的宽空格、连字符
    para = para.strip()  # 删除段尾的换行符
    return para.split("\n")


def sent2vec(sent):
    vec = model.encode(sent)
    vec = prepro.normalize(vec.reshape(1, -1))
    return vec


def lines2data(lines):
    text = []
    vec = []

    for line in lines:
        line = line.strip()
        if len(line) > 0:
            sents = splitPara(line)
            for sent in sents:
                text.append(sent)
                vec.append(sent2vec(sent))

    vec = np.vstack(vec)
    return text, vec


if __name__ == "__main__":
    text = splitPara("苹果（一种水果）")
    text = "\n".join(text[:])
    print(text)
    print(sent2vec(text).shape)
