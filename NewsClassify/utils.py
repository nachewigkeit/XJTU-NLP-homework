import os
import re
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import config
import numpy as np

stemmer = SnowballStemmer("english")
with open(os.path.join(config.projectPath, 'stopWords.txt'), 'r') as f:
    # 停用词
    stopWords = f.read().splitlines()


def ifValid(string):
    if '@' in string:
        # 邮箱
        return False
    elif any(char.isdigit() for char in string):
        # 数字
        return False
    elif string in stopWords:
        # 停用词
        return False
    elif len(string) < 3:
        # 碎片
        return False
    else:
        return True


def wordCount(filePath):
    fileCount = Counter()
    fileSet = set()

    with open(filePath, 'r', encoding='ISO-8859-1') as f:
        lines = f.read().splitlines()

    headFlag = True
    for line in lines:
        # 去除metaData
        line = line.strip().lower()
        if headFlag and (':' in line or len(line) == 0):
            continue
        else:
            headFlag = False

        splitLine = re.findall(r"\w+(?:[-'@.]\w+)*", line)  # 分词
        splitLine = [stemmer.stem(i) for i in splitLine if ifValid(i)]  # 清除无用词
        fileCount.update(splitLine)
        fileSet.update(splitLine)

    return fileCount, fileSet


def confusion_matrix(yTrue, yPred, clsNum):
    matrix = np.zeros((clsNum, clsNum))
    for i in range(clsNum):
        for j in range(clsNum):
            matrix[i, j] = np.sum(yPred[yTrue == i] == j)
    return matrix


def PR(matrix):
    p = []
    r = []
    for i in range(matrix.shape[0]):
        r.append(matrix[i, i] / (np.sum(matrix[i, :]) + 1e-6))
        p.append(matrix[i, i] / (np.sum(matrix[:, i]) + 1e-6))
    return p, r


def F1(p, r):
    f = []
    for i in range(len(p)):
        f.append(2 * p[i] * r[i] / (p[i] + r[i] + 1e-6))
    return f


def evaluate(yTrue, yPred, clsNum):
    matrix = confusion_matrix(yTrue, yPred, clsNum)
    p, r = PR(matrix)
    f = F1(p, r)
    return np.mean(p), np.mean(r), np.mean(f)


def validFold(fileNum, fold=5):
    split = []
    for i in range(fold):
        split.append(list(range(i, fileNum, fold)))

    answer = []
    for i in range(fold):
        answer.append([[], []])
        for j in range(fold):
            if i == j:
                answer[i][1] = split[j]
            else:
                answer[i][0] += split[j]
    return answer
