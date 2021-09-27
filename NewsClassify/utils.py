import os
import re
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import config

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
