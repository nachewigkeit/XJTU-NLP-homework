from tqdm import tqdm

# 读入词表
path = r"data/dict.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

words = [line.split()[0] for line in lines]
wordSet = set(words)

length = [len(word) for word in words]
maxLen = max(length)


def FMM(sentence):
    sentenceLen = len(sentence)
    seg = []
    i = 0
    while i < sentenceLen:
        wordNow = ""
        for j in range(min(maxLen, sentenceLen - i), 0, -1):
            wordNow = sentence[i:i + j]
            if wordNow in wordSet:
                i = i + j
                seg.append(wordNow)
                break
        if wordNow not in wordSet:
            seg.append(sentence[i])
            i = i + 1
    return seg


def BMM(sentence):
    sentenceLen = len(sentence)
    seg = []
    i = sentenceLen
    while i > 0:
        wordNow = ""
        for j in range(min(maxLen, i), 0, -1):
            wordNow = sentence[i - j:i]
            if wordNow in wordSet:
                i = i - j
                seg.append(wordNow)
                break
        if wordNow not in wordSet:
            seg.append(sentence[i - 1])
            i = i - 1

    seg.reverse()
    return seg


def word2pos(words):
    start = 0
    pos = []
    for word in words:
        end = start + len(word)
        pos.append((start, start + end))
        start = end
    return pos


def splitEvaluate(wordTrue, wordPred):
    posTrue = word2pos(wordTrue)
    posPred = word2pos(wordPred)

    num = 0
    for pos in posPred:
        if pos in posTrue:
            num += 1

    p = num / len(posPred)
    r = num / len(posTrue)
    f = 2 * p * r / (p + r + 1e-6)
    return p, r, f


def algorithmEvaluate(noSeg, seg, func):
    p = 0
    r = 0
    f = 0
    for i in tqdm(range(len(noSeg))):
        sentence = noSeg[i]
        score = splitEvaluate(seg[i], func(sentence))
        p += score[0]
        r += score[1]
        f += score[2]

    p /= len(noSeg)
    r /= len(noSeg)
    f /= len(noSeg)

    return p, r, f


if __name__ == "__main__":
    print("词表长度：", len(wordSet))
    print("最大词长度：", maxLen)
    sentence = input("请输入待分词句子：")  # 例：他研究生物化学
    print("FMM", FMM(sentence))
    print("BMM", BMM(sentence))
