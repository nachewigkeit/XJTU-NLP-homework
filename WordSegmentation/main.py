# 读入词表
path = r"data/dict.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

words = [line.split()[0] for line in lines]
wordSet = set(words)
print("词表长度：", len(wordSet))

length = [len(word) for word in words]
maxLen = max(length)
print("最大词长度：", maxLen)

sentence = input("请输入待分词句子：")  # 例：他研究生物化学
sentenceLen = len(sentence)

print("FMM:")
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
print(seg)

print("BMM:")
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
print(seg)
