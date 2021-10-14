# 读入词表
path = r"data/dict.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

words = [line.split()[0] for line in lines]
wordSet = set(words)

sentence = input("请输入待分词句子：")  # 他研究生物化学

print("FMM:")
seg = []
wordNow = ""
for i in range(len(sentence)):
    wordNext = wordNow + sentence[i]
    if wordNext in wordSet:
        wordNow = wordNext
    else:
        if len(wordNow) == 0:
            seg.append(sentence[i])
            wordNow = ""
        else:
            seg.append(wordNow)
            wordNow = sentence[i]
if len(wordNow) > 0:
    seg.append(wordNow)
print(seg)

print("BMM:")
seg = []
wordNow = ""
for i in range(1, len(sentence) + 1):
    wordNext = sentence[-i] + wordNow
    if wordNext in wordSet:
        wordNow = wordNext
    else:
        if len(wordNow) == 0:
            seg.append(sentence[-i])
            wordNow = ""
        else:
            seg.append(wordNow)
            wordNow = sentence[-i]
if len(wordNow) > 0:
    seg.append(wordNow)
seg.reverse()
print(seg)
