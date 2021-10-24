from sentencesplit import FMM, BMM, algorithmEvaluate

path = r"E:\AI\dataset\icwb2-data\training\pku_training.utf8"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

seg = [line.split() for line in lines if len(line) > 0]
noSeg = ["".join(line) for line in seg]

p, r, f = algorithmEvaluate(noSeg, seg, FMM)
print(p, r, f)

p, r, f = algorithmEvaluate(noSeg, seg, BMM)
print(p, r, f)
