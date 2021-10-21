import pickle
import matplotlib.pyplot as plt
import numpy as np

with open(r"../data/idfCount.p", "rb") as f:
    idfCount = pickle.load(f)

print(idfCount.most_common(10))

idfCount = np.array(list(idfCount.values()))
print("<=1:", np.sum(idfCount <= 1))
print("<=2:", np.sum(idfCount <= 2))
print("<=3:", np.sum(idfCount <= 3))
print("<=4:", np.sum(idfCount <= 4))
print(">=5:", np.sum(idfCount >= 5))

plt.axes(yscale="log")
plt.hist(idfCount)
plt.show()
