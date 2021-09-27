import pickle
import matplotlib.pyplot as plt

with open(r"data/idfCount.p", "rb") as f:
    idfCount = pickle.load(f)

plt.axes(yscale="log")
plt.title("idf")
plt.hist(idfCount.values(), range=(0, 100))
plt.show()
