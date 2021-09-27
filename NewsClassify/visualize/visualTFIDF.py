import numpy as np
import matplotlib.pyplot as plt

x = np.load(r'../data/x.npy')
y = np.load(r'../data/y.npy')

print(x.shape)
print(y.shape)

print(x[:, :-1].mean())
print(x[:, -1].mean())

print(y.mean())

plt.hist(x[:, -1])
plt.show()
