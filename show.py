import numpy as np
import matplotlib.pyplot as plt

log1 = np.load("th_CVaR-0.2.npy")
log2 = np.load("th_CVaR-0.3.npy")
log3 = np.load("th_CVaR-0.4.npy")
log4 = np.load("th_CVaR-1.0.npy")
x = range(10000)

y1 = log1[:, 0]
y2 = log2[:, 0]
y3 = log3[:, 0]
y4 = log4[:, 0]

c1 = np.mean(np.sort(y1)[:3000])

c1 = np.mean(np.sort(y1)[:100])
c2 = np.mean(np.sort(y2)[:100])
c3 = np.mean(np.sort(y3)[:100])
c4 = np.mean(np.sort(y4)[:100])
print(c1, c2, c3, c4)
plt.plot(x, y1, ',', label='0.2')
plt.plot(x, y2, ',', label='0.3')
plt.plot(x, y3, ',', label='0.4')
plt.plot(x, y4, ',', label='1.0')

plt.legend()
plt.show()