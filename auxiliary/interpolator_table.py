import numpy as np
import matplotlib.pyplot as plt

k = np.arange(-7, 9)

plt.figure(figsize=(11, 6))

for p in range(64):
    delta = p / 64

    h = np.sinc(k - delta)
    h *= np.hamming(16)
    h /= np.sum(h)

    plt.plot(k, h, alpha=0.25)

plt.xlabel("Relative sample index k")
plt.ylabel("Coefficient")
plt.title("64 fractional-delay kernels, each with 16 taps")
plt.grid(True)
plt.show()