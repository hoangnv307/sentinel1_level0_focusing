import numpy as np
import matplotlib.pyplot as plt

# Fractional position cần nội suy
delta = 24 / 64   # = 0.375

# 16 tap quanh vị trí cần nội suy
# Chọn support gần đối xứng quanh 0
k = np.arange(-7, 9)   # 16 samples: -7 ... +8

# Ideal sinc interpolation kernel
h_sinc = np.sinc(k - delta)

# Ví dụ taper bằng Hamming window
window = np.hamming(16)

# Windowed sinc
h = h_sinc * window

# Normalize giống ý tưởng DAD: tổng coefficients = 1
h /= np.sum(h)

print("Fractional offset =", delta)
print("Kernel coefficients:")
for ki, hi in zip(k, h):
    print(f"k={ki:2d}, h={hi:+.8f}")

print("Sum =", np.sum(h))

# Vẽ kernel
plt.figure(figsize=(10, 5))
plt.stem(k, h)
plt.axvline(delta, linestyle="--", label=f"Fractional position δ={delta}")
plt.xlabel("Relative sample index k")
plt.ylabel("Interpolation coefficient h[k]")
plt.title("16-tap fractional-delay sinc kernel")
plt.grid(True)
plt.legend()
plt.show()