import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

N_raw = 1000          # number of raw range samples
N_chirp = 201         # matched-filter length, use odd value for clarity

# Synthetic LFM chirp
t = np.linspace(-1, 1, N_chirp)
K = 20.0
tx_chirp = np.exp(1j * np.pi * K * t**2)

# Matched filter: h[n] = s*[-n]
h = np.conj(tx_chirp[::-1])

# ============================================================
# Create a simple raw echo
# ============================================================

# Distributed targets + one strong point target
rng = np.random.default_rng(0)

reflectivity = (
    rng.normal(size=N_raw)
    + 1j * rng.normal(size=N_raw)
) / np.sqrt(2)

reflectivity[500] += 20

# Raw received signal = scene convolved with transmitted chirp
raw_full = np.convolve(reflectivity, tx_chirp, mode="same")

# ============================================================
# Matched filtering
# ============================================================

# Full linear convolution
rc_full = np.convolve(raw_full, h, mode="full")

# Equivalent "same" output: same length as input
rc_same = np.convolve(raw_full, h, mode="same")

# ============================================================
# Matched-filter throwaway
# ============================================================

# For an odd-length matched filter:
throw_left = (N_chirp - 1) // 2
throw_right = (N_chirp - 1) // 2

rc_valid = rc_same[throw_left : N_raw - throw_right]

print(f"Raw range samples       : {N_raw}")
print(f"Matched filter samples  : {N_chirp}")
print(f"Throwaway left          : {throw_left}")
print(f"Throwaway right         : {throw_right}")
print(f"Total throwaway         : {throw_left + throw_right}")
print(f"Valid output samples    : {len(rc_valid)}")

# Expected:
# N_valid = N_raw - (N_chirp - 1)

# ============================================================
# Number of filter samples actually overlapping the input
# ============================================================

overlap = np.convolve(
    np.ones(N_raw),
    np.ones(N_chirp),
    mode="same"
)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Matched-filter output
ax[0].plot(np.abs(rc_same), label="Matched-filter output")

ax[0].axvspan(
    0, throw_left,
    alpha=0.2,
    label="Throwaway"
)

ax[0].axvspan(
    N_raw - throw_right, N_raw,
    alpha=0.2
)

ax[0].axvline(throw_left, linestyle="--")
ax[0].axvline(N_raw - throw_right, linestyle="--")

ax[0].set_ylabel("|y[n]|")
ax[0].set_title("Matched filter output")
ax[0].legend()

# Filter/data overlap
ax[1].plot(overlap)

ax[1].axhline(
    N_chirp,
    linestyle="--",
    label="Full filter overlap"
)

ax[1].axvline(throw_left, linestyle="--")
ax[1].axvline(N_raw - throw_right, linestyle="--")

ax[1].set_xlabel("Range sample")
ax[1].set_ylabel("Number of overlapping samples")
ax[1].set_title("Matched-filter overlap with available raw data")
ax[1].legend()

plt.tight_layout()
plt.show()