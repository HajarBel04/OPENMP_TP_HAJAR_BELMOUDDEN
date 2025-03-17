import numpy as np
import matplotlib.pyplot as plt

# Number of threads used
threads = np.array([1, 2, 4, 8, 16])
# Measured elapsed times in seconds
times = np.array([0.01292, 0.01004, 0.01098, 0.02172, 0.04048])

# Calculate speedup and efficiency
speedup = times[0] / times
efficiency = speedup / threads

# Print computed speedup and efficiency for reference
for t, s, e in zip(threads, speedup, efficiency):
    print(f"Threads: {t}, Speedup: {s:.3f}, Efficiency: {e:.3f}")

# Plot Speedup vs. Threads
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(threads, speedup, marker='o', linestyle='-')
plt.title("Speedup vs. Number of Threads")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.grid(True)

# Plot Efficiency vs. Threads
plt.subplot(1, 2, 2)
plt.plot(threads, efficiency, marker='o', linestyle='-', color='red')
plt.title("Efficiency vs. Number of Threads")
plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.grid(True)

plt.tight_layout()
plt.show()

