import numpy as np
import matplotlib.pyplot as plt

# Define the number of threads and measured times (in seconds)
threads = np.array([1, 2, 4, 8, 16])
times = np.array([11.029246, 5.932864, 3.452574, 2.641656, 3.082972])

# Compute speedup and efficiency
speedup = times[0] / times
efficiency = speedup / threads

# Create subplots for speedup and efficiency
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot Speedup
ax[0].plot(threads, speedup, marker='o', linestyle='-', color='b')
ax[0].set_title('Speedup vs. Number of Threads')
ax[0].set_xlabel('Number of Threads')
ax[0].set_ylabel('Speedup')
ax[0].grid(True)

# Plot Efficiency
ax[1].plot(threads, efficiency, marker='o', linestyle='-', color='r')
ax[1].set_title('Efficiency vs. Number of Threads')
ax[1].set_xlabel('Number of Threads')
ax[1].set_ylabel('Efficiency')
ax[1].grid(True)

plt.tight_layout()
plt.show()

