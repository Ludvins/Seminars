import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def pac_bound(r_min, C, M, n, delta):
    """PAC-style bound: r_min + C * sqrt( ln(M / delta) / (2n) )."""
    return r_min + C * np.sqrt(np.log(M / delta) / (2 * n))

# -----------------------------------------------------
# 1) Fixed hyperparameters and the (M, n) mesh
# -----------------------------------------------------
r_min_fixed = 0.26
C_fixed     = 1.0

M_vals = np.linspace(10, 2000, 60)   # from 10 to 2000
n_vals = np.linspace(100, 5000, 60)  # from 100 to 5000
M_grid, n_grid = np.meshgrid(M_vals, n_vals)

# Three different delta values we'll compare
delta_list = [0.001, 0.05, 0.1]

# We'll give each surface a distinct color
colors = ["red", "green", "blue"]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# -----------------------------------------------------
# 2) Plot three surfaces, one per delta
# -----------------------------------------------------
surfaces = []
for color, d in zip(colors, delta_list):
    Z = pac_bound(r_min_fixed, C_fixed, M_grid, n_grid, d)
    surf = ax.plot_surface(
        M_grid, n_grid, Z,
        color=color,
        alpha=0.6,  # transparency so surfaces don’t fully hide each other
        edgecolor="none"
    )
    surfaces.append(surf)

# -----------------------------------------------------
# 3) Build a simple legend for the colors
# -----------------------------------------------------
patches = [
    mpatches.Patch(color=c, label=r"$\delta = {d}$".format(d=d))
    for c, d in zip(colors, delta_list)
]
ax.legend(handles=patches, loc="upper right")

# Optional: set log scales for clarity if desired
# ax.set_xscale("log")
# ax.set_yscale("log")

ax.set_xlabel("Model Space Size")
ax.set_ylabel("Training Set Size")
ax.set_zlabel("PAC Bound")

plt.tight_layout()
plt.show()
