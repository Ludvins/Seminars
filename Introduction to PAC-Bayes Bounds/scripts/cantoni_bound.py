import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def cantoni_bound(emp_risk, lam, C, n, KL_rho_pi, delta):
    """
    Cantoni's Bound (2003):
      E_{theta~rho}[R(theta)] <= emp_risk
        + (lam*C^2)/(8 * n)
        + (KL(rho||pi) + ln(1/delta)) / lam
    """
    return (
        emp_risk
        + (lam * C**2) / (8.0 * n)
        + (KL_rho_pi + np.log(1.0 / delta)) / lam
    )

# -----------------------------------------------------
# 1) Choose parameters to reveal a visible "valley".
#    - Smaller n range, bigger KL, smaller delta
# -----------------------------------------------------
emp_risk = 0.2
C        = 10.0
delta    = 0.01  # somewhat small => bigger ln(1/delta)

# We'll draw 3 surfaces, each with a different KL
KL_list = [1.0, 5.0, 10.0]
colors  = ["red", "green", "blue"]

# -----------------------------------------------------
# 2) Build mesh for (n, lambda)
#    Keep n moderate so both terms matter.
# -----------------------------------------------------
n_vals   = np.linspace(10, 50, 10)    # from 50 to 500
lam_vals = np.linspace(1, 10.0, 50)  # from 0.01 to 3
N_grid, LAM_grid = np.meshgrid(n_vals, lam_vals)

# -----------------------------------------------------
# 3) Plot all 3 surfaces in a single 3D plot
# -----------------------------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

patches = []

for color, KL_val in zip(colors, KL_list):
    Z_bound = cantoni_bound(emp_risk, LAM_grid, C, N_grid, KL_val, delta)
    
    surf = ax.plot_surface(
        N_grid,         # x-axis
        LAM_grid,       # y-axis
        Z_bound,        # z-axis
        color=color,
        alpha=0.7,      # transparency helps see overlap
        edgecolor="none"
    )
    
    # Create patch for the legend
    patch = mpatches.Patch(color=color, label=f"KL = {KL_val}")
    patches.append(patch)

# Legend with our color patches
ax.legend(handles=patches, loc="upper right")

ax.set_xlabel("Training set size (n)")
ax.set_ylabel(r"$\lambda$")
ax.set_zlabel("Cantoni’s Bound")

plt.tight_layout()
plt.show()
