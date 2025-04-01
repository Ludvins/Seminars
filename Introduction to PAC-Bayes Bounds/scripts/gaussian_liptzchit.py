import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Example placeholders: fill in your real choices here
# ---------------------------------------------------------------------
def r_of_m(m):
    # A toy function for demonstration, e.g. a small parabola:
    return 0.5 * (m + 1)**2

# Some constants you might have in your formula:
lambda_   = 0.1      # regularization scale
C         = 1.0
n         = 100.0
sigma_s   = 1.0
delta     = 0.05
d         = 1        # dimension = 1 for plotting

# ---------------------------------------------------------------------
# The objective function from the bound, specialized to 1D.
# ---------------------------------------------------------------------
def objective(m, s):
    """
    Corresponds to the expression inside the braces:
      r(m) + (lambda C^2) / (8n) + (||m||^2)/(2 sigma^2)
        + (d/2) * [ (s^2)/(sigma_s^2) + log(sigma_s^2 / s^2) - 1 ]
        + log(1/delta)
      all divided by lambda, if you like. 
    But for plotting, we can leave out constants that do not depend on (m,s),
    or keep them in.  We'll keep them here, for completeness.
    """
    # first, a constant piece that does not depend on m, s:
    const_piece = (lambda_ * C**2)/(8 * n) + np.log(1.0 / delta)
    # the "m-part":
    m_part = (m**2) / (2.0 * sigma_s**2)
    # the "s-part":
    #   (d/2)*[ s^2/sigma_s^2 + log(sigma_s^2/s^2) -1 ]
    s_term = (d/2.0)*((s**2)/(sigma_s**2) + np.log((sigma_s**2)/(s**2)) - 1.0)

    # Full expression inside braces:
    expr = r_of_m(m) + const_piece + m_part + s_term
    # Then "divide by lambda_" if your final expression is / lambda:
    return expr / lambda_

# ---------------------------------------------------------------------
# Build a grid of (m, s) points for plotting
# ---------------------------------------------------------------------
m_vals = np.linspace(-3, 3, 200)  # range of m
s_vals = np.linspace(0.01, 2.0, 200)  # s>0, from near 0 to 2
M, S = np.meshgrid(m_vals, s_vals)

# Compute the objective on each grid point
Z = np.zeros_like(M)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        m_ij = M[i, j]
        s_ij = S[i, j]
        Z[i, j] = objective(m_ij, s_ij)

# ---------------------------------------------------------------------
# Plot a contour of the (m, s) landscape
# ---------------------------------------------------------------------
plt.figure(figsize=(7,5))
cs = plt.contourf(M, S, Z, levels=50, cmap='viridis')
plt.colorbar(cs, label='Objective value')
plt.xlabel('m')
plt.ylabel('s (must be > 0)')
plt.show()
