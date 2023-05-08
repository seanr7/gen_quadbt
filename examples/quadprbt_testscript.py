import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import cache, cached_property
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from gen_qbt.quadbt import *
from ABC_MIMO_mass_spring_damper import abc_mimo_mass_spring_damper

# Build toy positive-real system
n = 50
A, B, C = abc_mimo_mass_spring_damper(n, 4, 4, 1)
# Enforce SISO below (if desired-)
# B = B[:, 0]
# B = B[:, np.newaxis]
# C = C[0, :]
# C = C[np.newaxis]
# Add some input-output feedback
eps = 1e-3
# D = np.array(eps, ndmin=2)
D = np.eye(2) * eps

# Check quadrature error
# First, compute weights/modes via Trapezoidal rule
modesl, modesr, weightsl, weightsr = trapezoidal_rule(
    exp_limits=np.array((-3, 3)), N=200, ordering="interlace"
)

PRBT_sampler = QuadPRBTSampler(A, B, C, D)
QuadPRBTEngine = GeneralizedQuadBTReductor(
    PRBT_sampler, modesl, modesr, weightsl, weightsr, "quadprbt"
)
# Compute quadrature errors; just make sure its converging
for error in QuadPRBTEngine.quadrature_errors():
    print(error)

# Test Loewner build
Lbar, Mbar = QuadPRBTEngine.Lbar_Mbar
Gbar = QuadPRBTEngine.Gbar
Hbar = QuadPRBTEngine.Hbar
# Does L = L.T * U?
U_bar = PRBT_sampler.right_sqrt_fact_U(modesr, weightsr)
Lh_bar = PRBT_sampler.left_sqrt_fact_Lh(modesl, weightsl)

print("Does Loewner build work? Lbar:", np.linalg.norm((Lh_bar @ U_bar) - Lbar, 2))
print("Does Loewner build work? Mbar:", np.linalg.norm((Lh_bar @ A @ U_bar) - Mbar, 2))
print("Does Loewner build work? Gbar:", np.linalg.norm((C @ U_bar) - Gbar, 2))
print("Does Loewner build work? Hbar:", np.linalg.norm((Lh_bar @ B) - Hbar, 2))

# Compare approximate v. true pr-hsvs
# Compute true hsvs from chol factors of solutions to AREs
Ut = np.linalg.cholesky(PRBT_sampler.P)
L = np.linalg.cholesky(PRBT_sampler.Q)
_, hsv, _ = np.linalg.svd(L.T @ Ut, False)
print("Error in hsvs", np.linalg.norm(hsv - QuadPRBTEngine.hsvbar()[:n]))
# print(np.c_[hsv, QuadBTEngine.hsvbar()[:n]])
# Plot them now

# Aspect ratio is Golden
phi = (np.sqrt(5) + 1) / 2
plt.axes([0.125, 0.15, 0.75, phi - 1])
x = np.arange(n)  # x is 1:n
plt.semilogy(x, hsv, "o")
plt.semilogy(x, QuadPRBTEngine.hsvbar()[:n], "*")
plt.show()
# if __name__ == "__main__":
