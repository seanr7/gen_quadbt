from functools import cache, cached_property
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from quadbt import *
from ABC_MIMO_mass_spring_damper import abc_mimo_mass_spring_damper

# Build toy positive-real system
n = 8
A, B, C = abc_mimo_mass_spring_damper(n, 4, 4, 1)
# Enforce SISO below (if desired-)
B = B[:, 0]
B = B[:, np.newaxis]
C = C[0, :]
C = C[np.newaxis]
# Add some input-output feedback
eps = 1e-3
D = np.array(eps, ndmin=2)
# D = np.eye(2) * eps

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


# # Compute true hsvs
# Ut = np.linalg.cholesky(bt_sampler.P)
# L = np.linalg.cholesky(bt_sampler.Q)
# _, hsv, _ = np.linalg.svd(L.T @ Ut, False)
# print("Error in hsvs", np.linalg.norm(hsv - QuadBTEngine.hsvbar()[:n]))
# print(np.c_[hsv, QuadBTEngine.hsvbar()[:n]])

# print("Error in pr-hsvs", np.linalg.norm(hsv - QuadBTEngine.hsvbar()[:n]))
# print(np.c_[hsv, QuadBTEngine.hsvbar()[:n]])

# if __name__ == "__main__":
