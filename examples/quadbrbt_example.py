import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import cache, cached_property
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from gen_qbt.quadbt import QuadBRBTSampler, GeneralizedQuadBTReductor, trapezoidal_rule
from pymor.models.iosys import LTIModel
from models import abc_mimo_mass_spring_damper

# Result is a dict with matrices stored as Keys
# Use `print(sorted(iss.keys()))` to see the stored keys
heat = loadmat("benchmarks/heat-cont.mat")
iss = loadmat("benchmarks/iss.mat")
# print(sorted(iss.keys()))
# Convert sparse matrices to np arrays
A = iss["A"].toarray()  # (270, 270)
B = iss["B"].toarray()  # (270, 3)
C = iss["C"].toarray()  # (3, 270)
n = 270
m = 3
# A = heat["A"].toarray()  # (200, 200)
# B = heat["B"].toarray()  # (200, 1)
# C = heat["C"].toarray()  # (1, 200)
# n = 200
# Enforce SISO below (if desired-)
# B = B[:, 0]
# B = B[:, np.newaxis]
# C = C[0, :]
# C = C[np.newaxis]

# n = 50
# A, B, C = abc_mimo_mass_spring_damper(n, 4, 4, 1)
# Add some input-output feedback
# m = 2
eps = 1e-3
D = np.eye(m) * eps
# D = np.array(eps, ndmin=2)

# To compute Hinf norm... use pymor
ss = LTIModel.from_matrices(A, B, C, D)
gamma = ss.hinf_norm()
# Normalize the system to be bounded-real
C = C / np.sqrt(gamma)
B = B / np.sqrt(gamma)
D = D / gamma

# Sanity check
normalized_ss = LTIModel.from_matrices(A, B, C, D)
print(normalized_ss.hinf_norm())

# Check quadrature error
# First, compute weights/modes via Trapezoidal rule
modesl, modesr, weightsl, weightsr = trapezoidal_rule(
    exp_limits=np.array((-4, 4)), N=400, ordering="interlace"
)

BRBT_sampler = QuadBRBTSampler(A, B, C, D)
QuadBRBTEngine = GeneralizedQuadBTReductor(
    BRBT_sampler, modesl, modesr, weightsl, weightsr, "quadbrbt"
)
# Compute quadrature errors; just make sure its converging
for error in QuadBRBTEngine.quadrature_errors():
    print(error)

# Test Loewner build
Lbar, Mbar = QuadBRBTEngine.Lbar_Mbar
Gbar = QuadBRBTEngine.Gbar
Hbar = QuadBRBTEngine.Hbar
# Does L = L.T * U?
U_bar = BRBT_sampler.right_sqrt_fact_U(modesr, weightsr)
Lh_bar = BRBT_sampler.left_sqrt_fact_Lh(modesl, weightsl)

print("Does Loewner build work? Lbar:", np.linalg.norm((Lh_bar @ U_bar) - Lbar, 2))
print("Does Loewner build work? Mbar:", np.linalg.norm((Lh_bar @ A @ U_bar) - Mbar, 2))
print("Does Loewner build work? Gbar:", np.linalg.norm((C @ U_bar) - Gbar, 2))
print("Does Loewner build work? Hbar:", np.linalg.norm((Lh_bar @ B) - Hbar, 2))

# Compare approximate v. true pr-hsvs
# Compute true hsvs from chol factors of solutions to AREs
# print("Eigs of P", np.linalg.eigvals(BRBT_sampler.P))
# print("Eigs of Q", np.linalg.eigvals(BRBT_sampler.Q))
# Perturb matrices; not SPD due to rounding errors
Ut = np.linalg.cholesky(BRBT_sampler.P + 10e-10 * np.eye(n))
L = np.linalg.cholesky(BRBT_sampler.Q + 10e-10 * np.eye(n))
_, hsv, _ = np.linalg.svd(L.T @ Ut, False)
print("Error in hsvs", np.linalg.norm(hsv - QuadBRBTEngine.hsvbar()[:n]))
# print(np.c_[hsv, QuadBTEngine.hsvbar()[:n]])
# Plot them now

# Aspect ratio is Golden
phi = (np.sqrt(5) + 1) / 2
plt.axes([0.125, 0.15, 0.75, phi - 1])
x = np.arange(n)  # x is 1:n
plt.semilogy(x, hsv, "o")
plt.semilogy(x, QuadBRBTEngine.hsvbar()[:n], "*")
plt.show()
# if __name__ == "__main__":
