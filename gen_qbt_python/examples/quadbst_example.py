import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functools import cache, cached_property
import numpy as np
import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt
from gen_qbt.quadbt import QuadBSTSampler, GeneralizedQuadBTReductor, trapezoidal_rule

# Result is a dict with matrices stored as Keys
# Use `print(sorted(iss.keys()))` to see the stored keys
iss = loadmat("benchmarks/iss.mat")

# print(sorted(iss.keys()))
# Convert sparse matrices to np arrays
# A = iss["A"].toarray()  # (270, 270)
# B = iss["B"].toarray()  # (270, 3)
# C = iss["C"].toarray()  # (3, 270)
# n = 270
# Enforce SISO below (if desired-)
# B = B[:, 0]
# B = B[:, np.newaxis]
# C = C[0, :]
# C = C[np.newaxis]
# Add some input-output feedback
# eps = 1e-3
# D = np.array(eps, ndmin=2)
# D = np.eye(3) * eps 

heat = loadmat("benchmarks/heat-cont.mat")
n = 200
A = heat["A"].toarray()  # (200, 200)
B = heat["B"].toarray()  # (200, 1) 
C = heat["C"].toarray()  # (1, 200)
# Add some input-output feedback
eps = 10
D = np.array(eps, ndmin=2)
# D = np.eye(1) * eps


# Check quadrature error
# First, compute weights/modes via Trapezoidal rule
modesl, modesr, weightsl, weightsr = trapezoidal_rule(
    exp_limits=np.array((-3, 3)), N=200, ordering="interlace"
)

BST_sampler = QuadBSTSampler(A, B, C, D)
QuadBSTEngine = GeneralizedQuadBTReductor(
    BST_sampler, modesl, modesr, weightsl, weightsr, "quadbst"
)
# Compute quadrature errors; just make sure its converging
for error in QuadBSTEngine.quadrature_errors():
    print(error)

# Test Loewner build
Lbar, Mbar = QuadBSTEngine.Lbar_Mbar
Gbar = QuadBSTEngine.Gbar
Hbar = QuadBSTEngine.Hbar
# Does L = L.T * U?
U_bar = BST_sampler.right_sqrt_fact_U(modesr, weightsr)
Lh_bar = BST_sampler.left_sqrt_fact_Lh(modesl, weightsl)

print("Does Loewner build work? Lbar:", np.linalg.norm((Lh_bar @ U_bar) - Lbar, 2))
print("Does Loewner build work? Mbar:", np.linalg.norm((Lh_bar @ A @ U_bar) - Mbar, 2))
print("Does Loewner build work? Gbar:", np.linalg.norm((C @ U_bar) - Gbar, 2))
print("Does Loewner build work? Hbar:", np.linalg.norm((Lh_bar @ B) - Hbar, 2))

# Compare approximate v. true pr-hsvs
# Compute true hsvs from chol factors of solutions to AREs
# P = BST_sampler.P
# L, _ = np.linalg.eigh(P)
# print(L)

Ut = np.linalg.cholesky(-BST_sampler.P + 10e-6 * np.eye(200)) 
# print(np.linalg.eigvals(BST_sampler.Q + 10e-10 * np.eye(270)))
# Perturb matrix; not SPD due to rounding errors
L = np.linalg.cholesky(-BST_sampler.Q + 10e-6 * np.eye(200))
_, hsv, _ = np.linalg.svd(L.T @ Ut, False)
print("Error in hsvs", np.linalg.norm(hsv - QuadBSTEngine.hsvbar()[:n]))
# print(np.c_[hsv, QuadBTEngine.hsvbar()[:n]])
# Plot them now

# Aspect ratio is Golden
phi = (np.sqrt(5) + 1) / 2
plt.axes([0.125, 0.15, 0.75, phi - 1])
x = np.arange(n)  # x is 1:n

# import matlab.engine 

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Gulliver"
})
plt.semilogy(x, hsv, "o", markersize=2)
plt.semilogy(x, QuadBSTEngine.hsvbar()[:n], "*", markersize=2 )
plt.grid(True)
plt.xlabel(r'index $i$')
plt.ylabel(r'$\sigma_i=\sqrt{\lambda_i(\mathbf{P}\mathbf{Q}_{\mathcal{W}})}$')
plt.show()
# if __name__ == "__main__":
