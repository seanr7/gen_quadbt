from functools import cache, cached_property
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from quadbt import *
from ABC_MIMO_mass_spring_damper import abc_mimo_mass_spring_damper


if __name__ == "__main__":
    n = 100
    A, B, C = abc_mimo_mass_spring_damper(n, 4, 4, 1)
    # Add some input-output feedback
    # B = B[:, 0]
    # C = C[0, :]
    eps = 1e-3
    # D = np.array(eps, ndmin = 2)
    D = np.eye(2) * eps

    # Check quadrature error
    # First, compute weights/modes via Trapezoidal rule
    modesl, modesr, weightsl, weightsr = trapezoidal_rule()

    sampler = QuadPRBTSampler(A, B, C, D)
    QuadPRBTEngine = GeneralizedQuadBTReductor(
        sampler, modesl, modesr, weightsl, weightsr, "quadprbt"
    )
    for error in QuadPRBTEngine.quadrature_errors():
        print(error)
