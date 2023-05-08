import numpy as np
import scipy as sp


def abc_mimo_mass_spring_damper(n, m, k, d, ph=False):
    # Created by R. Polyuga on September 03, 2008
    # Modified by R. Polyuga on February 10, 2009
    # Modified by R. Polyuga on August 20, 2010
    # Ported from MATLAB to Python by S. Reiter on May 4, 2023
    """
    This function constructs a port-Hamiltonian representation of a mass-spring-damper system
    Namely, J, R, Q, B matrices

    To get system in A, B, C form set
        A = (J - R)*Q, C = B'*Q

    Parameters
    ----------
        n
            Integer dimension of the system (assumed to be even)
        m
            Mass coefficient
        d
            Damping coefficient
        k
            Spring constant
        ph
            Option to return in ph form, default is `False`
    Returns
    -------
        J, R, Q, B
            Port-Hamiltonian representation of the MIMO system, returned as |Numpy Arrays|


    For the simplest n-dimensional MIMO mass-spring-damper system (n is even)
    with inputs (u1, u2) = (F1, F2) being the forces applied to the first
    two masses m1 and m2
    and outputs (y1, y2) = (v1, v2) = (p1/m1, p2/m2) being the velocities
    of the first 2 masses m1 and m2.
    The state vector is x = [q1, p1, q2, p2, ..., q_(n/2), p_(n/2)]^T
    q - the displasment of the mass
    p - the momentum of the mass.
    Port-Hamiltonian matrices for the 6-dim system are of the form

    TODO: Make this a doctest!
    ABC_MIMO_mass_spring_damper(4, 4, 4, 1)
    J =

     0     1     0     0
    -1     0     0     0
     0     0     0     1
     0     0    -1     0


    F =

        0     0     0     0
        0     1     0     0
        0     0     0     0
        0     0     0     1


    Q =

         4.0000     0   -4.0000     0
            0    0.2500     0       0
        -4.0000     0    8.0000     0
            0       0       0    0.2500


    B =

        0     0
        1     0
        0     0
        0     1
    """
    if n % 2 == 0:
        # TODO: Variable coefficients
        # Store physical parameters as numpy arrays
        M = m * np.ones(n // 2)
        D = d * np.ones(n // 2)
        K = k * np.ones(n // 2)

        # Energy matrix Q
        Q = np.zeros([n, n])
        Q[0, 0] = K[1]
        for i, mi in enumerate(M):
            # Diagonals
            if i != 0:
                Q[2 * i, 2 * i] = K[i - 1] + K[i]
            Q[(2 * i) + 1, (2 * i) + 1] = 1 / M[i]
            # Off-diagonals
            if i != (n // 2) - 1:
                Q[2 * i, (2 * i) + 2] = -K[i]
                Q[(2 * i) + 2, 2 * i] = -K[i]

        # Dissipation matrix R
        R = np.zeros([n, n])
        for i, di in enumerate(D):
            R[(2 * i) + 1, (2 * i) + 1] = di

        # Structure matrix J
        J = np.zeros([n, n])
        for i in range(n // 2):
            J[2 * i, 2 * i + 1] = 1
        J = J - J.T

        # Input/output matrix B = C.T, m = p = 2
        B = np.zeros([n, 2])
        B[1, 0] = 1
        if n >= 4:
            B[3, 1] = 1

        if ph is True:
            return J, R, Q, B
        else:
            A = (J - R) @ Q
            C = B.T @ Q
            return A, B, C

    else:
        raise ValueError("System must be Port-Hamiltonian")
