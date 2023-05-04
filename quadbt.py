from functools import cache, cached_property
import numpy as np
import scipy as sp
# from pymor.algorithms.lyapunov import (
#     _chol,
#     solve_cont_lyap_dense,
#     solve_cont_lyap_lrcf,
#     solve_disc_lyap_dense,
#     solve_disc_lyap_lrcf,
# )
# from pymor.algorithms.riccati import solve_pos_ricc_lrcf, solve_ricc_lrcf
# from pymor.algorithms.projection import project
# from pymor.core.base import BasicObject
# from pymor.models.iosys import LTIModel
# from pymor.models.transfer_function import TransferFunction

class GenericSampleGenerator(object):
    """Class to generate relevant transfer function samples (evaluations, data) for use in quadrature-based balanced truncation
    
    """
    def __init__(self, A, B, C, D):
        self.n = np.shape(A)[0]
        self.m = np.shape[B][1]
        self.p = np.shape[C][0]
        self.I = np.eye([self.n, self.n])
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.sl = sl
        # self.sr = sr
        # # Number of left and right samples 
        # self.number_sl = np.shape(sl)[0]
        # self.number_sr = np.shape(sr)[0]

class QuadPRBTSampler(GenericSampleGenerator):
    """Class to generate relevant transfer function samples for use in Quadrature-Based Positive-real Balanced Truncation (QuadPRBT)

    """
    def __init__(self, A, B, C, D):
        # Only implemented for square systems with p = m 
        if np.shape(C)[0] != np.shape(B)[1]:
            raise NotImplementedError
        elif not np.all(np.linalg.eigvals(D + D.T) > 0):
            raise ValueError('System must be positive real')
        else:
            super().__init__(A, B, C, D)
            self.R = D + D.T
            self.R_sqrt = np.linalg.cholesky(self.R)
            self.Rinv = np.linalg.solve(self.R, np.eye(self.n, self.n))

    @cached_property
    def Qprbt(self):
        # Solve + caches the ARE 
        #   :math: A.T * Qprbt + Qprbt.T * A + (C - B.T * Qprbt).T * Rinv * (C - B.T * Qprbt) = 0
        return sp.linalg.solve_continuous_are(self.A, -1*(self.B), np.zeros(self.n, self.n), -1*(self.R), None, -1*(self.B))
    
    @cached_property
    def Pprbt(self):
        # Solve + caches the ARE 
        #   :math: A * Pprbt + Pprbt.T * A + (Pprbt * C.T - B) * Rinv * (Pprbt * C.T - B).T = 0
        return sp.linalg.solve_continuous_are(self.A.T, self.C.T, np.zeros(self.n, self.n), -1*(self.R), None, -1*(self.B))
    
    @cached_property
    def C_rsf(self):
    # Output matrix of right spectral factor (rsf) of the Popov function
    #   ..math: `G(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        return np.linalg.solve(self.R_sqrt, self.C - np.dot(self.B.T, self.Qprbt))
    
    @cached_property
    def B_lsf(self):
    # Input matrix of left spectral factor (lsf) of the Popov function
    #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        return np.linalg.solve(self.B - np.dot(self.Pprbt, self.C.T), self.R_sqrt)

    @cache
    def sample_rsf(self, s):
        # Artifcially sample the (strictly proper part) of the right spectral factor (rsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        # :math: `M(s)` is an m x m rational transfer function 

        # Get number of samples
        number_s = np.shape(s)[0]
        if self.m == 0: # SISO case
            # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
            Ms = np.zeros(number_s)
            for j in range(number_s): 
                Ms[j, :, :] = np.dot(self.C_rsf, np.linalg.solve((s[j] * self.I - self.A), self.B))
            
            return Ms
        else: # MIMO case
        # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
            Ms = np.zeros([number_s, self.m, self.m])
            for j in range(number_s): 
                Ms[j, :, :] = np.dot(self.C_rsf, np.linalg.solve((s[j] * self.I - self.A), self.B))
            
            return Ms

    @cache
    def sample_lsf(self, s):
        # Artifcially sample the (strictly proper part) of the left spectral factor (lsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        # :math: `N(s)` is an m x m rational transfer function 

        # Get number of samples
        number_s = np.shape(s)[0]
        if self.m == 1: # SISO case
            # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
            Ns = np.zeros(number_s)
            for j in range(number_s): 
                Ns[j] = np.dot(np.linalg.solve(self.C, (s[j] * self.I - self.A), self.B_lsf))

            return Ns
        else: # MIMO case
            # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
            Ns = np.zeros([number_s, self.m, self.m])
            for j in range(number_s): 
                Ns[j, :, :] = np.dot(np.linalg.solve(self.C, (s[j] * self.I - self.A), self.B_lsf))

            return Ns

    @cache 
    def sample_sfcascade(self, s):
        # Artifcially sample the system cascade 
        #   ..math: `H(s) : = [M(s) * N(-s).T]_+ = C_rsf * (s * I - A) \ B_lsf`
        # _+ denotes the stable part of the transfer function

        # Get number of samples
        number_s = np.shape(s)[0]
        if self.m == 1: # SISO case
            # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm. Store as 1d numpy arra
            Hs = np.zeros(number_s)
            for j in range(number_s): 
                Hs[j] = np.dot(np.linalg.solve(self.C_rsf, (s[j] * self.I - self.A), self.B_lsf))

            return Hs
        else: # MIMO case
            # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm. Store as 3d numpy array
            Hs = np.zeros([number_s, self.m, self.m])
            for j in range(number_s): 
                Hs[j, :, :] = np.dot(np.linalg.solve(self.C_rsf, (s[j] * self.I - self.A), self.B_lsf))

            return Hs

class LoewnerManager(object):
    """Class to generate Loewner quadruples from transfer function samples
    
    """
    def Ls(sl, sr, Gsl, Gsr, Hermite=1):
        '''
        Parameters
        ----------
        Hermite
            Binary option (default is 1 / False) to do Hermite interpolation
        '''
        p = np.shape(Gsl)[1]
        m = np.shape(Gsl)[2]


        if Hermite == 0:
            Ls = np.zeros()
        else: # Hermite interpolation not yet implemented
            raise NotImplementedError
        
 
# class GenericQuadBTReductor(BasicObject):
#     """Reductor based on Quadrature-based Balanced Truncation (QuadBT) framework.
    
#     The reductor implements approximate quadrature-based balancing as in :cite: `gosea2022data`.
    
#     Parameters
#     ----------
#     quad_rule
#         |Numpy Array| of shape ((N, 2), (N, 2)) containing left and right quadrature weights/modes or;
#         is an integer N for computing weights and modes via the Trapezoidal rule
#     Hs
#         |Numpy Array| of shape (N, p, m) for MIMO systems with p outputs and m inputs or;
#         |Numpy Array| of shape (N, ) for SISO systems where each |Numpy Array| is a scalar transfer function evaluation or;
#         |TransferFunction| or general `model' class with `transfer_function' attribute
#     """
#     def __init__(self, quad_rule, ):
#         if 
    
def trapezoidalrule(exp_limits=np.array(-3, 3), N=400, ordering='same'):
    """Prepare quadrature modes/weights according to the composite Trapezoidal rule.
    For use in QuadBT, integral representations of Gramians along the imaginary axis.
        
    Parameters
    ----------
    exp_limits
        |Numpy array| of shape (2,) containing the limits of integration. 
        If a = exp_limits[0] and b = exp_limits[1], the limits are 10**a and 10**b
    N
        Integer number of quadrature modes used in composite trapezoidal rules. 
        Assume the same number of modes are used for `left' and `right' rules
    ordering
        The ordering of quadrature modes with respect to each other. Options are:
        -`'same'` uses the same set of modes/weights for each quadrature rule
        -`'interlace'` interlaces the `l', `r' modes along the imaginary axis, and computes the weights accordingly.
        Defaults to 'same'.
    
    Returns
    -------
    modesl
        |Numpy array| of shape (2N, ) containing logarithmically spaced points along the imaginary axis, closed under complex conjugation. `l' modes
    modesr
        |Numpy array| of shape (2N, ) containing logarithmically spaced points along the imaginary axis, closed under complex conjugation. `r' modes
    weightsl
        |Numpy array| of shape (2N, ) containing weights affiliated with `l` modes
    weightsr
        |Numpy array| of shape (2N, ) containing weights affiliated with `r` modes
    """

    if ordering == 'same':
        modes = 1j*np.logspace(exp_limits[0], exp_limits[1], N)
        # Add complex conjugates
        modes = np.r_[np.conjugate(modes)[::-1], modes]
        weights = np.r_[modes[1]-modes[0], modes[2:]-modes[0:-2], modes[-1:]-modes[-2:-1]] * (1/2)
        return modes, modes, weights, weights
    elif ordering == 'interlace':
        modes = 1j*np.logspace(exp_limits[0], exp_limits[1], 2*N)
        # Interlace
        modesl = modes[0::2]
        modesr = modes[1::2]
        # Add complex conjugates
        modesl = np.r_[np.conjugate(modesl)[::-1], modesl]
        modesr = np.r_[np.conjugate(modesr)[::-1], modesr]
        weightsl = np.r_[modesl[1]-modesl[0], modesl[2:]-modesl[0:-2], modesl[-1:]-modesl[-2:-1]] * (1/2)
        weightsr = np.r_[modesr[1]-modesr[0], modesr[2:]-modesr[0:-2], modesr[-1:]-modesr[-2:-1]] * (1/2)
        return modesl, modesr, weightsl, weightsr
    else:
        raise NotImplementedError

# def build_loewner():
#     """
#     """

if __name__ == '__main__':
    run(main)