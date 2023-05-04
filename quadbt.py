from functools import cache, cached_property
import numpy as np
import scipy as sp

class GenericSampleGenerator(object):
    """Class to generate relevant transfer function samples (evaluations, data) for use in quadrature-based balanced truncation
    NOTE: All other Sample Generator classes (children) inherit the methods of this class!
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

    @cache
    def sampleG(self, s):   
        # Artificially sample the (strictly proper part of) the transfer function of the associated LTI model

        # Pre-allocate space for samples, store as 3d numpy array of dim (N, p, m)
        # So, N blocks of p x m transfer function evals
        Gs = np.zeros([np.shape(s)[0], self.m, self.m])
        for j, sj in enumerate(sj):
            Gs[j, :, :] = np.dot(self.C, np.linalg.solve((sj * self.I - self.A), self.B))
        
        return Gs
    
#    _____                 ___________________ _____ 
#   |  _  |               | | ___ \ ___ \ ___ \_   _|
#   | | | |_   _  __ _  __| | |_/ / |_/ / |_/ / | |  
#   | | | | | | |/ _` |/ _` |  __/|    /| ___ \ | |  
#   \ \/' / |_| | (_| | (_| | |   | |\ \| |_/ / | |  
#    \_/\_\\__,_|\__,_|\__,_\_|   \_| \_\____/  \_/  
#                                                    
#                                                    
#    _____                       _                   
#   /  ___|                     | |                  
#   \ `--.  __ _ _ __ ___  _ __ | | ___ _ __         
#    `--. \/ _` | '_ ` _ \| '_ \| |/ _ \ '__|        
#   /\__/ / (_| | | | | | | |_) | |  __/ |           
#   \____/ \__,_|_| |_| |_| .__/|_|\___|_|           
#                         | |                        
#                         |_|                        

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

        # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
        Ms = np.zeros([np.shape(s)[0], self.m, self.m])
        for j, sj in enumerate(s):
            Ms[j, :, :] = np.dot(self.C_rsf, np.linalg.solve((sj * self.I - self.A), self.B))
        
        return Ms

    @cache
    def sample_lsf(self, s):
        # Artifcially sample the (strictly proper part) of the left spectral factor (lsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        # :math: `N(s)` is an m x m rational transfer function 

        # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm 
        Ns = np.zeros([np.shape(s)[0], self.m, self.m])
        for j, sj in enumerate(s):
            Ns[j, :, :] = np.dot(np.linalg.solve(self.C, (sj * self.I - self.A), self.B_lsf))

        return Ns

    @cache 
    def sample_sfcascade(self, s):
        # Artifcially sample the system cascade 
        #   ..math: `H(s) : = [M(s) * N(-s).T]_+ = C_rsf * (s * I - A) \ B_lsf`
        # _+ denotes the stable part of the transfer function

        # Get number of samples
        number_s = np.shape(s)[0]
        # Pre-allocate space for number_sl samples ofN(sl[j]) \in \Cmm. Store as 3d numpy array
        Hs = np.zeros([number_s, self.m, self.m])
        for j in range(number_s): 
            Hs[j, :, :] = np.dot(np.linalg.solve(self.C_rsf, (s[j] * self.I - self.A), self.B_lsf))

        return Hs

#    _                                         
#   | |                                        
#   | |     ___   _____      ___ __   ___ _ __ 
#   | |    / _ \ / _ \ \ /\ / / '_ \ / _ \ '__|
#   | |___| (_) |  __/\ V  V /| | | |  __/ |   
#   \_____/\___/ \___| \_/\_/ |_| |_|\___|_|   
#                                              
#                                              
#   ___  ___                                   
#   |  \/  |                                   
#   | .  . | __ _ _ __   __ _  __ _  ___ _ __  
#   | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '__| 
#   | |  | | (_| | | | | (_| | (_| |  __/ |    
#   \_|  |_/\__,_|_| |_|\__,_|\__, |\___|_|    
#                              __/ |           
#                             |___/                    

class BTLoewnerManager:
    """Class to generate Loewner quadruples from transfer function samples
    
    """
    def L_Ls(sl, sr, Gsl, Gsr, weightsl, weightsr, Hermite=1):
        '''
        # TODO: Doctest!
        Build scaled Loewner matrix L with entries defined by
            ..math: `L[k, j, :. :] = -lscale[k] * lscale[j](Gsl[k, :, :] - Gsr[j, :, :]) ./ (sl[k] - sr[j])`
        And scaled shifted Loewner matrix Ls with extries defined by 
            ..math: `L[k, j, :. :] = -lscale[k] * lscale[j](sl[k] * Gsl[k, :, :] - sr[j] * Gsr[j, :, :]) ./ (sl[k] - sr[j])`

        Used in building Er, Ar

        Parameters
        ----------
        Hermite
            Binary option (default is 1 / False) to do Hermite interpolation

        Assumptions
        -----------
        Gsl and Gsr are generated by the same transfer function
        '''

        # Prep data in SISO case
        # Code below assumes tf samples are stored as 3d numpy arrays of dim (N, p, m)
        if len(np.shape(Gsl)) == 1: 
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        if Hermite == 0:
            # Output of broadcast is a (Nl, Nr, p, m) np.array
            L = Gsl[:, np.newaxis] - Gsr[np.newaxis]
            # Now, this differ sl - sr of size (Nl, Nr) is broadcast and divided into each (p, m) `entry` of L 
            L /= (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            L *= -(weightsl[:, np.newaxis] - weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Ls = sl[np.newaxis, np.newaxis] * Gsl[:, np.newaxis] - sr[np.newaxis, np.newaxis] * Gsr[np.newaxis]
            Ls /= (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Ls *= -(weightsl[:, np.newaxis] - weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]

            # `Unpack` into 2d numpy arrays
            L = np.concatenate(np.concatenate(L, axis=1), axis=1)
            Ls = np.concatenate(np.concatenate(Ls, axis=1), axis=1)
            return L, Ls
        else: # Hermite interpolation not yet implemented
            raise NotImplementedError
            
    def g(Gsr, weightsr):
        # TODO: Doctest!
        # Prep data in SISO case
        # Code below assumes tf samples are stored as 3d numpy arrays of dim (N, p, m)
        if len(np.shape(Gsl)) == 1: 
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        Gsr = Gsr[np.newaxis] * weightsr[np.newaxis, :, np.newaxis, np.newaxis]
        # Gsr now a 4d numpy array: `(1 x N) matrix with (p x m) entries`
        # `Unpack` into 2d numpy array: (p x (N * m)) matrix
        return np.concatenate(np.concatenate(Gsr, axis=0), axis=1)

    def h(Gsl, weightsl): 
        # TODO: Doctest!
        # Prep data in SISO case
        # Code below assumes tf samples are stored as 3d numpy arrays of dim (N, p, m)
        if len(np.shape(Gsl)) == 1: 
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        Gsl = (weightsl[:, np.newaxis, np.newaxis, np.newaxis] * Gsl[:, np.newaxis])
        # Gsl now a 4d numpy array: `(N x 1) matrix with (p x m) entries'
        # `Unpack` into 2d numpy array: ((N * p) x m) matrix
        return np.concatenate(np.concatenate(Gsl, axis=1), axis=1)
 
    
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
    
