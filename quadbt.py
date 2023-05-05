from functools import cache, cached_property
import numpy as np
import scipy as sp

#    _____                           _ _             _
#   |  __ \                         | (_)           | |
#   | |  \/ ___ _ __   ___ _ __ __ _| |_ _______  __| |
#   | | __ / _ \ '_ \ / _ \ '__/ _` | | |_  / _ \/ _` |
#   | |_\ \  __/ | | |  __/ | | (_| | | |/ /  __/ (_| |
#    \____/\___|_| |_|\___|_|  \__,_|_|_/___\___|\__,_|
#
#
#    _____                 _______ _____  ______         _            _
#   |  _  |               | | ___ \_   _| | ___ \       | |          | |
#   | | | |_   _  __ _  __| | |_/ / | |   | |_/ /___  __| |_   _  ___| |_ ___  _ __
#   | | | | | | |/ _` |/ _` | ___ \ | |   |    // _ \/ _` | | | |/ __| __/ _ \| '__|
#   \ \/' / |_| | (_| | (_| | |_/ / | |   | |\ \  __/ (_| | |_| | (__| || (_) | |
#    \_/\_\\__,_|\__,_|\__,_\____/  \_/   \_| \_\___|\__,_|\__,_|\___|\__\___/|_|
#
#


class GeneralizedQuadBTReductor(object):
    """Reductor class to implement the data-driven `Generalized` Quadrature-based Balanced Truncation (QuadBT).

    QuadBT was originally presented in :cite: `gosea2022data`

    This code handles a generalized version that extends to Positive-real Balanced Truncation (PRBT) :cite: `desai1984transformation`,
    Balanced Stochastic Truncation (BST) :cite: `desai1984transformation`, Bounded-real Balanced Truncation (BRBT) :cite: `opdenacker1988contraction`
    and Frequency-weighted Balanced Truncation (FWBT) :cite: `enns1984model`.

    Parameters
    ----------
    sampler
        Child of |GeneralizedSampler| class to synthetically generate transfer function `data` relevant to the approximate balancing being performed
    modesl
        `Left` interpolation points (quadrature modes) as an (Nl, ) dim |Numpy Array|.
        Implicitly quadrature modes used in approximating the `observability` Gramian Q relevant to the underlying balancing
            Use `left` designation because these points are used in the approximate `left` quadrature-based square-root factor L; Q ~ L * L.T
    modesr
        `Right` interpolation point (quadrature modes) as an (Nr, ) dim |Numpy Array|.
        Implicitly quadrature modes used in approximating the `reachability` Gramian P relevant to the underlying balancing
            Use `right` designation because these points are used in the approximate `right` quadrature-based square-root factor U; P ~ U * U.T
    weightsl
        `Left` quadrature weights as an (Nl, ) dim |Numpy Array|.
    weightsr
        `Right` quadrature weights as an (Nr, ) dim |Numpy Array|.
    typ
        `str` indicating the type of balancing

    Given fixed pairs of modes/weights; the only effective difference in performing data-driven balancing is the |GeneralizedSampler| class passed
    """

    def __init__(self, sampler, modesl, modesr, weightsl, weightsr, typ):
        # Takes instances of `Loewner_Manager` class and child of `GenericSampleGenerator` class
        # For purpose of generating relevant tf samples and building the relevant Leowner matrices from said samples
        # The only thing that changes between the different types of QuadBT is the sampler given
        self.sampler = sampler
        # Save prepped quadrature modes/weights with instance of the class
        self.modesl = modesl
        self.modesr = modesr
        self.weightsl = weightsl
        self.weightsl = weightsr
        assert typ in ("quadbt", "quadbrbt", "quadprbt", "quadbst", "quadfwbt")
        self.typ = typ

    #                                  _
    #    |   _   _       ._   _  ._   |_ ._   _  o ._   _
    #    |_ (_) (/_ \/\/ | | (/_ |    |_ | | (_| | | | (/_
    #                                         _|

    @staticmethod
    def _Lbar_Mbar(sl, sr, Gsl, Gsr, weightsl, weightsr, Hermite=False):
        """
        # TODO: Doctests!
        Build scaled Loewner matrix Lbar with entries defined by
            ..math: `Lbar[k, j, :. :] = -weightsl[k] * weightsr[j](Gsl[k, :, :] - Gsr[j, :, :]) ./ (sl[k] - sr[j])`
        And scaled shifted Loewner matrix Mbar with extries defined by
            ..math: `Mbar[k, j, :. :] = -weightsl[k] * weightsr[j](sl[k] * Gsl[k, :, :] - sr[j] * Gsr[j, :, :]) ./ (sl[k] - sr[j])`

        Lbar replaces the product of exact square-root factors (and thus its svs approximate the true hsvs) in QuadBT;
            ..math: `Lbar = L.T * U`
        Mbar is used in building the reduced Ar matrix in QuadBT;
            ..math: `Mbar = L.T * A * U`

        Parameters
        ----------
        sl
            `Left` interpolation points (quadrature modes) as an (Nl, ) dim |Numpy Array|.
        sr
            `Right` interpolation point (quadrature modes) as an (Nr, ) dim |Numpy Array|.
        Gsl
            `Left` transfer function data as (Nl, p, m) |Numpy Array|
        Gsr
            `Right` transfer function dataa as (Nr, p, m) |Numpy Array|
        weightsl
            `Left` quadrature weights as an (Nl, ) dim |Numpy Array|.
        weightsr
            `Right` quadrature weights as an (Nr, ) dim |Numpy Array|.
        Hermite
            Binary option (default is False) to do Hermite interpolation
            TODO: Implement this option

        Returns
        -------
        Lbar
            Scaled Loewner matrix as (Nl, Nr) |Numpy Array|
        Mbar
            Scaled shifted-Loewner matrix as (Nl, Nr) |Numpy Array|

        Assumptions
        -----------
        Gsl and Gsr are generated by the same transfer function
        """

        # Prep data in the SISO case
        if len(np.shape(Gsl)) == 1:
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        if Hermite is False:
            # Output of broadcast is a (Nl, Nr, p, m) np.array
            Lbar = Gsl[:, np.newaxis] - Gsr[np.newaxis]
            # Now, this differ sl - sr of size (Nl, Nr) is broadcast and divided into each (p, m) `entry` of L
            Lbar /= (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Lbar *= -(weightsl[:, np.newaxis] - weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Mbar = (
                sl[np.newaxis, np.newaxis] * Gsl[:, np.newaxis]
                - sr[np.newaxis, np.newaxis] * Gsr[np.newaxis]
            )
            Mbar /= (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Mbar *= -(weightsl[:, np.newaxis] - weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]

            # `Unpack` into 2d numpy arrays
            Lbar = np.concatenate(np.concatenate(Lbar, axis=1), axis=1)
            Mbar = np.concatenate(np.concatenate(Mbar, axis=1), axis=1)
            return Lbar, Mbar
        else:  # Hermite interpolation not yet implemented
            raise NotImplementedError

    @staticmethod
    def _Gbar(Gsr, weightsr):
        """
        TODO: Doctest!
        Build output matrix in Loewner quadruple
            ..math: `Gbar[k, :, :] = -weightsr[k] * Gsr[k, : :]`
        Gbar is used in constructing the reduced Cr matrix in QuadBT;
            ..math: `Gbar = C * U`

        Parameters
        ----------
        Gsr
            `Right` transfer function dataa as (Nr, p, m) |Numpy Array|
        weightsr
            `Right` quadrature weights as an (Nr, ) dim |Numpy Array|.

        Returns
        -------
        Gbar
            Scaled output matrix in Loewner quadruple as (p, (Nr * m)) |Numpy Array|
        """

        # Prep data in SISO case
        if len(np.shape(Gsl)) == 1:
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        Gsr = Gsr[np.newaxis] * weightsr[np.newaxis, :, np.newaxis, np.newaxis]
        # Gsr now a 4d numpy array: `(1 x Nr) matrix with (p x m) entries`
        # `Unpack` into 2d numpy array: (p x (Nr * m)) matrix
        return np.concatenate(np.concatenate(Gsr, axis=0), axis=1)

    @staticmethod
    def _Hbar(Gsl, weightsl):
        """
        TODO: Doctest!
        Build output matrix in Loewner quadruple
            ..math: `Hbar[j, :, :] = -weightsl[j] * Gsl[j, : :]`
        Hbar is used in constructing the reduced Br matrix in QuadBT;
            ..math: `Hbar = L.T * B`

        Parameters
        ----------
        sl
            `Left` interpolation points (quadrature modes) as an (Nl, ) dim |Numpy Array|.
        Gsl
            `Left` transfer function data as (Nl, p, m) |Numpy Array|
        weightsl
            `Left` quadrature weights as an (Nl, ) dim |Numpy Array|.

        Returns
        -------
        Hbar
            Scaled input matrix in Loewner quadruple as ((Nl * p), m) |Numpy Array|
        """

        # Prep data in SISO case
        if len(np.shape(Gsl)) == 1:
            Gsl = Gsl[:, np.newaxis, np.newaxis]
        if len(np.shape(Gsr)) == 1:
            Gsr = Gsr[:, np.newaxis, np.newaxis]

        Gsl = weightsl[:, np.newaxis, np.newaxis, np.newaxis] * Gsl[:, np.newaxis]
        # Gsl now a 4d numpy array: `(Nl x 1) matrix with (p x m) entries'
        # `Unpack` into 2d numpy array: ((Nl * p) x m) matrix
        return np.concatenate(np.concatenate(Gsl, axis=1), axis=1)

    # Key BT quantities from relevant data are computed once then cached, can be recycled for different orders of reduction
    @cached_property
    def Lbar_Mbar(self):
        if self.typ is "quadbrbt":
            raise NotImplementedError
        elif self.typ in ("quadbt", "quadprbt", "quadbst", "quadfwbt"):
            return _Lbar_Mbar(
                self.sl,
                self.sr,
                self.sampler.sample_sfcascade(self.sl),  # Left samples
                self.sampler.sample_sfcascade(self.sr),  # Right samples
                self.weightsl,
                self.weightsr,
            )

    @cached_property
    def Gbar(self):
        if self.typ is "quadbrbt":
            raise NotImplementedError
        elif self.typ in ("quadbt", "quadprbt", "quadbst", "quadfwbt"):
            return _Gbar(self.sampler.sample_rsf(self.sr), self.weightsr)

    @cached_property
    def Hbar(self):
        if self.typ is "quadbrbt":
            raise NotImplementedError
        elif self.typ in ("quadbt", "quadprbt", "quadbst", "quadfwbt"):
            return _Hbar(self.sampler.sample_lsf(self.sl), self.weightsl)

    @cached_property
    def svd_from_data(self):
        # Compute the SVD once, then cache it
        Zbar, sbar, Yhbar = np.linalg.svd(self.Lbar, full_matrices=False)
        return Zbar, sbar, Yhbar.H

    def reduce(self, r):
        # Compute SVD of Loewner matrix L
        Zbar, sbar, Ybar = self.svd_from_data
        # Build Petrov-Galerkin Projection matrices and project onto intermediate quantities from data
        Sbar1_invsqrt = np.diag(1 / np.sqrt(sbar[1:r]))
        Arbar = Sbar1_invsqrt @ Zbar[:, 1:r].H @ self.Mbar @ Ybar[:, 1:r] @ Sbar1_invsqrt
        Brbar = Sbar1_invsqrt @ Zbar[:, 1:r].H @ self.Hbar
        Crbar = self.Gbar @ Ybar[:, 1:r] @ Sbar1_invsqrt

        return Arbar, Crbar, Brbar

    def quadrature_errors(self):
        # Return error in underlying quadrature rules
        print("Error in Q")
        yield np.linalg.norm(
            self.sampler.Q
            - np.dot(self.sampler.left_sqrt_fact_Lh.H * self.sampler.left_sqrt_fact_Lh)
        )
        print("Error in P")
        yield np.linalg.norm(
            self.sampler.P
            - np.dot(self.sampler.right_sqrt_fact_U * self.sampler.left_sqrt_fact_U.H)
        )


#    _____                       _
#   /  ___|                     | |
#   \ `--.  __ _ _ __ ___  _ __ | | ___ _ __ ___
#    `--. \/ _` | '_ ` _ \| '_ \| |/ _ \ '__/ __|
#   /\__/ / (_| | | | | | | |_) | |  __/ |  \__ \
#   \____/ \__,_|_| |_| |_| .__/|_|\___|_|  |___/
#                         | |
#                         |_|


class GenericSampleGenerator(object):
    """Class to generate relevant transfer function samples (evaluations, data) for use in quadrature-based balanced truncation
    NOTE: All other Sample Generator classes (children) inherit the methods of this class!

    Paramters
    ---------
    A, B, C, D
        State-space realization of the FOM, all matrices are |Numpy Arrays| of size (n, n), (n, m), (p, n), and (p, m) respectively
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

    # @cache
    def sampleG(self, s):
        # Artificially sample the (strictly proper part of) the transfer function of the associated LTI model

        # Pre-allocate space for samples, store as 3d numpy array of dim (N, p, m)
        # So, N blocks of p x m transfer function evals
        Gs = np.zeros([np.shape(s)[0], self.m, self.m])
        for j, sj in enumerate(sj):
            Gs[j, :, :] = np.dot(self.C, np.linalg.solve((sj * self.I - self.A), self.B))

        return Gs


#     _               _ ___
#    / \      _.  _| |_) |
#    \_X |_| (_| (_| |_) |
#


class QuadBTSampler(GenericSampleGenerator):
    """ """

    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D)

    @cached_property
    def Q(self):
        # Solve + cache ALE
        #   :math: A.T * Q + Q * A + C.T * C = 0
        return sp.linalg.solve_continuous_lyapunov(self.A.T, -self.C.T * self.C)

    @cached_property
    def P(self):
        # Solve + cache ALE
        #   :math: A * P + P.T * A + B *B.T = 0
        return sp.linalg.solve_continuous_lyapunov(self.A, -self.B * self.B.T)

    def sample_rsf(self, s):
        return self.sampleG(s)

    def sample_lsf(self, s):
        return self.sampleG(s)

    def sample_sfcascade(self, s):
        return self.sampleG(s)

    def right_sqrt_fact_U(self, sr, weightsr):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `P \approx U * Uh,  U \in \C^{n \times (Nr * m)}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sr)[0] == np.shape(weightsr)[0]
        U = np.zeros([self.n, np.shape(sr)[0] * self.m])
        for j, sr_j in enumerate(sr):
            U[:, j] = weightsr[j] * np.linalg.solve((sr_j * self.I - self.A), self.B)

        return U

    def left_sqrt_fact_Lh(self, sl, weightsl):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `Qprbt \approx L * Lh,  Lh \in \C^{(p * Nl) \times n}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sl)[0] == np.shape(weightsl)[0]
        Lh = np.zeros([np.shape(sl)[0] * self.p, self.n])
        for k, sl_k in enumerate(sl):
            Lh[k, :] = weightsl[k] * np.dot(
                self.C, np.linalg.solve((sl_k * self.I - self.A), self.I)
            )

        return Lh


#     _               _   _   _ ___
#    / \      _.  _| |_) |_) |_) |
#    \_X |_| (_| (_| |   | \ |_) |
#


class QuadPRBTSampler(GenericSampleGenerator):
    """Class to generate relevant transfer function samples for use in Quadrature-Based Positive-real Balanced Truncation (QuadPRBT)

    Paramters
    ---------
    A, B, C, D
        State-space realization of the FOM, all matrices are |Numpy Arrays| of size (n, n), (n, m), (p, n), and (p, m) respectively
    """

    def __init__(self, A, B, C, D):
        # Only implemented for square systems with p = m
        if np.shape(C)[0] != np.shape(B)[1]:
            raise NotImplementedError
        elif not np.all(np.linalg.eigvals(D + D.T) > 0):
            raise ValueError("System must be positive real")
        else:
            super().__init__(A, B, C, D)
            self.R = D + D.T
            self.R_sqrt = np.linalg.cholesky(self.R)
            self.Rinv = np.linalg.solve(self.R, np.eye(self.n, self.n))

    @cached_property
    def Q(self):
        # Solve + cache ARE
        #   :math: A.T * Qprbt + Qprbt.T * A + (C - B.T * Qprbt).T * Rinv * (C - B.T * Qprbt) = 0
        return sp.linalg.solve_continuous_are(
            self.A, -1 * (self.B), np.zeros(self.n, self.n), -1 * (self.R), None, -1 * (self.B)
        )

    @cached_property
    def P(self):
        # Solve + cache ARE
        #   :math: A * Pprbt + Pprbt.T * A + (Pprbt * C.T - B) * Rinv * (Pprbt * C.T - B).T = 0
        return sp.linalg.solve_continuous_are(
            self.A.T, self.C.T, np.zeros(self.n, self.n), -1 * (self.R), None, -1 * (self.B)
        )

    @cached_property
    def C_rsf(self):
        # Output matrix of right spectral factor (rsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        return np.linalg.solve(self.R_sqrt, self.C - np.dot(self.B.T, self.Q))

    @cached_property
    def B_lsf(self):
        # Input matrix of left spectral factor (lsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        return np.linalg.solve(self.B - np.dot(self.P, self.C.T), self.R_sqrt)

    # @cache
    def sample_rsf(self, sl):
        # Artifcially sample the (strictly proper part) of the right spectral factor (rsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        # :math: `M(s)` is an m x m rational transfer function
        # In QuadPRBT, these samples are used in building the reduced Br = Lprbt.T * B

        Ms = np.zeros([np.shape(sl)[0], self.m, self.m])
        for k, sl_k in enumerate(sl):
            Ms[k, :, :] = np.dot(self.C_rsf, np.linalg.solve((sl_k * self.I - self.A), self.B))

        return Ms

    # @cache
    def sample_lsf(self, sr):
        # Artifcially sample the (strictly proper part) of the left spectral factor (lsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        # :math: `N(s)` is an m x m rational transfer function
        # In QuadPRBT, these samples are used in building the reduced Cr = C * Uprbt

        Ns = np.zeros([np.shape(sr)[0], self.m, self.m])
        for j, sr_j in enumerate(sr):
            Ns[j, :, :] = np.dot(self.C, np.linalg.solve((sr_j * self.I - self.A), self.B_lsf))

        return Ns

    # @cache
    def sample_sfcascade(self, s):
        # Artifcially sample the system cascade
        #   ..math: `H(s) : = [M(s) * N(-s).T]_+ = C_rsf * (s * I - A) \ B_lsf`
        # _+ denotes the stable part of the transfer function
        # In QuadPRBT, these samples are used in building the reduced Ar = Lprbt.T * A * Uprbt and approximate Hankel singular values

        Hs = np.zeros([np.shape(s)[0], self.m, self.m])
        for j, sj in enumerate(s):
            Hs[j, :, :] = np.dot(self.C_rsf, np.linalg.solve((sj * self.I - self.A), self.B_lsf))

        return Hs

    def right_sqrt_fact_U(self, sr, weightsr):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `Pprbt \approx U * Uh,  U \in \C^{n \times (Nr * m)}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sr)[0] == np.shape(weightsr)[0]
        U = np.zeros([self.n, np.shape(sr)[0] * self.m])
        for j, sr_j in enumerate(sr):
            U[:, j] = weightsr[j] * np.linalg.solve((sr_j * self.I - self.A), self.B_lsf)

        return U

    def left_sqrt_fact_Lh(self, sl, weightsl):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `Qprbt \approx L * Lh,  Lh \in \C^{(p * Nl) \times n}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sl)[0] == np.shape(weightsl)[0]
        Lh = np.zeros([np.shape(sl)[0] * self.p, self.n])
        for k, sl_k in enumerate(sl):
            Lh[k, :] = weightsl[k] * np.dot(
                self.C_rsf, np.linalg.solve((sl_k * self.I - self.A), self.I)
            )

        return Lh


def trapezoidalrule(exp_limits=np.array((-3, 3)), N=400, ordering="same"):
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

    if ordering == "same":
        modes = 1j * np.logspace(exp_limits[0], exp_limits[1], N)
        # Add complex conjugates
        modes = np.r_[np.conjugate(modes)[::-1], modes]
        weights = np.r_[modes[1] - modes[0], modes[2:] - modes[0:-2], modes[-1:] - modes[-2:-1]] * (
            1 / 2
        )
        return modes, modes, weights, weights
    elif ordering == "interlace":
        modes = 1j * np.logspace(exp_limits[0], exp_limits[1], 2 * N)
        # Interlace
        modesl = modes[0::2]
        modesr = modes[1::2]
        # Add complex conjugates
        modesl = np.r_[np.conjugate(modesl)[::-1], modesl]
        modesr = np.r_[np.conjugate(modesr)[::-1], modesr]
        weightsl = np.r_[
            modesl[1] - modesl[0], modesl[2:] - modesl[0:-2], modesl[-1:] - modesl[-2:-1]
        ] * (1 / 2)
        weightsr = np.r_[
            modesr[1] - modesr[0], modesr[2:] - modesr[0:-2], modesr[-1:] - modesr[-2:-1]
        ] * (1 / 2)
        return modesl, modesr, weightsl, weightsr
    else:
        raise NotImplementedError
