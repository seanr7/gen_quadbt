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

    def __init__(self, sampler, modesl, modesr, weightsl, weightsr, typ="quadbt"):
        # Takes instances of `Loewner_Manager` class and child of `GenericSampleGenerator` class
        # For purpose of generating relevant tf samples and building the relevant Leowner matrices from said samples
        # The only thing that changes between the different types of QuadBT is the sampler given
        self.sampler = sampler
        # Save prepped quadrature modes/weights with instance of the class
        self.modesl = modesl
        self.modesr = modesr
        self.weightsl = weightsl
        self.weightsr = weightsr
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
            Lbar = Lbar / (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Lbar = (
                Lbar
                * -(weightsl[:, np.newaxis] * weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            )
            # Output of broadcast is a (Nl, Nr, p, m) np.array
            Mbar = (sl[:, np.newaxis, np.newaxis] * Gsl)[:, np.newaxis] - (
                sr[:, np.newaxis, np.newaxis] * Gsr
            )[np.newaxis]
            Mbar = Mbar / (sl[:, np.newaxis] - sr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            Mbar = (
                Mbar
                * -(weightsl[:, np.newaxis] * weightsr[np.newaxis])[:, :, np.newaxis, np.newaxis]
            )
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
            ..math: `Gbar[k, :, :] = weightsr[k] * Gsr[k, : :]`
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
            ..math: `Hbar[j, :, :] = weightsl[j] * Gsl[j, : :]`
        Hbar is used in constructing the reduced Br matrix in QuadBT;
            ..math: `Hbar = Lh * B`

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

        Gsl = weightsl[:, np.newaxis, np.newaxis, np.newaxis] * Gsl[:, np.newaxis]
        # Gsl now a 4d numpy array: `(Nl x 1) matrix with (p x m) entries'
        # `Unpack` into 2d numpy array: ((Nl * p) x m) matrix
        return np.concatenate(np.concatenate(Gsl, axis=1), axis=1)

    # Key BT quantities from relevant data are computed once then cached, can be recycled for different orders of reduction
    @cached_property
    def Lbar_Mbar(self):
        # Used in computing A_r = W_r.T @ Mbar @ V_r and _, hsvbar, _ = svd(Lbar)
        if self.typ == "quadbrbt":
            Hsl, Nsl, Msl, Gsl = self.sampler.samples_for_Lbar_Mbar(self.modesl)
            Hsr, Nsr, Msr, Gsr = self.sampler.samples_for_Lbar_Mbar(self.modesr)
            Lbar11, Mbar11 = self._Lbar_Mbar(
                self.modesl, self.modesr, Gsl, Gsr, self.weightsl, self.weightsr
            )
            Lbar12, Mbar12 = self._Lbar_Mbar(
                self.modesl, self.modesr, Nsl, Nsr, self.weightsl, self.weightsr
            )
            Lbar21, Mbar21 = self._Lbar_Mbar(
                self.modesl, self.modesr, Msl, Msr, self.weightsl, self.weightsr
            )
            Lbar22, Mbar22 = self._Lbar_Mbar(
                self.modesl, self.modesr, Hsl, Hsr, self.weightsl, self.weightsr
            )
            Lbar = np.c_[np.r_[Lbar11, Lbar21], np.r_[Lbar12, Lbar22]]
            Mbar = np.c_[np.r_[Mbar11, Mbar21], np.r_[Mbar12, Mbar22]]
            return Lbar, Mbar
        else:
            # Using samples C_lsf @ (s * I - A) \ B_rsf
            return self._Lbar_Mbar(
                self.modesl,
                self.modesr,
                self.sampler.samples_for_Lbar_Mbar(self.modesl),  # Left samples
                self.sampler.samples_for_Lbar_Mbar(self.modesr),  # Right samples
                self.weightsl,
                self.weightsr,
            )

    @cached_property
    def Gbar(self):
        # Used in computing C_r = Gbar @ V_r
        if self.typ == "quadbrbt":
            Nsr, Gsr = self.sampler.samples_for_Gbar(self.modesr)
            Gbar1 = self._Gbar(Gsr, self.weightsr)
            Gbar2 = self._Gbar(Nsr, self.weightsr)
            Gbar = np.c_[Gbar1, Gbar2]
            return Gbar
        else:
            # Using samples C * (s * I - A) \ B_rsf
            return self._Gbar(self.sampler.samples_for_Gbar(self.modesr), self.weightsr)

    @cached_property
    def Hbar(self):
        # Used in computing B_r = W_r.T @ B
        if self.typ == "quadbrbt":
            Msl, Gsl = self.sampler.samples_for_Hbar(self.modesl)
            Hbar1 = self._Hbar(Gsl, self.weightsl)
            Hbar2 = self._Hbar(Msl, self.weightsl)
            Hbar = np.r_[Hbar1, Hbar2]
            return Hbar
        else:
            # Using samples C_lsf * (s * I - A) \ B
            return self._Hbar(self.sampler.samples_for_Hbar(self.modesl), self.weightsl)

    @cached_property
    def svd_from_data(self):
        # Compute the SVD once, then cache it
        Lbar, _ = self.Lbar_Mbar
        Zbar, sbar, Yhbar = np.linalg.svd(Lbar, full_matrices=False)
        return Zbar, sbar, Yhbar.conj().T

    def hsvbar(self):
        # Return cached approximate hsvs, for easier reference
        _, sbar, _ = self.svd_from_data
        return sbar

    def reduce(self, r):
        # Compute SVD of Loewner matrix L
        Zbar, sbar, Ybar = self.svd_from_data
        # Build Petrov-Galerkin Projection matrices and project onto intermediate quantities from data
        Sbar1_invsqrt = np.diag(1 / np.sqrt(sbar[1:r]))
        Arbar = Sbar1_invsqrt @ Zbar[:, 1:r].conj().T @ self.Mbar @ Ybar[:, 1:r] @ Sbar1_invsqrt
        Brbar = Sbar1_invsqrt @ Zbar[:, 1:r].conj().T @ self.Hbar
        Crbar = self.Gbar @ Ybar[:, 1:r] @ Sbar1_invsqrt

        return Arbar, Crbar, Brbar

    def quadrature_errors(self):
        # Return error in underlying quadrature rules
        print("Relative Error in Q")
        Lh = self.sampler.left_sqrt_fact_Lh(self.modesl, self.weightsl)
        yield np.linalg.norm(self.sampler.Q - (Lh.conj().T @ Lh), 2) / np.linalg.norm(
            self.sampler.Q, 2
        )

        U = self.sampler.right_sqrt_fact_U(self.modesr, self.weightsr)
        print("Relative Error in P")
        yield np.linalg.norm(self.sampler.P - (U @ U.conj().T), 2) / np.linalg.norm(
            self.sampler.P, 2
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
        if len(np.shape(B)) > 1:
            self.m = np.shape(B)[1]
        else:
            raise ValueError("B must be a 2d |Numpy Array| with dimensions (n, m)")
        if len(np.shape(C)) > 1:
            self.p = np.shape(C)[0]
        else:
            raise ValueError("C must be a 2d |Numpy Array| with dimensions (p, n)")
        if len(np.shape(D)) != 2 and D is not None:
            raise ValueError("D must be a 2d |Numpy Array| with dimensions (p, m)")
        self.I = np.eye(self.n)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def sampleG(self, s):
        # Artificially sample the (strictly proper part of) the transfer function of the associated LTI model

        # Pre-allocate space for samples, store as 3d numpy array of dim (N, p, m)
        # So, N blocks of p x m transfer function evals
        Gs = np.zeros([np.shape(s)[0], self.m, self.m], dtype="complex_")
        for j, sj in enumerate(s):
            Gs[j, :, :] = self.C @ np.linalg.solve((sj * self.I - self.A), self.B)

        return Gs

    def B_rsf(self):
        # Included to be general; in some Child classes, this will simply return self.B
        raise NotImplementedError

    def C_lsf(self):
        # Included to be general; in some Child classes, this will simply return self.C
        raise NotImplementedError

    def right_sqrt_fact_U(self, sr, weightsr):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `P ~ U * Uh,  U \in \C^{n \times (Nr * m)}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sr)[0] == np.shape(weightsr)[0]
        U = np.zeros([np.shape(sr)[0], self.n, self.m], dtype="complex_")
        for j, sr_j in enumerate(sr):
            U[j, :, :] = weightsr[j] * np.linalg.solve((sr_j * self.I - self.A), self.B_rsf)

        return np.concatenate(U, axis=1)

    def left_sqrt_fact_Lh(self, sl, weightsl):
        # Return approximate quadrature-based sqrt factor
        #   ..math: `Q ~ L * Lh,  Lh \in \C^{(p * Nl) \times n}`
        # Used in checking error of the implicity quadrature rule
        assert np.shape(sl)[0] == np.shape(weightsl)[0]
        Lh = np.zeros([np.shape(sl)[0], self.p, self.n], dtype="complex_")
        for k, sl_k in enumerate(sl):
            Lh[k, :, :] = weightsl[k] * (
                self.C_lsf @ np.linalg.solve((sl_k * self.I - self.A), self.I)
            )

        return np.concatenate(Lh, axis=0)


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
        return sp.linalg.solve_continuous_lyapunov(self.A.T, -(self.C.T @ self.C))

    @cached_property
    def P(self):
        # Solve + cache ALE
        #   :math: A * P + P.T * A + B *B.T = 0
        return sp.linalg.solve_continuous_lyapunov(self.A, -(self.B @ self.B.T))

    @cached_property
    def C_lsf(self):
        # Return output matrix C as self.C_lsf. For use in checkig quadrature error
        return self.C

    @cached_property
    def B_rsf(self):
        # Return input matrix B as self.B_rsf. For use in checking quadrature error
        return self.B

    # Return samples of G from method of parent class
    def samples_for_Gbar(self, s):
        # Samples used in building Gbar = Lh @ B from data
        # Gbar used in computing Br
        return self.sampleG(s)

    def samples_for_Hbar(self, sl):
        # Samples used in building Hbar = C @ U from data
        # Hbar used in computing Cr
        return self.sampleG(sl)

    def samples_for_Lbar_Mbar(self, sr):
        # Samples used in building Lbar = Lh @ U and Mbar = Lh @ A @ U from data
        # Mbar is usded in computing Ar, Lbar used in approximating the hsvs
        return self.sampleG(sr)


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
        super().__init__(A, B, C, D)
        if self.m != self.p:
            raise NotImplementedError
        self.R = D + D.T
        if self.m > 1:  # MIMO case
            if not np.all(np.linalg.eigvals(self.R) > 0):
                raise ValueError("System must be positive real")
            self.Rinv = np.linalg.solve(self.R, np.eye(self.m))
            self.R_invsqrt = np.linalg.cholesky(self.Rinv)
        else:  # SISO case
            if not (self.R > 0):
                raise ValueError("System must be positive real")
            self.Rinv = 1 / self.R
            self.R_invsqrt = np.sqrt(self.Rinv)

    @cached_property
    def Q(self):
        # Solve + cache ARE
        #   :math: A.T * Qprbt + Qprbt.T * A + (C - B.T * Qprbt).T * Rinv * (C - B.T * Qprbt) = 0
        return sp.linalg.solve_continuous_are(
            self.A, -1 * (self.B), np.zeros([self.n, self.n]), -1 * (self.R), self.I, self.C.T
        )

    @cached_property
    def P(self):
        # Solve + cache ARE
        #   :math: A * Pprbt + Pprbt.T * A + (Pprbt * C.T - B) * Rinv * (Pprbt * C.T - B).T = 0
        return sp.linalg.solve_continuous_are(
            self.A.T, self.C.T, np.zeros([self.n, self.n]), -1 * (self.R), self.I, -1 * (self.B)
        )

    @cached_property
    def C_lsf(self):
        # Output matrix of left spectral factor (lsf) of the Popov function
        #   ..math: `Phi(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        return self.R_invsqrt @ (self.C - (self.B.T @ self.Q))

    @cached_property
    def B_rsf(self):
        # Input matrix of right spectral factor (rsf) of the Popov function
        #   ..math: `Phi(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        return (self.B - (self.P @ self.C.T)) @ self.R_invsqrt

    def samples_for_Hbar(self, sl):
        # Artifcially sample the (strictly proper part) of the left spectral factor (lsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = M(-s).T*M(s)`
        # :math: `M(s)` is an m x m rational transfer function
        # In QuadPRBT, these samples are used in building Hbar = Lh_prbt @ B from data
        # Called during `GeneralizedQuadBTReductor.Hbar`

        Ms = np.zeros([np.shape(sl)[0], self.m, self.m], dtype="complex_")
        for k, sl_k in enumerate(sl):
            Ms[k, :, :] = self.C_lsf @ np.linalg.solve((sl_k * self.I - self.A), self.B)

        return Ms

    def samples_for_Gbar(self, sr):
        # Artifcially sample the (strictly proper part) of the right spectral factor (rsf) of the Popov function
        #   ..math: `G(s) = G(s) + G(-s).T = N(s)*N(-s).T`
        # :math: `N(s)` is an m x m rational transfer function
        # In QuadPRBT, these samples are used in building Gbar = C @ U_prbt from data
        # Called during `GeneralizedQuadBTReductor.Gbar`

        Ns = np.zeros([np.shape(sr)[0], self.m, self.m], dtype="complex_")
        for j, sr_j in enumerate(sr):
            Ns[j, :, :] = self.C @ np.linalg.solve((sr_j * self.I - self.A), self.B_rsf)

        return Ns

    def samples_for_Lbar_Mbar(self, s):
        # Artifcially sample the system cascade
        #   ..math: `H(s) : = [M(s) * (N(-s).T)^{-1}]_+ = C_lsf * (s * I - A) \ B_rsf`
        # _+ denotes the stable part of the transfer function
        # In QuadPRBT, these samples are used in building Lbar = Lh_prbt @ U_prbt and Mbar = Lh_prbt @ A @ U_prbt
        # Called during `GeneralizedQuadBTReductor.Lbar_Mbar`

        Hs = np.zeros([np.shape(s)[0], self.m, self.m], dtype="complex_")
        for j, sj in enumerate(s):
            Hs[j, :, :] = self.C_lsf @ np.linalg.solve((sj * self.I - self.A), self.B_rsf)

        return Hs


#     _               _   __ ___
#    / \      _.  _| |_) (_   |
#    \_X |_| (_| (_| |_) __)  |
#


class QuadBSTSampler(GenericSampleGenerator):
    """Class to generate relevant transfer function samples for use in Quadrature-Based Balanced Stochastic Truncation (QuadBST)

    Paramters
    ---------
    A, B, C, D
        State-space realization of the FOM, all matrices are |Numpy Arrays| of size (n, n), (n, m), (p, n), and (p, m) respectively
    """

    def __init__(self, A, B, C, D):
        # Only implemented for square systems with p = m
        super().__init__(A, B, C, D)
        if self.m != self.p:
            raise NotImplementedError
        # self.R = D @ D.T
        if self.m > 1:  # MIMO case
            if not np.all(np.linalg.eigvals(self.D) > 0):
                raise ValueError("D must be nonsingular")
            self.Dinv = np.linalg.solve(self.D, np.eye(self.m))
        else:  # SISO case
            if not (self.R > 0):
                raise ValueError("D must be nonsingular")
            self.Dinv = 1 / self.D

    @cached_property
    def Q(self):
        # Solve + cache ARE
        #   :math: A.T * Qbst + Qbst.T * A + (C - B_W.T * Qbst).T * Rinv * (C - B_W.T * Qbst) = 0
        return sp.linalg.solve_continuous_are(
            self.A,
            -1 * (self.B_W),
            np.zeros([self.n, self.n]),
            -1 * (self.D @ self.D.T),
            self.I,
            self.C.T,
        )

    @cached_property
    def P(self):
        #   :math: A * P + P.T * A + B *B.T = 0
        return sp.linalg.solve_continuous_lyapunov(self.A, -(self.B @ self.B.T))

    @cached_property
    def B_W(self):
        # Compute input matrix of left spectral factor (lsf) W(s) s.t.
        #   ..math: `G(s) @ G(-s).T = W(-s).T @ W(s)`
        # Used in computing observability Gramian (solution to ARE)
        return self.P @ self.C.T + self.B @ self.D

    @cached_property
    def C_lsf(self):
        # Output matrix of left spectral factor (lsf) s.t
        #   ..math: `G(s) @ G(-s).T = W(-s).T @ W(s)`
        return self.Dinv @ (self.C - (self.B_W.T @ self.Q))

    @cached_property
    def B_rsf(self):
        # Here, right spectral factor (rsf) is :math: `G(s)` itself, so return B
        # Called during computing approximate square-root factors
        return self.B

    def samples_for_Hbar(self, sl):
        # Artifcially sample the system cascade
        #   ..math: `H(s) : = [(W(-s).T)^{-1} * G(s)]_+ = C_lsf * (s * I - A) \ B`
        # _+ denotes the stable part of the transfer function
        # Returns self.samples_for_Lbar_Mbar(sl) for generality
        # Used in building Hbar = Lh_bst @ B from data
        # Called during `GeneralizedQuadBTReductor.Hbar`
        return self.samples_for_Lbar_Mbar(sl)

    def samples_for_Gbar(self, sr):
        # Artifcially sample the relevant rsf, here this is just :math: `G(s)` s.t.
        #   ..math: `G(s) @ G(-s).T = W(-s).T @ W(s)`
        # Used in building Gbar = C @ U_bst from data
        # Called during `GeneralizedQuadBTReductor.Gbar`
        return self.sampleG(sr)

    def samples_for_Lbar_Mbar(self, s):
        # Artifcially sample the system cascade
        #   ..math: `H(s) : = [(W(-s).T)^{-1} * G(s)]_+ = C_lsf * (s * I - A) \ B`
        # _+ denotes the stable part of the transfer function
        # Used in building the reduced Lbar = Lh_bst @ U_bst and Hbar = Lh_bst @ A @ U_bst from daya
        # Called during `GeneralizedQuadBTReductor.Lbar_Mbar`

        Hs = np.zeros([np.shape(s)[0], self.m, self.m], dtype="complex_")
        for j, sj in enumerate(s):
            Hs[j, :, :] = self.C_lsf @ np.linalg.solve((sj * self.I - self.A), self.B)

        return Hs


#     _               _   _   _ ___
#    / \      _.  _| |_) |_) |_) |
#    \_X |_| (_| (_| |_) | \ |_) |
#


class QuadBRBTSampler(GenericSampleGenerator):
    """Class to generate relevant transfer function samples for use in Quadrature-Based Bounded-real Balanced Truncation (QuadBRBT)

    Paramters
    ---------
    A, B, C, D
        State-space realization of the FOM, all matrices are |Numpy Arrays| of size (n, n), (n, m), (p, n), and (p, m) respectively
    """

    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D)
        self.R_B = np.eye(self.p) - self.D @ self.D.T
        self.R_C = np.eye(self.m) - self.D.T @ self.D
        if self.m > 1:  # MIMO case
            # If R_C is not invertible, neither is R_B
            if not np.all(np.linalg.eigvals(self.R_C) > 0):
                raise ValueError("System must be bounded real")
            self.R_Cinv = np.linalg.solve(self.R_C, np.eye(self.m))
            self.R_Cinvsqrt = np.linalg.cholesky(self.R_Cinv)
        else:  # SISO case
            if not (self.R_C > 0):
                raise ValueError("System must be bounded real")
            self.R_Cinv = 1 / self.R_C
            self.R_Cinvsqrt = np.sqrt(self.R_Cinv)
        if self.p > 1:  # MIMO case
            # If R_C is not invertible, neither is R_C
            if not np.all(np.linalg.eigvals(self.R_B) > 0):
                raise ValueError("System must be bounded real")
            self.R_Binv = np.linalg.solve(self.R_B, np.eye(self.p))
            self.R_Binvsqrt = np.linalg.cholesky(self.R_Binv)
        else:  # SISO case
            if not (self.R_B > 0):
                raise ValueError("System must be bounded real")
            self.R_Binv = 1 / self.R_B
            self.R_Binvsqrt = np.sqrt(self.R_Binv)

    @cached_property
    def Q(self):
        # Solve + cache ARE
        #   :math: A.T * Qbrbt + Qbrbt * A + C.T * C + (B.T * Qbrbt + D.T * C).T * R_Cinv * (B.T * Qbrbt + D.T * C) = 0
        return sp.linalg.solve_continuous_are(
            self.A, self.B, self.C.T @ self.C, -1 * (self.R_C), self.I, self.C.T @ self.D
        )

    @cached_property
    def P(self):
        # Solve + cache ARE
        #   :math: A * Pbrbt + Pbrbt * A.T + B * B.T + (Pbrbt * C.T + B * D.T) * R_Binv * (Pbrbt * C.T + B * D.T).T = 0
        return sp.linalg.solve_continuous_are(
            self.A.T, self.C.T, self.B @ self.B.T, -1 * (self.R_B), self.I, self.B @ self.D.T
        )

    @cached_property
    def C_lsf(self):
        # Output matrix of left spectral factor (lsf) such that
        #   ..math: `I_m - G(-s).T * G(s) = M(-s).T * M(s)`
        return -(self.R_Cinvsqrt @ ((self.B.T @ self.Q) + (self.D.T @ self.C)))

    @cached_property
    def B_rsf(self):
        # Output matrix of right spectral factor (rsf) such that
        #   ..math: `I_p - G(s) * G(-s).T = N(s) * N(-s).T`
        return -((self.P @ self.C.T) + (self.B @ self.D.T)) @ self.R_Binvsqrt

    def samples_for_Hbar(self, sl):
        # Artifcially sample the (strictly proper part) of the left spectral factor (lsf) such that
        #   ..math: `I_m - G(-s).T * G(s) = M(-s).T * M(s)`
        # :math: `M(s)` is an m x m rational transfer function...
        # ... and artificially sample G(s)
        # In QuadBRBT, these samples are used in building Hbar = Lh_brbt @ B from data
        # Called during `GeneralizedQuadBTReductor.Hbar`

        Ms = np.zeros([np.shape(sl)[0], self.m, self.m], dtype="complex_")
        for k, sl_k in enumerate(sl):
            Ms[k, :, :] = self.C_lsf @ np.linalg.solve((sl_k * self.I - self.A), self.B)

        # Returns samples of the spectral factor, as well as G(s)
        return Ms, self.sampleG(sl)

    def samples_for_Gbar(self, sr):
        # Artifcially sample the (strictly proper part) of the right spectral factor (rsf) such that
        #   ..math: `I_p - G(s) * G(-s).T = N(s) * N(-s).T``
        # :math: `N(s)` is a p x p rational transfer function...
        # ... and artificially sample G(s)
        # In QuadBRBT, these samples are used in building Gbar = C @ U_brbt from data
        # Called during `GeneralizedQuadBTReductor.Hbar`

        Ns = np.zeros([np.shape(sr)[0], self.m, self.m], dtype="complex_")
        for j, sr_j in enumerate(sr):
            Ns[j, :, :] = self.C @ np.linalg.solve((sr_j * self.I - self.A), self.B_rsf)

        # Returns samples of the spectral factor, as well as G(s)
        return Ns, self.sampleG(sr)

    def samples_for_Lbar_Mbar(self, s):
        # Artifcially sample the system cascade
        #   ..math: `H(s) : = [M(s) * (N(-s).T)^{-1}]_+ = C_lsf * (s * I - A) \ B_rsf`
        # _+ denotes the stable part of the transfer function...
        # ... also artificially sample G(s), M(s), and N(s0)
        # In QuadBRBT, these samples are used in building Lbar = Lh_brbt @ U_brbt and Mbar = Lh_brbt @ A @ U_brbt
        # Called during `GeneralizedQuadBTReductor.Lbar_Mbar`

        Hs = np.zeros([np.shape(s)[0], self.m, self.m], dtype="complex_")
        for j, sj in enumerate(s):
            Hs[j, :, :] = self.C_lsf @ np.linalg.solve((sj * self.I - self.A), self.B_rsf)

        # Returns samples of the cascade, G(s), M(s), N(s)
        Ns, _ = self.samples_for_Gbar(s)
        Ms, _ = self.samples_for_Hbar(s)
        return Hs, Ns, Ms, self.sampleG(s)

    def right_sqrt_fact_U(self, sr, weightsr):
        # TODO: Option to do different quadrature rules
        # Return approximate quadrature-based sqrt factors
        #   ..math: `P ~ U * Uh + U_N * U_Nh,  U, U_N \in \C^{n \times (Nr * m)}`
        # Used in checking error of the implicity= quadrature rule
        assert np.shape(sr)[0] == np.shape(weightsr)[0]
        U = np.zeros([np.shape(sr)[0], self.n, self.m], dtype="complex_")
        U_N = np.zeros([np.shape(sr)[0], self.n, self.m], dtype="complex_")
        for j, sr_j in enumerate(sr):
            U[j, :, :] = weightsr[j] * np.linalg.solve((sr_j * self.I - self.A), self.B)
            U_N[j, :, :] = weightsr[j] * np.linalg.solve((sr_j * self.I - self.A), self.B_rsf)
        # Actual quadrature factor is [U, U_N]
        return np.c_[np.concatenate(U, axis=1), np.concatenate(U_N, axis=1)]

    def left_sqrt_fact_Lh(self, sl, weightsl):
        # TODO: Option to do different quadrature rules
        # Return approximate quadrature-based sqrt factors, applied to both Gramians
        #   ..math: `Q ~ L * Lh + L_M * L_Mh,  Lh, L_Mh \in \C^{(p * Nl) \times n}`
        # Used in checking error of the implicity= quadrature rule
        assert np.shape(sl)[0] == np.shape(weightsl)[0]
        Lh = np.zeros([np.shape(sl)[0], self.p, self.n], dtype="complex_")
        L_Mh = np.zeros([np.shape(sl)[0], self.p, self.n], dtype="complex_")
        for k, sl_k in enumerate(sl):
            Lh[k, :, :] = weightsl[k] * (self.C @ np.linalg.solve((sl_k * self.I - self.A), self.I))
            L_Mh[k, :, :] = weightsl[k] * (
                self.C_lsf @ np.linalg.solve((sl_k * self.I - self.A), self.I)
            )
        # Actual quadrature factor is [Lh; L_Mh]
        return np.r_[np.concatenate(Lh, axis=0), np.concatenate(L_Mh, axis=0)]


def trapezoidal_rule(exp_limits=np.array((-3, 3)), N=100, ordering="interlace"):
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
        weightsl = 1 / np.sqrt(2 * np.pi) * np.sqrt(np.absolute(weightsl))
        weightsr = 1 / np.sqrt(2 * np.pi) * np.sqrt(np.absolute(weightsr))
        return modesl, modesr, weightsl, weightsr
    else:
        raise NotImplementedError
