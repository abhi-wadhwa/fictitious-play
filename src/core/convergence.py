"""Convergence diagnostics for Fictitious Play.

Provides tools to measure how close empirical strategies are to a Nash
equilibrium and to detect cycling behaviour (as in Shapley's 3x3 game).

Key diagnostics
---------------
* **L2 distance to NE** -- Euclidean distance from the current empirical
  profile to the nearest Nash equilibrium.
* **Exploitability** -- The maximum gain any player can obtain by deviating,
  a direct measure of approximate equilibrium quality.
* **Cycling detection** -- Autocorrelation analysis on empirical strategy
  time series to detect periodic orbits.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional


class ConvergenceDiagnostics:
    """Convergence analysis tools for two-player normal-form games.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Row player payoff matrix.
    B : array_like, shape (m, n)
        Column player payoff matrix.
    nash_equilibria : list of (NDArray, NDArray), optional
        Known Nash equilibria as ``(row_strategy, col_strategy)`` pairs.
        If provided, distance-to-NE computations use these directly.
    """

    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        nash_equilibria: Optional[List[Tuple[NDArray, NDArray]]] = None,
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.m, self.n = self.A.shape
        self.nash_equilibria = nash_equilibria or []

    # -----------------------------------------------------------------
    # Exploitability
    # -----------------------------------------------------------------

    def exploitability(
        self, row_strategy: NDArray, col_strategy: NDArray
    ) -> float:
        """Compute the exploitability of a strategy profile.

        Exploitability is defined as the sum of the maximum gains the
        two players could achieve by unilateral deviation:

            exploit(sigma) = max_i [e_i^T A sigma_c] - sigma_r^T A sigma_c
                           + max_j [sigma_r^T B e_j] - sigma_r^T B sigma_c

        A profile is a Nash equilibrium iff its exploitability is zero.

        Parameters
        ----------
        row_strategy : (m,) array
        col_strategy : (n,) array

        Returns
        -------
        float
            Non-negative exploitability value.
        """
        rs = np.asarray(row_strategy, dtype=np.float64)
        cs = np.asarray(col_strategy, dtype=np.float64)

        row_payoff = rs @ self.A @ cs
        col_payoff = rs @ self.B @ cs

        best_row_dev = np.max(self.A @ cs)
        best_col_dev = np.max(self.B.T @ rs)

        return float((best_row_dev - row_payoff) + (best_col_dev - col_payoff))

    # -----------------------------------------------------------------
    # Distance to NE
    # -----------------------------------------------------------------

    def distance_to_ne(
        self, row_strategy: NDArray, col_strategy: NDArray
    ) -> float:
        """L2 distance from a strategy profile to the nearest known NE.

        Parameters
        ----------
        row_strategy : (m,) array
        col_strategy : (n,) array

        Returns
        -------
        float
            Euclidean distance.  Returns ``inf`` if no NE are stored.
        """
        if not self.nash_equilibria:
            return float("inf")

        rs = np.asarray(row_strategy, dtype=np.float64)
        cs = np.asarray(col_strategy, dtype=np.float64)

        min_dist = float("inf")
        for ne_r, ne_c in self.nash_equilibria:
            dist = np.sqrt(
                np.sum((rs - ne_r) ** 2) + np.sum((cs - ne_c) ** 2)
            )
            min_dist = min(min_dist, dist)
        return min_dist

    def distance_trajectory(
        self,
        row_empiricals: List[NDArray],
        col_empiricals: List[NDArray],
    ) -> NDArray:
        """Compute the distance-to-NE at every time step.

        Parameters
        ----------
        row_empiricals : list of (m,) arrays
        col_empiricals : list of (n,) arrays

        Returns
        -------
        (T,) array
            Distance at each round.
        """
        T = len(row_empiricals)
        dists = np.empty(T)
        for t in range(T):
            dists[t] = self.distance_to_ne(row_empiricals[t], col_empiricals[t])
        return dists

    def exploitability_trajectory(
        self,
        row_empiricals: List[NDArray],
        col_empiricals: List[NDArray],
    ) -> NDArray:
        """Compute exploitability at every time step.

        Parameters
        ----------
        row_empiricals : list of (m,) arrays
        col_empiricals : list of (n,) arrays

        Returns
        -------
        (T,) array
            Exploitability at each round.
        """
        T = len(row_empiricals)
        exploits = np.empty(T)
        for t in range(T):
            exploits[t] = self.exploitability(
                row_empiricals[t], col_empiricals[t]
            )
        return exploits

    # -----------------------------------------------------------------
    # Cycling detection
    # -----------------------------------------------------------------

    @staticmethod
    def autocorrelation(signal: NDArray, max_lag: Optional[int] = None) -> NDArray:
        """Compute the normalized autocorrelation of a 1-D signal.

        Parameters
        ----------
        signal : (T,) array
        max_lag : int, optional
            Maximum lag to compute.  Defaults to ``T // 2``.

        Returns
        -------
        (max_lag + 1,) array
            Autocorrelation at lags 0, 1, ..., max_lag.
        """
        x = np.asarray(signal, dtype=np.float64)
        x = x - x.mean()
        T = len(x)
        if max_lag is None:
            max_lag = T // 2

        var = np.dot(x, x)
        if var < 1e-15:
            return np.zeros(max_lag + 1)

        ac = np.empty(max_lag + 1)
        for lag in range(max_lag + 1):
            ac[lag] = np.dot(x[: T - lag], x[lag:]) / var
        return ac

    def detect_cycling(
        self,
        row_empiricals: List[NDArray],
        col_empiricals: List[NDArray],
        threshold: float = 0.3,
        min_period: int = 5,
        max_lag: Optional[int] = None,
    ) -> Tuple[bool, Optional[int], Optional[float]]:
        """Detect cycling in the empirical strategy trajectory.

        Analyses the autocorrelation of the first principal component of
        the joint strategy trajectory.  A peak in the autocorrelation
        beyond ``min_period`` that exceeds ``threshold`` indicates
        periodic behaviour.

        Parameters
        ----------
        row_empiricals : list of (m,) arrays
        col_empiricals : list of (n,) arrays
        threshold : float
            Minimum autocorrelation peak height to declare cycling.
        min_period : int
            Minimum period (lag) to consider as a cycle.
        max_lag : int, optional
            Maximum autocorrelation lag.

        Returns
        -------
        (is_cycling, period, peak_autocorr)
            * ``is_cycling`` -- ``True`` if cycling is detected.
            * ``period`` -- Estimated cycle period in rounds (or ``None``).
            * ``peak_autocorr`` -- Height of the dominant autocorrelation
              peak (or ``None``).
        """
        T = len(row_empiricals)
        if T < 2 * min_period:
            return False, None, None

        # Build joint trajectory matrix  (T x (m+n))
        joint = np.column_stack(
            [np.array(row_empiricals), np.array(col_empiricals)]
        )

        # Use the first action's empirical frequency as the signal
        # (more robust: project onto the first principal component)
        centered = joint - joint.mean(axis=0)
        # SVD to get first PC
        try:
            U, S, _ = np.linalg.svd(centered, full_matrices=False)
            signal = U[:, 0] * S[0]
        except np.linalg.LinAlgError:
            signal = centered[:, 0]

        ac = self.autocorrelation(signal, max_lag=max_lag)

        # Find peaks in autocorrelation beyond min_period
        # A peak is a local maximum: ac[i] > ac[i-1] and ac[i] > ac[i+1]
        peaks_lag = []
        peaks_val = []
        for i in range(max(1, min_period), len(ac) - 1):
            if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] > threshold:
                peaks_lag.append(i)
                peaks_val.append(ac[i])

        if not peaks_lag:
            return False, None, None

        # Pick the first (shortest period) significant peak
        best_idx = int(np.argmax(peaks_val))
        period = peaks_lag[best_idx]
        peak_val = peaks_val[best_idx]

        return True, period, float(peak_val)

    # -----------------------------------------------------------------
    # Convergence rate estimation
    # -----------------------------------------------------------------

    @staticmethod
    def estimate_convergence_rate(
        distances: NDArray, window: int = 50
    ) -> Tuple[float, float]:
        """Estimate the convergence rate from a distance trajectory.

        Fits a line to ``log(distance)`` vs ``log(t)`` via least squares.
        For FP in zero-sum games, the theoretical rate is O(1/t), so
        the slope should be approximately -1.

        Parameters
        ----------
        distances : (T,) array
            Distance-to-NE at each round.
        window : int
            Use the last ``window`` entries for the fit.

        Returns
        -------
        (slope, intercept)
            Coefficients of the log-log linear fit.
        """
        dists = np.asarray(distances, dtype=np.float64)
        T = len(dists)
        start = max(0, T - window)
        idx = np.arange(start, T) + 1  # 1-indexed rounds
        d = dists[start:T]

        # Filter out zeros / near-zeros
        mask = d > 1e-15
        if mask.sum() < 2:
            return 0.0, 0.0

        log_t = np.log(idx[mask])
        log_d = np.log(d[mask])

        # Least squares
        A_mat = np.column_stack([log_t, np.ones_like(log_t)])
        result, _, _, _ = np.linalg.lstsq(A_mat, log_d, rcond=None)
        slope, intercept = result
        return float(slope), float(intercept)
