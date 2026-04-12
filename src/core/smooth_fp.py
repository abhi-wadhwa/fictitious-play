"""Smooth (logit / softmax) Fictitious Play.

Instead of a hard argmax best response, each player selects actions
according to a softmax (logistic) distribution over expected payoffs.
When the temperature parameter ``tau`` is high the response is nearly
uniform; as ``tau -> 0`` it recovers the classical (hard) best response.

Fudenberg, D. & Levine, D. K. (1998). The Theory of Learning in Games.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Optional, Union, Callable


@dataclass
class SmoothFPResult:
    """Container for a smooth fictitious play run.

    Attributes
    ----------
    row_actions : list[int]
        Pure action sampled by the row player each round.
    col_actions : list[int]
        Pure action sampled by the column player each round.
    row_empirical : list[NDArray]
        Empirical strategy of the row player after each round.
    col_empirical : list[NDArray]
        Empirical strategy of the column player after each round.
    row_smooth_br : list[NDArray]
        Smooth best-response distribution of the row player each round.
    col_smooth_br : list[NDArray]
        Smooth best-response distribution of the column player each round.
    temperatures : list[float]
        Temperature used at each round.
    """

    row_actions: List[int] = field(default_factory=list)
    col_actions: List[int] = field(default_factory=list)
    row_empirical: List[NDArray] = field(default_factory=list)
    col_empirical: List[NDArray] = field(default_factory=list)
    row_smooth_br: List[NDArray] = field(default_factory=list)
    col_smooth_br: List[NDArray] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)


class SmoothFictitiousPlay:
    """Smooth (logit) Fictitious Play for two-player normal-form games.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Payoff matrix for the row player.
    B : array_like, shape (m, n)
        Payoff matrix for the column player.
    temperature : float or callable
        Softmax temperature.  If a callable, it should map the round
        number ``t`` (1-indexed) to a positive float, enabling cooling
        schedules such as ``lambda t: 1.0 / t``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[3, 0], [5, 1]])
    >>> B = np.array([[3, 5], [0, 1]])
    >>> sfp = SmoothFictitiousPlay(A, B, temperature=0.5)
    >>> res = sfp.run(500, seed=42)
    """

    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        temperature: Union[float, Callable[[int], float]] = 1.0,
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.A.shape != self.B.shape:
            raise ValueError("Payoff matrices must have the same shape.")
        if self.A.ndim != 2:
            raise ValueError("Payoff matrices must be 2-dimensional.")
        self.m, self.n = self.A.shape

        if callable(temperature):
            self._tau_fn = temperature
        else:
            if temperature <= 0:
                raise ValueError("Temperature must be positive.")
            self._tau_fn = lambda _t: float(temperature)

    # -----------------------------------------------------------------
    # Core helpers
    # -----------------------------------------------------------------

    @staticmethod
    def softmax(logits: NDArray, temperature: float) -> NDArray:
        """Numerically stable softmax with temperature scaling.

        Parameters
        ----------
        logits : (k,) array
            Raw expected payoffs.
        temperature : float
            Softmax temperature (> 0).

        Returns
        -------
        (k,) array
            Probability distribution over actions.
        """
        scaled = logits / temperature
        shifted = scaled - scaled.max()
        exp_vals = np.exp(shifted)
        return exp_vals / exp_vals.sum()

    def _smooth_br(
        self, payoff_matrix: NDArray, opponent_freq: NDArray, tau: float
    ) -> NDArray:
        """Compute smooth best-response distribution.

        Parameters
        ----------
        payoff_matrix : (k, l) array
        opponent_freq : (l,) array
        tau : float

        Returns
        -------
        (k,) array
            Smooth best-response probabilities.
        """
        expected = payoff_matrix @ opponent_freq
        return self.softmax(expected, tau)

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(
        self,
        num_iterations: int = 1000,
        seed: Optional[int] = None,
        row_init: Optional[NDArray] = None,
        col_init: Optional[NDArray] = None,
    ) -> SmoothFPResult:
        """Execute smooth fictitious play.

        Parameters
        ----------
        num_iterations : int
            Number of rounds.
        seed : int, optional
            Random seed.
        row_init, col_init : array_like, optional
            Initial empirical frequencies (default: uniform).

        Returns
        -------
        SmoothFPResult
        """
        if seed is not None:
            np.random.seed(seed)

        result = SmoothFPResult()

        row_counts = np.zeros(self.m, dtype=np.float64)
        col_counts = np.zeros(self.n, dtype=np.float64)

        if row_init is not None:
            row_freq = np.asarray(row_init, dtype=np.float64)
            row_freq /= row_freq.sum()
        else:
            row_freq = np.ones(self.m) / self.m

        if col_init is not None:
            col_freq = np.asarray(col_init, dtype=np.float64)
            col_freq /= col_freq.sum()
        else:
            col_freq = np.ones(self.n) / self.n

        for t in range(1, num_iterations + 1):
            tau = self._tau_fn(t)
            result.temperatures.append(tau)

            # Smooth best responses
            row_br = self._smooth_br(self.A, col_freq, tau)
            col_br = self._smooth_br(self.B.T, row_freq, tau)

            result.row_smooth_br.append(row_br.copy())
            result.col_smooth_br.append(col_br.copy())

            # Sample actions from the smooth BR distributions
            row_action = int(np.random.choice(self.m, p=row_br))
            col_action = int(np.random.choice(self.n, p=col_br))

            result.row_actions.append(row_action)
            result.col_actions.append(col_action)

            # Update empirical frequencies
            row_counts[row_action] += 1
            col_counts[col_action] += 1
            row_freq = row_counts / t
            col_freq = col_counts / t

            result.row_empirical.append(row_freq.copy())
            result.col_empirical.append(col_freq.copy())

        return result

    # -----------------------------------------------------------------
    # Convenience: deterministic smooth FP (no sampling)
    # -----------------------------------------------------------------

    def run_deterministic(
        self,
        num_iterations: int = 1000,
        row_init: Optional[NDArray] = None,
        col_init: Optional[NDArray] = None,
    ) -> SmoothFPResult:
        """Deterministic smooth FP variant.

        Instead of sampling from the smooth BR, the empirical frequency
        is updated directly with the smooth BR distribution (continuous
        averaging).  This removes stochastic noise and is useful for
        visualizing the deterministic trajectory on the simplex.

        Parameters
        ----------
        num_iterations : int
        row_init, col_init : array_like, optional

        Returns
        -------
        SmoothFPResult
        """
        result = SmoothFPResult()

        if row_init is not None:
            row_freq = np.asarray(row_init, dtype=np.float64)
            row_freq /= row_freq.sum()
        else:
            row_freq = np.ones(self.m) / self.m

        if col_init is not None:
            col_freq = np.asarray(col_init, dtype=np.float64)
            col_freq /= col_freq.sum()
        else:
            col_freq = np.ones(self.n) / self.n

        for t in range(1, num_iterations + 1):
            tau = self._tau_fn(t)
            result.temperatures.append(tau)

            row_br = self._smooth_br(self.A, col_freq, tau)
            col_br = self._smooth_br(self.B.T, row_freq, tau)

            result.row_smooth_br.append(row_br.copy())
            result.col_smooth_br.append(col_br.copy())

            # Deterministic update: weighted average
            alpha = 1.0 / (t + 1)
            row_freq = (1 - alpha) * row_freq + alpha * row_br
            col_freq = (1 - alpha) * col_freq + alpha * col_br

            # Record dominant action for compatibility
            result.row_actions.append(int(np.argmax(row_br)))
            result.col_actions.append(int(np.argmax(col_br)))
            result.row_empirical.append(row_freq.copy())
            result.col_empirical.append(col_freq.copy())

        return result
