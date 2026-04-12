"""Classical Fictitious Play implementation.

In classical fictitious play each player maintains an empirical frequency
distribution over the opponent's past actions and plays a best response to
that distribution at every round.

References
----------
Brown, G. W. (1951). Iterative solution of games by fictitious play.
    Activity Analysis of Production and Allocation, 13(1), 374-376.
Robinson, J. (1951). An iterative method of solving a game.
    Annals of Mathematics, 54(2), 296-301.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class FPResult:
    """Container for a full fictitious play run.

    Attributes
    ----------
    row_actions : list[int]
        Pure action chosen by the row player at each round.
    col_actions : list[int]
        Pure action chosen by the column player at each round.
    row_empirical : list[NDArray]
        Empirical strategy (frequency) of the row player after each round.
    col_empirical : list[NDArray]
        Empirical strategy (frequency) of the column player after each round.
    row_payoffs : list[float]
        Stage payoff received by the row player at each round.
    col_payoffs : list[float]
        Stage payoff received by the column player at each round.
    """

    row_actions: List[int] = field(default_factory=list)
    col_actions: List[int] = field(default_factory=list)
    row_empirical: List[NDArray] = field(default_factory=list)
    col_empirical: List[NDArray] = field(default_factory=list)
    row_payoffs: List[float] = field(default_factory=list)
    col_payoffs: List[float] = field(default_factory=list)


class FictitiousPlay:
    """Classical Fictitious Play for two-player normal-form games.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Payoff matrix for the row player.
    B : array_like, shape (m, n)
        Payoff matrix for the column player.
        For zero-sum games set ``B = -A``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])  # RPS
    >>> fp = FictitiousPlay(A, -A)
    >>> result = fp.run(1000)
    >>> result.row_empirical[-1]  # should approach (1/3, 1/3, 1/3)
    """

    def __init__(self, A: NDArray, B: NDArray) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.A.shape != self.B.shape:
            raise ValueError("Payoff matrices must have the same shape.")
        if self.A.ndim != 2:
            raise ValueError("Payoff matrices must be 2-dimensional.")
        self.m, self.n = self.A.shape  # m = row actions, n = col actions

    # -----------------------------------------------------------------
    # Core helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _best_response(payoff_matrix: NDArray, opponent_freq: NDArray) -> int:
        """Return a pure best response (argmax) to the opponent's empirical
        frequency.  Ties are broken uniformly at random.

        Parameters
        ----------
        payoff_matrix : (k, l) array
            The player's payoff matrix.
        opponent_freq : (l,) array
            Empirical frequency of the opponent.

        Returns
        -------
        int
            Index of the best-response action.
        """
        expected = payoff_matrix @ opponent_freq
        max_val = expected.max()
        best_actions = np.flatnonzero(np.isclose(expected, max_val))
        return int(np.random.choice(best_actions))

    @staticmethod
    def _update_counts(
        counts: NDArray, action: int, t: int
    ) -> NDArray:
        """Increment the count vector and return the updated empirical
        frequency.

        Parameters
        ----------
        counts : (k,) array
            Running action counts.
        action : int
            Action index played this round.
        t : int
            Current round number (1-indexed).

        Returns
        -------
        NDArray
            Normalized frequency vector.
        """
        counts[action] += 1
        return counts / t

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------

    def run(
        self,
        num_iterations: int = 1000,
        seed: Optional[int] = None,
        row_init: Optional[NDArray] = None,
        col_init: Optional[NDArray] = None,
    ) -> FPResult:
        """Execute classical fictitious play.

        Parameters
        ----------
        num_iterations : int
            Number of rounds to simulate.
        seed : int, optional
            Random seed for reproducibility.
        row_init : array_like, optional
            Initial empirical frequency for the row player.  Defaults to
            uniform.
        col_init : array_like, optional
            Initial empirical frequency for the column player.  Defaults to
            uniform.

        Returns
        -------
        FPResult
            Recorded history of the run.
        """
        if seed is not None:
            np.random.seed(seed)

        result = FPResult()

        # Initialize counts
        row_counts = np.zeros(self.m, dtype=np.float64)
        col_counts = np.zeros(self.n, dtype=np.float64)

        # Use initial frequencies for the first best-response computation
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
            # Best responses
            row_action = self._best_response(self.A, col_freq)
            col_action = self._best_response(self.B.T, row_freq)

            # Record actions
            result.row_actions.append(row_action)
            result.col_actions.append(col_action)

            # Update counts and frequencies
            row_freq = self._update_counts(row_counts, row_action, t)
            col_freq = self._update_counts(col_counts, col_action, t)

            result.row_empirical.append(row_freq.copy())
            result.col_empirical.append(col_freq.copy())

            # Stage payoffs
            result.row_payoffs.append(float(self.A[row_action, col_action]))
            result.col_payoffs.append(float(self.B[row_action, col_action]))

        return result

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def expected_payoffs(
        self, row_strategy: NDArray, col_strategy: NDArray
    ) -> Tuple[float, float]:
        """Compute expected payoffs for mixed strategy profiles.

        Parameters
        ----------
        row_strategy : (m,) array
        col_strategy : (n,) array

        Returns
        -------
        (float, float)
            Expected payoffs for the row and column player.
        """
        row_pay = float(row_strategy @ self.A @ col_strategy)
        col_pay = float(row_strategy @ self.B @ col_strategy)
        return row_pay, col_pay
