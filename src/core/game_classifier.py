"""Game classification utilities.

Detects structural properties of two-player normal-form games that
have implications for Fictitious Play convergence:

* **Zero-sum games** (A + B^T = 0): FP is guaranteed to converge
  to the set of minimax (Nash) equilibria (Robinson, 1951).
* **Potential games**: FP converges to a Nash equilibrium because
  a common ordinal potential function is increased at every step.
* **Generic / unclassified**: No structural guarantee; FP may or
  may not converge (cf. Shapley's 3x3 counterexample).

We also provide utilities for computing the minimax value of
zero-sum games and checking for dominant strategies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Tuple, List


class GameType(Enum):
    """Classification of a two-player game."""

    ZERO_SUM = auto()
    POTENTIAL = auto()
    SYMMETRIC = auto()
    GENERIC = auto()


@dataclass
class ClassificationResult:
    """Result of game classification.

    Attributes
    ----------
    game_type : GameType
        Primary classification.
    is_zero_sum : bool
    is_potential : bool
    is_symmetric : bool
    details : str
        Human-readable description.
    potential_matrix : NDArray or None
        The potential matrix P, if the game is a potential game.
    """

    game_type: GameType
    is_zero_sum: bool
    is_potential: bool
    is_symmetric: bool
    details: str
    potential_matrix: Optional[NDArray] = None


class GameClassifier:
    """Classify a two-player normal-form game.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Row player payoff matrix.
    B : array_like, shape (m, n)
        Column player payoff matrix.
    tol : float
        Numerical tolerance for classification checks.
    """

    def __init__(
        self, A: NDArray, B: NDArray, tol: float = 1e-10
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        if self.A.shape != self.B.shape:
            raise ValueError("Payoff matrices must have the same shape.")
        self.m, self.n = self.A.shape
        self.tol = tol

    # -----------------------------------------------------------------
    # Zero-sum check
    # -----------------------------------------------------------------

    def is_zero_sum(self) -> bool:
        """Check whether A + B = 0 (constant-sum is also detected).

        A game is zero-sum if A[i,j] + B[i,j] = c for all i, j and
        some constant c.  Pure zero-sum has c = 0.

        Returns
        -------
        bool
        """
        S = self.A + self.B
        return bool(np.allclose(S, S[0, 0], atol=self.tol))

    # -----------------------------------------------------------------
    # Potential game check
    # -----------------------------------------------------------------

    def is_potential_game(self) -> Tuple[bool, Optional[NDArray]]:
        """Check whether the game is an exact potential game.

        A game (A, B) is an exact potential game if there exists a
        matrix P such that for all i, i', j:
            A[i, j] - A[i', j] = P[i, j] - P[i', j]
        and for all i, j, j':
            B[i, j] - B[i, j'] = P[i, j] - P[i, j']

        The potential matrix P is constructed (if it exists) by
        verifying the discrete analog of the curl-free condition on
        the payoff differences.

        Returns
        -------
        (is_potential, P)
            ``P`` is the potential matrix if the game is a potential
            game, otherwise ``None``.
        """
        m, n = self.m, self.n

        # Build P from A's row differences and B's column differences.
        # Set P[0, 0] = 0 and reconstruct:
        #   P[i, 0] = P[0, 0] + sum_{k=0}^{i-1} (A[k+1, 0] - A[k, 0])
        #   P[i, j] = P[i, 0] + sum_{l=0}^{j-1} (B[i, l+1] - B[i, l])
        P = np.zeros((m, n), dtype=np.float64)

        # Fill first column using A
        for i in range(1, m):
            P[i, 0] = P[i - 1, 0] + (self.A[i, 0] - self.A[i - 1, 0])

        # Fill each row using B
        for i in range(m):
            for j in range(1, n):
                P[i, j] = P[i, j - 1] + (self.B[i, j] - self.B[i, j - 1])

        # Verify: check A differences match P differences along rows
        for j in range(n):
            for i in range(1, m):
                a_diff = self.A[i, j] - self.A[i - 1, j]
                p_diff = P[i, j] - P[i - 1, j]
                if abs(a_diff - p_diff) > self.tol:
                    return False, None

        # Verify: check B differences match P differences along columns
        for i in range(m):
            for j in range(1, n):
                b_diff = self.B[i, j] - self.B[i, j - 1]
                p_diff = P[i, j] - P[i, j - 1]
                if abs(b_diff - p_diff) > self.tol:
                    return False, None

        return True, P

    # -----------------------------------------------------------------
    # Symmetric check
    # -----------------------------------------------------------------

    def is_symmetric(self) -> bool:
        """Check whether the game is symmetric (A = B^T and m = n).

        Returns
        -------
        bool
        """
        if self.m != self.n:
            return False
        return bool(np.allclose(self.A, self.B.T, atol=self.tol))

    # -----------------------------------------------------------------
    # Dominant strategy detection
    # -----------------------------------------------------------------

    def dominant_strategies(self) -> Tuple[Optional[int], Optional[int]]:
        """Find strictly dominant strategies, if any.

        Returns
        -------
        (row_dominant, col_dominant)
            Index of the dominant strategy for each player, or ``None``
            if no strictly dominant strategy exists.
        """
        row_dom = None
        for i in range(self.m):
            if all(
                np.all(self.A[i, :] > self.A[k, :])
                for k in range(self.m)
                if k != i
            ):
                row_dom = i
                break

        col_dom = None
        for j in range(self.n):
            if all(
                np.all(self.B[:, j] > self.B[:, l])
                for l in range(self.n)
                if l != j
            ):
                col_dom = j
                break

        return row_dom, col_dom

    # -----------------------------------------------------------------
    # Pure Nash equilibria
    # -----------------------------------------------------------------

    def pure_nash_equilibria(self) -> List[Tuple[int, int]]:
        """Find all pure-strategy Nash equilibria.

        Returns
        -------
        list of (int, int)
            List of action profiles ``(i, j)`` that are NE in pure
            strategies.
        """
        ne_list = []
        for i in range(self.m):
            for j in range(self.n):
                # Check row player: i is best response to j
                if self.A[i, j] < np.max(self.A[:, j]) - self.tol:
                    continue
                # Check column player: j is best response to i
                if self.B[i, j] < np.max(self.B[i, :]) - self.tol:
                    continue
                ne_list.append((i, j))
        return ne_list

    # -----------------------------------------------------------------
    # Minimax value (zero-sum games)
    # -----------------------------------------------------------------

    def minimax_value(self) -> Optional[float]:
        """Compute the minimax value of a zero-sum game via LP.

        Uses a simple iterative method (linear programming would be
        more robust but we avoid extra dependencies).  For small
        matrices, we use the exact formula for 2x2 games or a
        brute-force approach.

        Returns
        -------
        float or None
            The value of the game, or ``None`` if not zero-sum.
        """
        if not self.is_zero_sum():
            return None

        # Use the maximin approach:
        # v = max_{p in Delta_m} min_j  p^T A e_j
        #   = max_p min_j sum_i p_i A_{ij}

        # For small games, enumerate vertices.  For 2x2, solve exactly.
        if self.m == 2 and self.n == 2:
            return self._minimax_2x2()

        # General case: approximate via fictitious play itself
        # (we import here to avoid circular dependency)
        from src.core.fictitious_play import FictitiousPlay

        fp = FictitiousPlay(self.A, self.B)
        result = fp.run(5000, seed=0)
        row_strat = result.row_empirical[-1]
        col_strat = result.col_empirical[-1]
        return float(row_strat @ self.A @ col_strat)

    def _minimax_2x2(self) -> float:
        """Exact minimax for 2x2 zero-sum games."""
        A = self.A
        # Check for saddle point
        row_min = A.min(axis=1)
        col_max = A.max(axis=0)
        maximin = row_min.max()
        minimax = col_max.min()
        if np.isclose(maximin, minimax, atol=self.tol):
            return float(maximin)

        # Mixed strategy solution
        # p* = (A[1,1] - A[1,0]) / (A[0,0] - A[0,1] - A[1,0] + A[1,1])
        denom = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
        if abs(denom) < self.tol:
            return float(maximin)
        p = (A[1, 1] - A[1, 0]) / denom
        p = np.clip(p, 0, 1)
        value = p * A[0, 0] + (1 - p) * A[1, 0]
        # But also need column mix
        q = (A[1, 1] - A[0, 1]) / denom
        q = np.clip(q, 0, 1)
        value = p * (q * A[0, 0] + (1 - q) * A[0, 1]) + (1 - p) * (
            q * A[1, 0] + (1 - q) * A[1, 1]
        )
        return float(value)

    # -----------------------------------------------------------------
    # Full classification
    # -----------------------------------------------------------------

    def classify(self) -> ClassificationResult:
        """Run all classification checks and return a summary.

        Returns
        -------
        ClassificationResult
        """
        zs = self.is_zero_sum()
        pg, P = self.is_potential_game()
        sym = self.is_symmetric()

        if zs:
            gtype = GameType.ZERO_SUM
            details = (
                "Zero-sum (or constant-sum) game.  Fictitious play is "
                "guaranteed to converge to the set of minimax equilibria "
                "(Robinson, 1951)."
            )
        elif pg:
            gtype = GameType.POTENTIAL
            details = (
                "Exact potential game.  Fictitious play converges to a "
                "Nash equilibrium because the potential function increases "
                "monotonically along the FP trajectory (Monderer & Shapley, "
                "1996)."
            )
        elif sym:
            gtype = GameType.SYMMETRIC
            details = (
                "Symmetric game (A = B^T).  FP convergence depends on the "
                "specific payoff structure."
            )
        else:
            gtype = GameType.GENERIC
            details = (
                "Generic bimatrix game.  No structural convergence guarantee "
                "for fictitious play.  Cycling is possible (Shapley, 1964)."
            )

        return ClassificationResult(
            game_type=gtype,
            is_zero_sum=zs,
            is_potential=pg,
            is_symmetric=sym,
            details=details,
            potential_matrix=P,
        )


# -----------------------------------------------------------------
# Predefined game zoo
# -----------------------------------------------------------------

def game_zoo() -> dict:
    """Return a dictionary of classic games for demonstration.

    Returns
    -------
    dict
        Keys are game names, values are dicts with keys ``A``, ``B``,
        ``description``, and optionally ``nash_equilibria``.
    """
    ne_third = np.array([1 / 3, 1 / 3, 1 / 3])

    games = {
        "Rock-Paper-Scissors": {
            "A": np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64),
            "B": np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=np.float64),
            "description": (
                "Classic 3x3 zero-sum game.  FP converges to the unique "
                "Nash equilibrium (1/3, 1/3, 1/3)."
            ),
            "nash_equilibria": [(ne_third.copy(), ne_third.copy())],
        },
        "Matching Pennies": {
            "A": np.array([[1, -1], [-1, 1]], dtype=np.float64),
            "B": np.array([[-1, 1], [1, -1]], dtype=np.float64),
            "description": (
                "2x2 zero-sum game.  FP converges to (1/2, 1/2) for both "
                "players."
            ),
            "nash_equilibria": [
                (np.array([0.5, 0.5]), np.array([0.5, 0.5]))
            ],
        },
        "Prisoner's Dilemma": {
            "A": np.array([[3, 0], [5, 1]], dtype=np.float64),
            "B": np.array([[3, 5], [0, 1]], dtype=np.float64),
            "description": (
                "Classic 2x2 game with a dominant strategy equilibrium "
                "(Defect, Defect).  FP converges immediately."
            ),
            "nash_equilibria": [
                (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
            ],
        },
        "Battle of the Sexes": {
            "A": np.array([[3, 0], [0, 2]], dtype=np.float64),
            "B": np.array([[2, 0], [0, 3]], dtype=np.float64),
            "description": (
                "Coordination game with two pure NE: (Opera, Opera) and "
                "(Football, Football), plus a mixed NE."
            ),
            "nash_equilibria": [
                (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
                (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                (np.array([0.6, 0.4]), np.array([0.4, 0.6])),
            ],
        },
        "Shapley's Game (3x3 cycling)": {
            "A": np.array(
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64
            ),
            "B": np.array(
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64
            ),
            "description": (
                "Shapley's (1964) 3x3 game where classical fictitious play "
                "does NOT converge.  Empirical strategies cycle on the "
                "boundary of the simplex and never approach the unique NE "
                "at (1/3, 1/3, 1/3)."
            ),
            "nash_equilibria": [(ne_third.copy(), ne_third.copy())],
        },
        "Coordination Game": {
            "A": np.array([[2, 0], [0, 1]], dtype=np.float64),
            "B": np.array([[2, 0], [0, 1]], dtype=np.float64),
            "description": (
                "Symmetric coordination game.  Also an exact potential game.  "
                "FP converges to a pure NE."
            ),
            "nash_equilibria": [
                (np.array([1.0, 0.0]), np.array([1.0, 0.0])),
                (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
            ],
        },
    }
    return games
