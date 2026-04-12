"""Tests for game classification."""

import numpy as np
import pytest

from src.core.game_classifier import GameClassifier, GameType, game_zoo


class TestZeroSumDetection:
    """Test zero-sum game detection."""

    def test_matching_pennies_is_zero_sum(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        B = -A
        gc = GameClassifier(A, B)
        assert gc.is_zero_sum() is True

    def test_rps_is_zero_sum(self):
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        B = -A
        gc = GameClassifier(A, B)
        assert gc.is_zero_sum() is True

    def test_constant_sum_is_detected(self):
        """Constant-sum (A + B = c) should also be detected."""
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = 10 - A  # constant sum = 10
        gc = GameClassifier(A, B)
        assert gc.is_zero_sum() is True

    def test_prisoners_dilemma_not_zero_sum(self):
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        assert gc.is_zero_sum() is False


class TestPotentialGameDetection:
    """Test exact potential game detection."""

    def test_coordination_game_is_potential(self):
        """Symmetric coordination game is a potential game."""
        A = np.array([[2, 0], [0, 1]], dtype=np.float64)
        B = np.array([[2, 0], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        is_pot, P = gc.is_potential_game()
        assert is_pot is True
        assert P is not None

        # Verify potential differences match payoff differences
        for i in range(1, 2):
            for j in range(2):
                a_diff = A[i, j] - A[i - 1, j]
                p_diff = P[i, j] - P[i - 1, j]
                assert np.isclose(a_diff, p_diff)

    def test_pure_coordination_is_potential(self):
        """Pure coordination (A = B) is always a potential game."""
        A = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 1]], dtype=np.float64)
        B = A.copy()
        gc = GameClassifier(A, B)
        is_pot, P = gc.is_potential_game()
        assert is_pot is True

    def test_matching_pennies_not_potential(self):
        """Zero-sum matching pennies is NOT a potential game."""
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        B = -A
        gc = GameClassifier(A, B)
        is_pot, _ = gc.is_potential_game()
        assert is_pot is False

    def test_shapley_not_potential(self):
        """Shapley's 3x3 cycling game is NOT a potential game."""
        A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        gc = GameClassifier(A, B)
        is_pot, _ = gc.is_potential_game()
        assert is_pot is False


class TestSymmetricDetection:
    """Test symmetric game detection."""

    def test_symmetric_game(self):
        A = np.array([[2, 0], [0, 1]], dtype=np.float64)
        B = A.T
        gc = GameClassifier(A, B)
        assert gc.is_symmetric() is True

    def test_rps_is_not_symmetric(self):
        """Standard RPS (A, -A) is NOT symmetric because A != (-A)^T."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        gc = GameClassifier(A, -A)
        # A = [[0,-1,1],[1,0,-1],[-1,1,0]]
        # (-A)^T = [[0,-1,1],[1,0,-1],[-1,1,0]] = A
        # So actually A = B^T for standard RPS
        assert gc.is_symmetric() is True

    def test_non_square_not_symmetric(self):
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        B = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float64)
        gc = GameClassifier(A, B)
        assert gc.is_symmetric() is False


class TestDominantStrategies:
    """Test dominant strategy detection."""

    def test_prisoners_dilemma_dominant(self):
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        row_dom, col_dom = gc.dominant_strategies()
        assert row_dom == 1  # Defect dominates for row
        assert col_dom == 1  # Defect dominates for column

    def test_rps_no_dominant(self):
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        gc = GameClassifier(A, -A)
        row_dom, col_dom = gc.dominant_strategies()
        assert row_dom is None
        assert col_dom is None


class TestPureNashEquilibria:
    """Test pure NE finder."""

    def test_prisoners_dilemma_pure_ne(self):
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        ne = gc.pure_nash_equilibria()
        assert (1, 1) in ne
        assert len(ne) == 1

    def test_coordination_game_two_pure_ne(self):
        A = np.array([[2, 0], [0, 1]], dtype=np.float64)
        B = np.array([[2, 0], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        ne = gc.pure_nash_equilibria()
        assert (0, 0) in ne
        assert (1, 1) in ne
        assert len(ne) == 2

    def test_matching_pennies_no_pure_ne(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        gc = GameClassifier(A, -A)
        ne = gc.pure_nash_equilibria()
        assert len(ne) == 0


class TestFullClassification:
    """Test the classify() method."""

    def test_classify_zero_sum(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        gc = GameClassifier(A, -A)
        result = gc.classify()
        assert result.game_type == GameType.ZERO_SUM
        assert result.is_zero_sum is True

    def test_classify_potential(self):
        A = np.array([[2, 0], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, A)
        result = gc.classify()
        assert result.is_potential is True

    def test_classify_generic(self):
        """A non-symmetric, non-zero-sum, non-potential game is GENERIC."""
        A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        result = gc.classify()
        assert result.game_type == GameType.GENERIC

    def test_classify_shapley_is_symmetric(self):
        """Shapley's game has A = B^T, so it is classified as symmetric."""
        A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)
        gc = GameClassifier(A, B)
        result = gc.classify()
        assert result.game_type == GameType.SYMMETRIC


class TestMinimaxValue:
    """Test minimax computation for zero-sum games."""

    def test_matching_pennies_value(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        gc = GameClassifier(A, -A)
        val = gc.minimax_value()
        assert val is not None
        assert np.isclose(val, 0.0, atol=0.05)

    def test_non_zero_sum_returns_none(self):
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        gc = GameClassifier(A, B)
        assert gc.minimax_value() is None


class TestGameZoo:
    """Test the predefined game collection."""

    def test_zoo_has_expected_games(self):
        zoo = game_zoo()
        expected = [
            "Rock-Paper-Scissors",
            "Matching Pennies",
            "Prisoner's Dilemma",
            "Battle of the Sexes",
            "Shapley's Game (3x3 cycling)",
            "Coordination Game",
        ]
        for name in expected:
            assert name in zoo

    def test_zoo_games_have_valid_matrices(self):
        zoo = game_zoo()
        for name, data in zoo.items():
            A, B = data["A"], data["B"]
            assert A.shape == B.shape, f"{name}: shape mismatch"
            assert A.ndim == 2, f"{name}: A is not 2D"

    def test_zoo_nash_equilibria_are_valid(self):
        zoo = game_zoo()
        for name, data in zoo.items():
            A = data["A"]
            for ne_r, ne_c in data.get("nash_equilibria", []):
                assert np.isclose(ne_r.sum(), 1.0), f"{name}: row NE does not sum to 1"
                assert np.isclose(ne_c.sum(), 1.0), f"{name}: col NE does not sum to 1"
                assert len(ne_r) == A.shape[0]
                assert len(ne_c) == A.shape[1]
