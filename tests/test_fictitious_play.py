"""Tests for classical Fictitious Play."""

import numpy as np
import pytest

from src.core.fictitious_play import FictitiousPlay, FPResult


class TestFictitiousPlayBasic:
    """Basic sanity checks."""

    def test_result_lengths(self):
        """FP result lists have correct length."""
        A = np.array([[1, 0], [0, 1]], dtype=np.float64)
        B = np.array([[1, 0], [0, 1]], dtype=np.float64)
        fp = FictitiousPlay(A, B)
        result = fp.run(100, seed=0)

        assert len(result.row_actions) == 100
        assert len(result.col_actions) == 100
        assert len(result.row_empirical) == 100
        assert len(result.col_empirical) == 100
        assert len(result.row_payoffs) == 100
        assert len(result.col_payoffs) == 100

    def test_empirical_sums_to_one(self):
        """Empirical frequencies must sum to 1 at every step."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        fp = FictitiousPlay(A, -A)
        result = fp.run(200, seed=1)

        for t in range(200):
            assert np.isclose(result.row_empirical[t].sum(), 1.0)
            assert np.isclose(result.col_empirical[t].sum(), 1.0)

    def test_actions_in_range(self):
        """Actions are valid indices."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        fp = FictitiousPlay(A, -A)
        result = fp.run(100, seed=2)

        for a in result.row_actions:
            assert 0 <= a < 3
        for a in result.col_actions:
            assert 0 <= a < 3

    def test_shape_mismatch_raises(self):
        """Mismatched payoff matrices should raise."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            FictitiousPlay(A, B)

    def test_1d_raises(self):
        """1-D payoff should raise."""
        with pytest.raises(ValueError):
            FictitiousPlay(np.array([1, 2]), np.array([3, 4]))


class TestZeroSumConvergence:
    """FP must converge to minimax in zero-sum games."""

    def test_matching_pennies_converges(self):
        """Matching Pennies: NE = (1/2, 1/2)."""
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        fp = FictitiousPlay(A, -A)
        result = fp.run(5000, seed=10)

        row_final = result.row_empirical[-1]
        col_final = result.col_empirical[-1]
        np.testing.assert_allclose(row_final, [0.5, 0.5], atol=0.05)
        np.testing.assert_allclose(col_final, [0.5, 0.5], atol=0.05)

    def test_rps_converges(self):
        """Rock-Paper-Scissors: NE = (1/3, 1/3, 1/3)."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        fp = FictitiousPlay(A, -A)
        result = fp.run(5000, seed=20)

        row_final = result.row_empirical[-1]
        col_final = result.col_empirical[-1]
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(row_final, expected, atol=0.05)
        np.testing.assert_allclose(col_final, expected, atol=0.05)


class TestDominantStrategy:
    """Games with dominant strategies should converge immediately."""

    def test_prisoners_dilemma(self):
        """PD: dominant strategy (Defect, Defect) = action (1, 1)."""
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        fp = FictitiousPlay(A, B)
        result = fp.run(100, seed=0)

        # After enough rounds, empirical should be concentrated on (1, 1)
        row_final = result.row_empirical[-1]
        col_final = result.col_empirical[-1]
        assert row_final[1] > 0.9
        assert col_final[1] > 0.9


class TestExpectedPayoffs:
    """Test expected payoff computation."""

    def test_pure_strategy_payoffs(self):
        A = np.array([[3, 0], [5, 1]], dtype=np.float64)
        B = np.array([[3, 5], [0, 1]], dtype=np.float64)
        fp = FictitiousPlay(A, B)

        # Pure strategy (0, 0)
        r, c = fp.expected_payoffs(np.array([1, 0]), np.array([1, 0]))
        assert r == 3.0
        assert c == 3.0

        # Pure strategy (1, 1)
        r, c = fp.expected_payoffs(np.array([0, 1]), np.array([0, 1]))
        assert r == 1.0
        assert c == 1.0

    def test_mixed_strategy_payoffs(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        fp = FictitiousPlay(A, -A)

        # Uniform mix in matching pennies => 0 payoff
        r, c = fp.expected_payoffs(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
        assert np.isclose(r, 0.0)
        assert np.isclose(c, 0.0)
