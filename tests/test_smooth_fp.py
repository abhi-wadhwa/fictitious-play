"""Tests for Smooth Fictitious Play."""

import numpy as np
import pytest

from src.core.smooth_fp import SmoothFictitiousPlay, SmoothFPResult


class TestSmoothFPBasic:
    """Basic sanity checks for smooth FP."""

    def test_result_lengths(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.float64)
        B = np.array([[1, 0], [0, 1]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, B, temperature=1.0)
        result = sfp.run(100, seed=0)

        assert len(result.row_actions) == 100
        assert len(result.col_actions) == 100
        assert len(result.row_empirical) == 100
        assert len(result.col_empirical) == 100
        assert len(result.row_smooth_br) == 100
        assert len(result.col_smooth_br) == 100
        assert len(result.temperatures) == 100

    def test_smooth_br_is_distribution(self):
        """Smooth BR must be a valid probability distribution."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, -A, temperature=0.5)
        result = sfp.run(100, seed=1)

        for br in result.row_smooth_br:
            assert np.all(br >= 0)
            assert np.isclose(br.sum(), 1.0)
        for br in result.col_smooth_br:
            assert np.all(br >= 0)
            assert np.isclose(br.sum(), 1.0)

    def test_invalid_temperature_raises(self):
        A = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError):
            SmoothFictitiousPlay(A, A, temperature=-1.0)
        with pytest.raises(ValueError):
            SmoothFictitiousPlay(A, A, temperature=0.0)


class TestSoftmax:
    """Test the softmax function directly."""

    def test_uniform_with_equal_logits(self):
        """Equal logits produce uniform distribution."""
        logits = np.array([1.0, 1.0, 1.0])
        result = SmoothFictitiousPlay.softmax(logits, temperature=1.0)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3])

    def test_high_temperature_is_uniform(self):
        """High temperature makes any logits nearly uniform."""
        logits = np.array([10.0, 0.0, -10.0])
        result = SmoothFictitiousPlay.softmax(logits, temperature=1000.0)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    def test_low_temperature_concentrates(self):
        """Low temperature concentrates on the max."""
        logits = np.array([10.0, 0.0, -10.0])
        result = SmoothFictitiousPlay.softmax(logits, temperature=0.001)
        assert result[0] > 0.99

    def test_numerical_stability(self):
        """Softmax should not overflow with large logits."""
        logits = np.array([1000.0, 999.0, 998.0])
        result = SmoothFictitiousPlay.softmax(logits, temperature=1.0)
        assert np.all(np.isfinite(result))
        assert np.isclose(result.sum(), 1.0)


class TestSmoothFPConvergence:
    """Smooth FP convergence tests."""

    def test_cooling_converges_matching_pennies(self):
        """Smooth FP with 1/t cooling should converge in matching pennies."""
        A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, -A, temperature=lambda t: 1.0 / t)
        result = sfp.run_deterministic(3000)

        row_final = result.row_empirical[-1]
        np.testing.assert_allclose(row_final, [0.5, 0.5], atol=0.1)

    def test_cooling_converges_rps(self):
        """Smooth FP with cooling should approach NE in RPS."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, -A, temperature=lambda t: 5.0 / t)
        result = sfp.run_deterministic(3000)

        row_final = result.row_empirical[-1]
        expected = np.array([1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(row_final, expected, atol=0.1)

    def test_constant_temperature_produces_smooth_response(self):
        """With constant temperature, smooth BR should differ from argmax."""
        A = np.array([[3, 0], [0, 1]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, A, temperature=1.0)
        result = sfp.run(200, seed=5)

        # At temperature 1.0, responses should not be pure (all 0s and 1s)
        for br in result.row_smooth_br:
            assert br.min() > 0.001  # Not purely concentrated


class TestDeterministicSmooth:
    """Tests for deterministic smooth FP variant."""

    def test_result_is_deterministic(self):
        """Two runs with same params give same trajectory."""
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, -A, temperature=1.0)

        r1 = sfp.run_deterministic(100)
        r2 = sfp.run_deterministic(100)

        np.testing.assert_array_equal(
            np.array(r1.row_empirical), np.array(r2.row_empirical)
        )
        np.testing.assert_array_equal(
            np.array(r1.col_empirical), np.array(r2.col_empirical)
        )

    def test_empirical_sums_to_one(self):
        A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
        sfp = SmoothFictitiousPlay(A, -A, temperature=0.5)
        result = sfp.run_deterministic(200)

        for emp in result.row_empirical:
            assert np.isclose(emp.sum(), 1.0, atol=1e-10)
