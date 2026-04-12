"""Demonstration of Fictitious Play analysis.

Run from the project root:
    python examples/demo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.core.fictitious_play import FictitiousPlay
from src.core.smooth_fp import SmoothFictitiousPlay
from src.core.convergence import ConvergenceDiagnostics
from src.core.game_classifier import GameClassifier, game_zoo


def demo_zero_sum_convergence():
    """Demonstrate FP convergence in Rock-Paper-Scissors."""
    print("=" * 60)
    print("DEMO 1: Classical FP in Rock-Paper-Scissors (zero-sum)")
    print("=" * 60)

    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=np.float64)
    B = -A

    # Classify
    gc = GameClassifier(A, B)
    result = gc.classify()
    print(f"Game type: {result.game_type.name}")
    print(f"  {result.details}")

    # Run FP
    fp = FictitiousPlay(A, B)
    fp_result = fp.run(3000, seed=42)

    ne = [(np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3]))]
    diag = ConvergenceDiagnostics(A, B, nash_equilibria=ne)

    dist_final = diag.distance_to_ne(
        fp_result.row_empirical[-1], fp_result.col_empirical[-1]
    )
    exploit = diag.exploitability(
        fp_result.row_empirical[-1], fp_result.col_empirical[-1]
    )

    print(f"\nAfter 3000 iterations:")
    print(f"  Row strategy: {np.array2string(fp_result.row_empirical[-1], precision=4)}")
    print(f"  Col strategy: {np.array2string(fp_result.col_empirical[-1], precision=4)}")
    print(f"  Distance to NE: {dist_final:.6f}")
    print(f"  Exploitability: {exploit:.6f}")
    print()


def demo_shapley_cycling():
    """Demonstrate cycling in Shapley's 3x3 game."""
    print("=" * 60)
    print("DEMO 2: Classical FP in Shapley's 3x3 Game (cycling)")
    print("=" * 60)

    A = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    B = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)

    gc = GameClassifier(A, B)
    result = gc.classify()
    print(f"Game type: {result.game_type.name}")
    print(f"  {result.details}")

    fp = FictitiousPlay(A, B)
    fp_result = fp.run(5000, seed=0)

    ne = [(np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3]))]
    diag = ConvergenceDiagnostics(A, B, nash_equilibria=ne)

    dist_final = diag.distance_to_ne(
        fp_result.row_empirical[-1], fp_result.col_empirical[-1]
    )
    exploit = diag.exploitability(
        fp_result.row_empirical[-1], fp_result.col_empirical[-1]
    )

    is_cycling, period, peak = diag.detect_cycling(
        fp_result.row_empirical, fp_result.col_empirical,
        threshold=0.15, min_period=10,
    )

    print(f"\nAfter 5000 iterations:")
    print(f"  Row strategy: {np.array2string(fp_result.row_empirical[-1], precision=4)}")
    print(f"  Col strategy: {np.array2string(fp_result.col_empirical[-1], precision=4)}")
    print(f"  Distance to NE: {dist_final:.6f}")
    print(f"  Exploitability: {exploit:.6f}")
    print(f"  Cycling detected: {is_cycling}")
    if period:
        print(f"  Estimated period: ~{period}")
    print()


def demo_smooth_fp():
    """Demonstrate smooth FP with cooling."""
    print("=" * 60)
    print("DEMO 3: Smooth FP with cooling in Matching Pennies")
    print("=" * 60)

    A = np.array([[1, -1], [-1, 1]], dtype=np.float64)
    B = -A

    sfp = SmoothFictitiousPlay(A, B, temperature=lambda t: 5.0 / t)
    result = sfp.run_deterministic(3000)

    ne = [(np.array([0.5, 0.5]), np.array([0.5, 0.5]))]
    diag = ConvergenceDiagnostics(A, B, nash_equilibria=ne)

    dist_final = diag.distance_to_ne(
        result.row_empirical[-1], result.col_empirical[-1]
    )

    print(f"\nAfter 3000 iterations (deterministic smooth FP, tau = 5/t):")
    print(f"  Row strategy: {np.array2string(result.row_empirical[-1], precision=4)}")
    print(f"  Col strategy: {np.array2string(result.col_empirical[-1], precision=4)}")
    print(f"  Distance to NE: {dist_final:.6f}")
    print()


def demo_game_zoo():
    """Demonstrate the game zoo and classification."""
    print("=" * 60)
    print("DEMO 4: Game Zoo — Classification Summary")
    print("=" * 60)

    zoo = game_zoo()
    for name, data in zoo.items():
        A, B = data["A"], data["B"]
        gc = GameClassifier(A, B)
        result = gc.classify()
        pure_ne = gc.pure_nash_equilibria()
        dom = gc.dominant_strategies()

        print(f"\n  {name}:")
        print(f"    Type: {result.game_type.name}")
        print(f"    Zero-sum: {result.is_zero_sum}")
        print(f"    Potential: {result.is_potential}")
        print(f"    Symmetric: {result.is_symmetric}")
        print(f"    Pure NE: {pure_ne if pure_ne else 'None'}")
        print(f"    Dominant strategies: Row={dom[0]}, Col={dom[1]}")

    print()


if __name__ == "__main__":
    demo_zero_sum_convergence()
    demo_shapley_cycling()
    demo_smooth_fp()
    demo_game_zoo()
    print("All demos completed successfully.")
