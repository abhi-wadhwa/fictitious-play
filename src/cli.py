"""Command-line interface for Fictitious Play analysis.

Usage examples:
    python -m src.cli --game rps --iterations 2000 --algo classical
    python -m src.cli --game shapley --iterations 5000 --algo smooth --tau 0.5
    python -m src.cli --game matching-pennies --plot
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

from src.core.fictitious_play import FictitiousPlay
from src.core.smooth_fp import SmoothFictitiousPlay
from src.core.convergence import ConvergenceDiagnostics
from src.core.game_classifier import GameClassifier, game_zoo


GAME_ALIASES = {
    "rps": "Rock-Paper-Scissors",
    "matching-pennies": "Matching Pennies",
    "prisoners-dilemma": "Prisoner's Dilemma",
    "battle-of-sexes": "Battle of the Sexes",
    "shapley": "Shapley's Game (3x3 cycling)",
    "coordination": "Coordination Game",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fictitious Play Convergence Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--game",
        type=str,
        default="rps",
        choices=list(GAME_ALIASES.keys()),
        help="Predefined game to analyze (default: rps)",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=1000,
        help="Number of FP iterations (default: 1000)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="classical",
        choices=["classical", "smooth", "smooth-det"],
        help="Algorithm variant (default: classical)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for smooth FP (default: 1.0)",
    )
    parser.add_argument(
        "--cooling",
        action="store_true",
        help="Use 1/t cooling schedule for smooth FP",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plots after the run",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load game
    full_name = GAME_ALIASES[args.game]
    zoo = game_zoo()
    game_data = zoo[full_name]
    A, B = game_data["A"], game_data["B"]
    nash_eq = game_data.get("nash_equilibria", [])

    print(f"Game: {full_name}")
    print(f"Description: {game_data['description']}")
    print(f"Payoff matrix A:\n{A}")
    print(f"Payoff matrix B:\n{B}")
    print()

    # Classify
    classifier = GameClassifier(A, B)
    cls_result = classifier.classify()
    print(f"Classification: {cls_result.game_type.name}")
    print(f"  {cls_result.details}")
    pure_ne = classifier.pure_nash_equilibria()
    if pure_ne:
        print(f"Pure NE: {pure_ne}")
    print()

    # Run FP
    if args.algo == "classical":
        print(f"Running classical FP for {args.iterations} iterations (seed={args.seed})...")
        fp = FictitiousPlay(A, B)
        result = fp.run(args.iterations, seed=args.seed)
        row_emp = result.row_empirical
        col_emp = result.col_empirical
    elif args.algo == "smooth":
        tau_fn = (lambda t: 1.0 / t) if args.cooling else args.tau
        print(f"Running smooth FP (stochastic) for {args.iterations} iterations...")
        sfp = SmoothFictitiousPlay(A, B, temperature=tau_fn)
        sfp_result = sfp.run(args.iterations, seed=args.seed)
        row_emp = sfp_result.row_empirical
        col_emp = sfp_result.col_empirical
    else:
        tau_fn = (lambda t: 1.0 / t) if args.cooling else args.tau
        print(f"Running smooth FP (deterministic) for {args.iterations} iterations...")
        sfp = SmoothFictitiousPlay(A, B, temperature=tau_fn)
        sfp_result = sfp.run_deterministic(args.iterations)
        row_emp = sfp_result.row_empirical
        col_emp = sfp_result.col_empirical

    # Diagnostics
    diag = ConvergenceDiagnostics(A, B, nash_equilibria=nash_eq)
    exploit = diag.exploitability(row_emp[-1], col_emp[-1])
    print(f"\nFinal empirical strategies:")
    print(f"  Row: {np.array2string(row_emp[-1], precision=4)}")
    print(f"  Col: {np.array2string(col_emp[-1], precision=4)}")
    print(f"  Exploitability: {exploit:.6f}")

    if nash_eq:
        dist = diag.distance_to_ne(row_emp[-1], col_emp[-1])
        print(f"  Distance to NE: {dist:.6f}")

        dist_traj = diag.distance_trajectory(row_emp, col_emp)
        slope, _ = ConvergenceDiagnostics.estimate_convergence_rate(dist_traj)
        print(f"  Convergence rate: ~t^({slope:.3f})")

    is_cycling, period, peak_ac = diag.detect_cycling(row_emp, col_emp)
    if is_cycling:
        print(f"  Cycling detected! Period ~{period}, autocorrelation peak={peak_ac:.3f}")
    else:
        print("  No cycling detected.")

    # Plots
    if args.plot:
        from src.viz.simplex import (
            plot_dual_simplex,
            plot_convergence,
            plot_belief_evolution,
        )
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        m, n = A.shape
        if m == 3 and n == 3:
            fig1 = plot_dual_simplex(row_emp, col_emp, title=full_name)

        if nash_eq:
            dist_traj = diag.distance_trajectory(row_emp, col_emp)
            exploit_traj = diag.exploitability_trajectory(row_emp, col_emp)
            fig2 = plot_convergence(dist_traj, exploit_traj, title=f"{full_name} Convergence")
        else:
            exploit_traj = diag.exploitability_trajectory(row_emp, col_emp)
            fig2 = plot_convergence(exploit_traj, title=f"{full_name} Exploitability")

        fig3 = plot_belief_evolution(row_emp, title="Row Player Beliefs")
        plt.show()


if __name__ == "__main__":
    main()
