# Fictitious Play Convergence Analyzer

A comprehensive Python toolkit for analyzing **Fictitious Play** learning dynamics in two-player normal-form games. Implements classical and smooth (logit) fictitious play, convergence diagnostics, game classification, cycling detection, and interactive simplex trajectory visualization.

## Background

### What is Fictitious Play?

**Fictitious Play** (Brown, 1951; Robinson, 1951) is one of the oldest and most fundamental learning algorithms in game theory. Two players repeatedly play a normal-form game. At each round, each player:

1. Observes the empirical frequency of the opponent's past actions.
2. Plays a **best response** to that empirical distribution.

Formally, let $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{m \times n}$ be the payoff matrices for the row and column player respectively. At time $t$, the empirical strategy of the column player is:

$$\hat{\sigma}_2^t = \frac{1}{t} \sum_{s=1}^{t} e_{j_s}$$

The row player then selects:

$$i_{t+1} \in \arg\max_{i} \; (A \hat{\sigma}_2^t)_i$$

and symmetrically for the column player.

### Convergence Theory

**Theorem (Robinson, 1951).** In every finite two-player **zero-sum** game, the empirical strategies of fictitious play converge to the set of **minimax** (Nash) equilibria.

The convergence rate is $O(1/t)$ in the distance to the Nash equilibrium set, which is tight for certain games.

**Theorem (Monderer & Shapley, 1996).** In every finite **potential game**, fictitious play converges to a Nash equilibrium.

### Shapley's Counterexample

Not all games converge under FP. **Shapley (1964)** constructed a $3 \times 3$ game where fictitious play cycles forever:

$$A = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}, \qquad B = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$$

The unique Nash equilibrium is $\left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right)$ for both players, but the empirical strategies spiral along the boundary of the simplex and never converge.

### Smooth Fictitious Play

**Smooth (logit) Fictitious Play** replaces the hard argmax with a **softmax** (logistic) best response:

$$\sigma_i^{BR}(a) = \frac{\exp\left( (A \hat{\sigma}_2^t)_a / \tau \right)}{\sum_{a'} \exp\left( (A \hat{\sigma}_2^t)_{a'} / \tau \right)}$$

where $\tau > 0$ is a temperature parameter. As $\tau \to 0$, this recovers classical FP. With a **cooling schedule** $\tau(t) \to 0$, smooth FP can converge in games where classical FP cycles.

### Exploitability

The **exploitability** of a strategy profile $(\sigma_1, \sigma_2)$ measures how far it is from a Nash equilibrium:

$$\text{exploit}(\sigma) = \max_i e_i^T A \sigma_2 - \sigma_1^T A \sigma_2 + \max_j \sigma_1^T B e_j - \sigma_1^T B \sigma_2$$

A profile is a Nash equilibrium if and only if its exploitability is zero.

## Features

- **Classical Fictitious Play** with full history tracking
- **Smooth (Logit) Fictitious Play** with configurable temperature and cooling schedules
- **Deterministic Smooth FP** variant for clean trajectory visualization
- **Convergence diagnostics**: L2 distance to NE, exploitability trajectory, convergence rate estimation
- **Cycling detection** via autocorrelation analysis
- **Game classification**: zero-sum, potential, symmetric, generic
- **Pure Nash equilibrium finder** and dominant strategy detection
- **Simplex trajectory visualization** for 3-strategy games
- **Interactive Streamlit dashboard** with preloaded game zoo
- **CLI** for quick experiments

## Installation

```bash
# Clone
git clone https://github.com/abhi-wadhwa/fictitious-play.git
cd fictitious-play

# Install (production)
pip install -e .

# Install (development)
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
import numpy as np
from src.core.fictitious_play import FictitiousPlay
from src.core.convergence import ConvergenceDiagnostics

# Rock-Paper-Scissors
A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
fp = FictitiousPlay(A, -A)
result = fp.run(5000, seed=42)

print(f"Final row strategy: {result.row_empirical[-1]}")
# ~[0.333, 0.333, 0.333]

# Check convergence
ne = [(np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3]))]
diag = ConvergenceDiagnostics(A, -A, nash_equilibria=ne)
print(f"Exploitability: {diag.exploitability(result.row_empirical[-1], result.col_empirical[-1]):.6f}")
```

### Smooth FP with Cooling

```python
from src.core.smooth_fp import SmoothFictitiousPlay

sfp = SmoothFictitiousPlay(A, -A, temperature=lambda t: 5.0/t)
result = sfp.run_deterministic(3000)
# Converges even in games where classical FP cycles
```

### Game Classification

```python
from src.core.game_classifier import GameClassifier

gc = GameClassifier(A, -A)
result = gc.classify()
print(result.game_type)     # GameType.ZERO_SUM
print(result.is_zero_sum)   # True
print(result.is_potential)  # False
```

### CLI

```bash
# Classical FP on Rock-Paper-Scissors
python -m src.cli --game rps --iterations 5000

# Smooth FP on Shapley's cycling game
python -m src.cli --game shapley --algo smooth --cooling --iterations 5000

# With plots
python -m src.cli --game matching-pennies --plot
```

### Streamlit Dashboard

```bash
streamlit run src/viz/app.py
```

## Game Zoo

| Game | Type | FP Convergence |
|------|------|----------------|
| Rock-Paper-Scissors | Zero-sum | Converges to (1/3, 1/3, 1/3) |
| Matching Pennies | Zero-sum | Converges to (1/2, 1/2) |
| Prisoner's Dilemma | Generic | Converges immediately (dominant strategy) |
| Battle of the Sexes | Generic | Converges to a NE |
| Shapley's 3x3 | Generic | **Cycles** (does NOT converge) |
| Coordination Game | Potential | Converges to a pure NE |

## Project Structure

```
fictitious-play/
├── src/
│   ├── core/
│   │   ├── fictitious_play.py    # Classical FP algorithm
│   │   ├── smooth_fp.py          # Smooth (logit) FP
│   │   ├── convergence.py        # Diagnostics & cycling detection
│   │   └── game_classifier.py    # Zero-sum/potential/symmetric detection
│   ├── viz/
│   │   ├── simplex.py            # Simplex trajectory plots
│   │   └── app.py                # Streamlit dashboard
│   └── cli.py                    # Command-line interface
├── tests/                        # Comprehensive test suite
├── examples/
│   └── demo.py                   # Runnable demo script
├── pyproject.toml
├── Makefile
├── Dockerfile
└── .github/workflows/ci.yml
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Type check
make type-check

# Run demo
make demo
```

## Docker

```bash
docker build -t fictitious-play .
docker run -p 8501:8501 fictitious-play
# Open http://localhost:8501
```

## References

- Brown, G. W. (1951). *Iterative solution of games by fictitious play.* Activity Analysis of Production and Allocation.
- Robinson, J. (1951). *An iterative method of solving a game.* Annals of Mathematics, 54(2), 296-301.
- Shapley, L. S. (1964). *Some topics in two-person games.* Advances in Game Theory, 1-28.
- Fudenberg, D. & Levine, D. K. (1998). *The Theory of Learning in Games.* MIT Press.
- Monderer, D. & Shapley, L. S. (1996). *Potential games.* Games and Economic Behavior, 14(1), 124-143.

## License

MIT
