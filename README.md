# fictitious-play

fictitious play learning dynamics — the oldest and simplest algorithm for learning in games. each player best-responds to the empirical frequency of opponents.

## what this is

- **classical fictitious play** — each round, play a best response to the opponent historical average. provably converges in 2x2 games, zero-sum games, potential games, and supermodular games
- **smooth fictitious play** — add a logistic perturbation (entropy regularization). converges more broadly but to QRE, not NE
- **convergence analysis** — track exploitability over time. does it go to zero?
- **simplex trajectory visualization** — watch the empirical distribution trace a path on the strategy simplex

## running it

```bash
pip install -r requirements.txt
python main.py
```

## the idea

fictitious play (brown 1951) is beautifully naive: assume your opponent is playing a stationary mixed strategy equal to their historical frequencies, then best-respond to that. repeat.

the surprising part: this often converges to nash equilibrium despite the assumption being wrong (opponents are also adapting). the famous non-convergence example is shapley's 3x3 game where the dynamics cycle. smooth FP fixes this at the cost of converging to a perturbed equilibrium.
