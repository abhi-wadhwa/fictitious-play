"""Streamlit interactive dashboard for Fictitious Play analysis.

Launch with:
    streamlit run src/viz/app.py
"""

from __future__ import annotations

import sys
import os
import numpy as np
import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.fictitious_play import FictitiousPlay
from src.core.smooth_fp import SmoothFictitiousPlay
from src.core.convergence import ConvergenceDiagnostics
from src.core.game_classifier import GameClassifier, game_zoo
from src.viz.simplex import (
    plot_simplex_trajectory,
    plot_dual_simplex,
    plot_convergence,
    plot_belief_evolution,
)

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Fictitious Play Analyzer",
    page_icon="🎯",
    layout="wide",
)

st.title("Fictitious Play Convergence Analyzer")
st.markdown(
    """
    Explore classical and smooth fictitious play dynamics on two-player
    normal-form games.  Visualize strategy trajectories on the simplex,
    track convergence to Nash equilibria, and detect cycling behaviour.
    """
)

# -----------------------------------------------------------------
# Sidebar: game selection
# -----------------------------------------------------------------
st.sidebar.header("Game Selection")

zoo = game_zoo()
game_names = list(zoo.keys()) + ["Custom"]
selected_game = st.sidebar.selectbox("Choose a game", game_names)

if selected_game == "Custom":
    st.sidebar.subheader("Custom Payoff Matrices")
    m = st.sidebar.number_input("Rows (m)", min_value=2, max_value=10, value=3)
    n = st.sidebar.number_input("Columns (n)", min_value=2, max_value=10, value=3)
    st.sidebar.markdown("**Row player payoff A:**")
    A_text = st.sidebar.text_area(
        "A (rows separated by newlines, values by commas)",
        value="0, -1, 1\n1, 0, -1\n-1, 1, 0",
    )
    st.sidebar.markdown("**Column player payoff B:**")
    B_text = st.sidebar.text_area(
        "B (same format)",
        value="0, 1, -1\n-1, 0, 1\n1, -1, 0",
    )
    try:
        A = np.array(
            [[float(x) for x in row.split(",")] for row in A_text.strip().split("\n")]
        )
        B = np.array(
            [[float(x) for x in row.split(",")] for row in B_text.strip().split("\n")]
        )
    except Exception as e:
        st.error(f"Error parsing matrices: {e}")
        st.stop()

    ne_text = st.sidebar.text_area(
        "Nash equilibria (optional, row;col per line)",
        value="",
        help="Format: 0.33,0.33,0.34;0.33,0.33,0.34",
    )
    nash_equilibria = []
    if ne_text.strip():
        for line in ne_text.strip().split("\n"):
            parts = line.split(";")
            if len(parts) == 2:
                row_ne = np.array([float(x) for x in parts[0].split(",")])
                col_ne = np.array([float(x) for x in parts[1].split(",")])
                nash_equilibria.append((row_ne, col_ne))
    description = "Custom game."
else:
    game_data = zoo[selected_game]
    A = game_data["A"]
    B = game_data["B"]
    description = game_data["description"]
    nash_equilibria = game_data.get("nash_equilibria", [])

m_size, n_size = A.shape

# -----------------------------------------------------------------
# Sidebar: algorithm settings
# -----------------------------------------------------------------
st.sidebar.header("Algorithm Settings")

algo = st.sidebar.radio(
    "Algorithm",
    ["Classical FP", "Smooth FP (stochastic)", "Smooth FP (deterministic)"],
)

num_iter = st.sidebar.slider(
    "Iterations", min_value=50, max_value=10000, value=1000, step=50
)

seed = st.sidebar.number_input("Random seed", min_value=0, value=42)

if algo != "Classical FP":
    temp_mode = st.sidebar.radio(
        "Temperature schedule", ["Constant", "Cooling (1/t)", "Cooling (1/sqrt(t))"]
    )
    if temp_mode == "Constant":
        tau = st.sidebar.slider(
            "Temperature", min_value=0.01, max_value=5.0, value=1.0, step=0.01
        )
        tau_fn = tau  # type: ignore
    elif temp_mode == "Cooling (1/t)":
        tau_fn = lambda t: 1.0 / t
    else:
        tau_fn = lambda t: 1.0 / np.sqrt(t)

# -----------------------------------------------------------------
# Game info
# -----------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Game Info")
    st.markdown(f"**{selected_game}**: {description}")
    st.markdown(f"**Size**: {m_size} x {n_size}")

    classifier = GameClassifier(A, B)
    result = classifier.classify()
    st.markdown(f"**Type**: {result.game_type.name}")
    st.markdown(f"*{result.details}*")

    pure_ne = classifier.pure_nash_equilibria()
    if pure_ne:
        st.markdown(f"**Pure NE**: {pure_ne}")
    dom = classifier.dominant_strategies()
    if dom[0] is not None or dom[1] is not None:
        st.markdown(f"**Dominant strategies**: Row={dom[0]}, Col={dom[1]}")

with col2:
    st.subheader("Payoff Matrices")
    st.markdown("**A (Row player)**")
    st.dataframe(A, use_container_width=True)
    st.markdown("**B (Column player)**")
    st.dataframe(B, use_container_width=True)

# -----------------------------------------------------------------
# Run algorithm
# -----------------------------------------------------------------
st.divider()
st.subheader("Simulation Results")

if algo == "Classical FP":
    fp = FictitiousPlay(A, B)
    fp_result = fp.run(num_iter, seed=seed)
    row_emp = fp_result.row_empirical
    col_emp = fp_result.col_empirical
elif algo == "Smooth FP (stochastic)":
    sfp = SmoothFictitiousPlay(A, B, temperature=tau_fn)
    sfp_result = sfp.run(num_iter, seed=seed)
    row_emp = sfp_result.row_empirical
    col_emp = sfp_result.col_empirical
else:
    sfp = SmoothFictitiousPlay(A, B, temperature=tau_fn)
    sfp_result = sfp.run_deterministic(num_iter)
    row_emp = sfp_result.row_empirical
    col_emp = sfp_result.col_empirical

# Convergence diagnostics
diag = ConvergenceDiagnostics(A, B, nash_equilibria=nash_equilibria)
exploit_traj = diag.exploitability_trajectory(row_emp, col_emp)

if nash_equilibria:
    dist_traj = diag.distance_trajectory(row_emp, col_emp)
else:
    dist_traj = None

is_cycling, period, peak_ac = diag.detect_cycling(row_emp, col_emp)

# -----------------------------------------------------------------
# Display results
# -----------------------------------------------------------------
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Final exploitability", f"{exploit_traj[-1]:.6f}")
with metric_cols[1]:
    if dist_traj is not None:
        st.metric("Final dist to NE", f"{dist_traj[-1]:.6f}")
    else:
        st.metric("Final dist to NE", "N/A (no NE provided)")
with metric_cols[2]:
    st.metric("Cycling detected", "Yes" if is_cycling else "No")
with metric_cols[3]:
    if is_cycling and period is not None:
        st.metric("Cycle period", f"~{period} rounds")
    else:
        st.metric("Cycle period", "N/A")

# -----------------------------------------------------------------
# Simplex plots (only for 3-strategy games)
# -----------------------------------------------------------------
if m_size == 3 or n_size == 3:
    st.subheader("Simplex Trajectories")
    if m_size == 3 and n_size == 3:
        fig_simplex = plot_dual_simplex(
            row_emp, col_emp,
            title=f"{selected_game} — Simplex Trajectories",
        )
        st.pyplot(fig_simplex)
    elif m_size == 3:
        import matplotlib.pyplot as plt
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 5.5))
        plot_simplex_trajectory(row_emp, ax=ax_s, title="Row Player Simplex")
        st.pyplot(fig_s)
    elif n_size == 3:
        import matplotlib.pyplot as plt
        fig_s, ax_s = plt.subplots(1, 1, figsize=(6, 5.5))
        plot_simplex_trajectory(col_emp, ax=ax_s, title="Column Player Simplex")
        st.pyplot(fig_s)

# -----------------------------------------------------------------
# Belief evolution
# -----------------------------------------------------------------
st.subheader("Belief Evolution")
tab1, tab2 = st.tabs(["Row Player", "Column Player"])
with tab1:
    fig_be_r = plot_belief_evolution(
        row_emp, title="Row Player — Empirical Strategy Over Time"
    )
    st.pyplot(fig_be_r)
with tab2:
    fig_be_c = plot_belief_evolution(
        col_emp, title="Column Player — Empirical Strategy Over Time"
    )
    st.pyplot(fig_be_c)

# -----------------------------------------------------------------
# Convergence plots
# -----------------------------------------------------------------
st.subheader("Convergence Analysis")
if dist_traj is not None:
    fig_conv = plot_convergence(
        dist_traj, exploit_traj,
        title=f"{selected_game} — Convergence to NE",
    )
else:
    fig_conv = plot_convergence(
        exploit_traj,
        title=f"{selected_game} — Exploitability Over Time",
    )
st.pyplot(fig_conv)

# Convergence rate
if dist_traj is not None and np.any(dist_traj > 1e-15):
    slope, intercept = ConvergenceDiagnostics.estimate_convergence_rate(dist_traj)
    st.markdown(
        f"**Estimated convergence rate**: distance ~ t^({slope:.3f})  "
        f"(theoretical for zero-sum: t^(-1))"
    )

# -----------------------------------------------------------------
# Final strategy
# -----------------------------------------------------------------
st.subheader("Final Empirical Strategies")
fcol1, fcol2 = st.columns(2)
with fcol1:
    st.markdown("**Row player**")
    st.bar_chart(
        {f"Action {i}": float(row_emp[-1][i]) for i in range(m_size)},
    )
with fcol2:
    st.markdown("**Column player**")
    st.bar_chart(
        {f"Action {j}": float(col_emp[-1][j]) for j in range(n_size)},
    )
