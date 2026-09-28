"""
Interactive demo: Hawking/Unruh radiation via Gaussian state tomography.

Run from the repo root:
    python -m streamlit run app/app.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import streamlit as st

# src/ uses flat imports (import simulate), so put it on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import simulate
import metrics

st.set_page_config(page_title="Horizon Radiation Tomography", layout="wide")


# helpers

def ellipse(cov, n_sigma=2.0, n_pts=200):
    """Points on the n-sigma contour of a zero-mean Gaussian with covariance cov."""
    vals, vecs = jnp.linalg.eigh(cov)
    vals = jnp.clip(vals, 0, None)
    t = jnp.linspace(0, 2 * jnp.pi, n_pts)
    circle = jnp.stack([jnp.cos(t), jnp.sin(t)])
    points = (vecs @ (n_sigma * jnp.sqrt(vals)[:, None] * circle)).T
    return jax.device_get(points)


def reconstruct(samples, measurement_type, sigma, subtract_noise):
    """
    Estimate the accessible-mode covariance from measured samples.

    - homodyne: x and p are never observed in the same shot, so Cov(x,p) is
      unmeasurable. We estimate Var(x), Var(p) from their own sub-samples and
      ASSUME zero cross-covariance (true for the thermal state family).
    - heterodyne: the known +I vacuum penalty is subtracted.
    - optionally subtract the known detector-noise variance sigma^2 * I.
    """
    if bool(jnp.isnan(samples).any()):
        vx = jnp.nanvar(samples[:, 0], ddof=1)
        vp = jnp.nanvar(samples[:, 1], ddof=1)
        V = jnp.diag(jnp.array([vx, vp]))
    else:
        V = jnp.cov(samples.T, ddof=1)
    if measurement_type == "heterodyne":
        V = V - jnp.eye(2)
    if subtract_noise:
        V = V - sigma ** 2 * jnp.eye(2)
    return V


def draw_cov(ax, cov, **kw):
    e = ellipse(cov)
    ax.plot(e[:, 0], e[:, 1], **kw)


# ---------- sidebar ----------

st.sidebar.header("Parameters")
r = st.sidebar.slider("Squeezing r", 0.05, 2.0, 0.5, 0.05)
n_samples = st.sidebar.select_slider(
    "Samples N", options=[50, 100, 500, 1000, 5000, 10000], value=1000)
sigma = st.sidebar.select_slider(
    "Detector noise σ", options=[0.0, 0.05, 0.1, 0.2, 0.5], value=0.1)
measurement_type = st.sidebar.selectbox(
    "Measurement type", ["joint", "homodyne", "heterodyne"])
subtract_noise = st.sidebar.checkbox("Subtract known detector noise σ²", value=False)

if "seed" not in st.session_state:
    st.session_state.seed = 42
if st.sidebar.button("Resample (new random seed)"):
    st.session_state.seed += 1
seed = st.session_state.seed
st.sidebar.caption(f"seed = {seed}")

V_full = simulate.generate_covariance_matrix(r)
V_A = simulate.reduce_to_accessible(V_full)
joint_key, sample_key, *comparison_keys = random.split(random.key(seed), 5)

st.title("Horizon Radiation Tomography")
st.write(
    "A horizon is modeled as a two-mode squeezed state: one mode the outside "
    "observer can measure, one hidden behind the horizon. Tracing out the hidden "
    "mode leaves the observer a **thermal** state. How much can they recover "
    "from finite, noisy data?"
)

tab1, tab2, tab3 = st.tabs(
    ["1. Why is it thermal?", "2. Live reconstruction", "3. Measurement types"])

# Tab 1 - Why is it thermal?

with tab1:
    joint = jax.device_get(
        simulate.sample_idealized_joint_access(V_full, 2000, 0.0, joint_key))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(joint[:, 0], joint[:, 2], s=6, alpha=0.4)
        ax.set(title="Both modes: X_A vs X_B (correlated)",
               xlabel="X_A (accessible)", ylabel="X_B (hidden)")
        ax.axis("equal"); ax.grid(True)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(joint[:, 0], joint[:, 1], s=6, alpha=0.4, color="tab:orange")
        ax.set(title="Accessible mode only: X_A vs P_A (thermal blob)",
               xlabel="X_A", ylabel="P_A")
        ax.axis("equal"); ax.grid(True)
        st.pyplot(fig)
    st.info(
        f"The joint state is a pure, strongly correlated pair. Ignoring the hidden "
        f"mode, the observer sees a round thermal state with variance cosh(2r) = "
        f"{float(jnp.cosh(2 * r)):.2f} in each quadrature, and mean particle number "
        f"N = sinh²(r) = {float(jnp.sinh(r) ** 2):.3f}."
    )

# Tab 2 - Live reconstruction

with tab2:
    samples = simulate.sample_measurements(
        V_A, n_samples, sigma, measurement_type, sample_key)
    samples_host = jax.device_get(samples)
    V_hat = reconstruct(samples, measurement_type, sigma, subtract_noise)

    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(samples_host[:, 0], samples_host[:, 1], s=6, alpha=0.3,
               label="measured samples")
        draw_cov(ax, V_A, color="green", lw=2, label="true covariance (2σ)")
        draw_cov(ax, V_hat, color="red", lw=2, ls="--", label="estimated covariance (2σ)")
        ax.set(xlabel="X quadrature", ylabel="P quadrature",
               title=f"{measurement_type}, N={n_samples}, σ={sigma}")
        ax.axis("equal"); ax.grid(True); ax.legend(loc="upper right")
        st.pyplot(fig)
    with c2:
        r_hat = float(metrics.estimate_squeezing(V_hat))
        rows = {
            "r": (float(r), r_hat),
            "Particle number N": (float(metrics.compute_particle_number(V_A)),
                                  float(metrics.compute_particle_number(V_hat))),
            "Entropy S (bits)": (float(metrics.compute_entropy(V_A)),
                                 float(metrics.compute_entropy(V_hat))),
        }
        st.table({
            "quantity": list(rows),
            "true": [f"{t:.3f}" for t, _ in rows.values()],
            "estimated": [f"{e:.3f}" for _, e in rows.values()],
            "abs. error": [f"{abs(t - e):.3f}" for t, e in rows.values()],
        })
        st.metric("Covariance error (Frobenius)",
              f"{float(metrics.matrix_error(V_A, V_hat)):.4f}")
        if measurement_type == "homodyne":
            st.caption("Homodyne never sees x and p together, so the off-diagonal "
                       "covariance is assumed to be zero.")
        if sigma > 0 and not subtract_noise:
            st.caption("Detector noise inflates the estimated variance, biasing "
                       "r, N and S upward. Tick the sidebar box to correct for it.")

# Tab 3 - Measurment types

with tab3:
    cols = st.columns(3)
    notes = {
        "joint": "Idealized: both quadratures exact.",
        "homodyne": "One quadrature per shot; points sit on the axes (other is missing).",
        "heterodyne": "Both quadratures, but +1 unit of vacuum noise: a fatter blob.",
    }
    for col, mt, comparison_key in zip(
            cols, ["joint", "homodyne", "heterodyne"], comparison_keys):
        s = simulate.sample_measurements(
            V_A, n_samples, sigma, mt, comparison_key)
        s_host = jax.device_get(jnp.nan_to_num(s))
        with col:
            fig, ax = plt.subplots(figsize=(4, 4))
            # homodyne: plot missing quadrature as 0 purely for display
            ax.scatter(s_host[:, 0], s_host[:, 1], s=5, alpha=0.3)
            draw_cov(ax, V_A, color="green", lw=2)
            lim = float(4 * jnp.sqrt(jnp.cosh(2 * r) + 1))
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=mt,
                   xlabel="X", ylabel="P")
            ax.grid(True)
            st.pyplot(fig)
            st.caption(notes[mt] + " (green = true state covariance)")