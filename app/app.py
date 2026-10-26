"""
Interactive demo: Hawking/Unruh radiation via Gaussian state tomography.

Run from the repo root with the project venv:
    .venv/Scripts/python -m streamlit run app/app.py      (Windows)
    .venv/bin/python -m streamlit run app/app.py          (macOS/Linux)
`python app/app.py` (e.g. VS Code's Run button) also works: it relaunches
itself under Streamlit.
"""
import io
import sys
from pathlib import Path

import streamlit as st
from streamlit import runtime

if __name__ == "__main__" and not runtime.exists():
    # launched as a plain script: hand off to `streamlit run` with this interpreter
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    sys.exit(stcli.main())

st.set_page_config(page_title="Horizon radiation tomography",
                   page_icon=":material/blur_on:", layout="wide")

# src/ uses flat imports (import simulate), so put it on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import numpy as np
    import jax.numpy as jnp
    from jax import random
    import matplotlib.pyplot as plt

    import simulate
    import estimate
    import metrics
except ModuleNotFoundError as e:
    # usual cause: `streamlit` on PATH belongs to a Python without the project deps
    st.error(
        f"Module `{e.name}` is not installed for the Python running this app "
        f"(`{sys.executable}`).\n\nRun it with the project venv instead:\n\n"
        "```\n.venv/Scripts/python -m streamlit run app/app.py\n```\n\n"
        "or install the requirements: `pip install -r requirements.txt`.",
        icon=":material/error:")
    st.stop()

MEASUREMENT_TYPES = ["joint", "heterodyne"]
COLORS = {"joint": "tab:blue", "heterodyne": "tab:green"}


# cached simulation (plain numpy out, so results pickle cleanly into the cache)

@st.cache_data(max_entries=64)
def simulate_cloud(r, n_samples, sigma, measurement_type, seed):
    V_A = simulate.reduce_to_accessible(simulate.generate_covariance_matrix(r))
    key = random.fold_in(random.key(seed), MEASUREMENT_TYPES.index(measurement_type))
    return np.asarray(simulate.sample_measurements(V_A, n_samples, sigma, measurement_type, key))


@st.cache_data(max_entries=16)
def simulate_joint_access(r, n_samples, seed):
    V = simulate.generate_covariance_matrix(r)
    return np.asarray(simulate.sample_idealized_joint_access(V, n_samples, 0.0, random.key(seed)))


@st.cache_data(max_entries=16)
def accuracy_sweep(r, sigma, n_grid, n_trials, seed):
    """RMSE and bias of r_hat over n_trials independent datasets per (scheme, N)."""
    V_A = simulate.reduce_to_accessible(simulate.generate_covariance_matrix(r))
    rows = []
    for measurement_type in MEASUREMENT_TYPES:
        for n in n_grid:
            errs = []
            for trial in range(n_trials):
                key = random.fold_in(random.key(seed), trial)
                samples = simulate.sample_measurements(V_A, n, sigma, measurement_type, key)
                V_hat = estimate.estimate_covariance(samples, measurement_type, sigma)
                errs.append(float(metrics.estimate_squeezing(V_hat)) - r)
            errs = np.array(errs)
            rows.append({
                "measurement_type": measurement_type, "n": n,
                "rmse": float(np.sqrt(np.mean(errs ** 2))),
                "bias": float(np.mean(errs)),
                "crb": float(metrics.squeezing_crb(r, n, sigma, measurement_type)),
            })
    return rows


def to_csv(samples, r, sigma, measurement_type, seed):
    """Point cloud as CSV, with the simulation parameters in the header."""
    buf = io.StringIO()
    header = (f"r={r:.6f}, noise_level={sigma}, measurement_type={measurement_type}, "
              f"seed={seed}, n_samples={len(samples)}, V_A=cosh(2r)*I2={np.cosh(2 * r):.6f}*I2\n"
              "x,p")
    np.savetxt(buf, samples, delimiter=",", header=header, comments="# ", fmt="%.8g")
    return buf.getvalue()


# plotting helpers

def ellipse(cov, n_sigma=2.0, n_pts=200):
    """Points on the n-sigma contour of a zero-mean Gaussian with covariance cov."""
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)])
    return (vecs @ (n_sigma * np.sqrt(np.clip(vals, 0, None))[:, None] * circle)).T


def draw_cov(ax, cov, **kw):
    e = ellipse(cov)
    ax.plot(e[:, 0], e[:, 1], **kw)


def show(fig):
    # close after rendering: pyplot otherwise keeps every figure alive across reruns
    st.pyplot(fig)
    plt.close(fig)


# sidebar

if "seed" not in st.session_state:
    st.session_state.seed = 42

with st.sidebar:
    st.header("Parameters")
    source = st.segmented_control(
        "Set the state via", ["Squeezing r", "Unruh Ω/a"], default="Squeezing r",
        required=True,
        help="Unruh mode derives r from the Rindler frequency over acceleration "
             "(units ħ = c = k_B = 1): sinh²r = 1/(exp(2πΩ/a) − 1), paper eq. 2.6.2.")
    if source == "Squeezing r":
        r = st.slider("Squeezing r", 0.05, 2.0, 0.5, 0.05)
    else:
        omega_over_a = st.slider("Ω / a", 0.01, 1.0, 0.1, 0.01,
                                 help="Smaller Ω/a = hotter Unruh bath = stronger squeezing.")
        r = float(simulate.r_from_unruh(omega_over_a, 1.0))
        st.caption(f"n_Ω = {float(simulate.unruh_occupation(omega_over_a, 1.0)):.4f}, "
                   f"Ω/T_U = {2 * np.pi * omega_over_a:.3f} → r = {r:.4f}")
    n_samples = st.select_slider(
        "Samples N", options=[50, 100, 500, 1000, 5000, 10000], value=1000)
    sigma = st.select_slider(
        "Detector noise σ", options=[0.0, 0.05, 0.1, 0.2, 0.5], value=0.1)
    measurement_type = st.segmented_control(
        "Measurement type", MEASUREMENT_TYPES, default="joint", required=True)
    subtract_noise = st.toggle("Subtract known detector noise σ²", value=True)

    if st.button("Resample", icon=":material/refresh:", help="New random seed"):
        st.session_state.seed += 1
    seed = st.session_state.seed
    st.caption(f"seed = {seed}")

r = float(r)
V_A = np.cosh(2 * r) * np.eye(2)

st.title("Horizon radiation tomography")
st.markdown(
    "A horizon is modeled as a two-mode squeezed state: one mode the outside "
    "observer can measure, one hidden behind the horizon. Tracing out the hidden "
    "mode leaves the observer a **thermal** state. How much can they recover "
    "from finite, noisy data?"
)

tab_thermal, tab_recon, tab_types, tab_sweep = st.tabs(
    [":material/blur_on: Why is it thermal?", ":material/scatter_plot: Live reconstruction",
     ":material/compare: Measurement types", ":material/query_stats: Accuracy vs N"],
    on_change="rerun")

# Tab 1 - Why is it thermal?

if tab_thermal.open:
    with tab_thermal:
        joint = simulate_joint_access(r, 2000, seed)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(joint[:, 0], joint[:, 2], s=6, alpha=0.4)
            ax.set(title="Both modes: X_A vs X_B (correlated)",
                   xlabel="X_A (accessible)", ylabel="X_B (hidden)")
            ax.axis("equal"); ax.grid(True)
            show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(joint[:, 0], joint[:, 1], s=6, alpha=0.4, color="tab:orange")
            ax.set(title="Accessible mode only: X_A vs P_A (thermal blob)",
                   xlabel="X_A", ylabel="P_A")
            ax.axis("equal"); ax.grid(True)
            show(fig)
        st.info(
            f"The joint state is a pure, strongly correlated pair. Ignoring the hidden "
            f"mode, the observer sees a round thermal state with variance cosh(2r) = "
            f"{np.cosh(2 * r):.2f} in each quadrature, and mean particle number "
            f"N = sinh²(r) = {np.sinh(r) ** 2:.3f}.",
            icon=":material/info:")

# Tab 2 - Live reconstruction

if tab_recon.open:
    with tab_recon:
        samples = simulate_cloud(r, n_samples, sigma, measurement_type, seed)
        V_hat = np.asarray(estimate.estimate_covariance(
            samples, measurement_type, sigma if subtract_noise else 0.0))

        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.3, label="measured samples")
            draw_cov(ax, V_A, color="green", lw=2, label="true covariance (2σ)")
            draw_cov(ax, V_hat, color="red", lw=2, ls="--", label="estimated covariance (2σ)")
            ax.set(xlabel="X quadrature", ylabel="P quadrature",
                   title=f"{measurement_type}, N={n_samples}, σ={sigma}")
            ax.axis("equal"); ax.grid(True); ax.legend(loc="upper right")
            show(fig)
        with c2:
            r_hat = float(metrics.estimate_squeezing(V_hat))
            rows = {
                "r": (r, r_hat),
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
            crb = float(metrics.squeezing_crb(r, n_samples, sigma, measurement_type))
            m1, m2 = st.columns(2)
            m1.metric("Covariance error (Frobenius)",
                      f"{float(metrics.matrix_error(V_A, V_hat)):.4f}", border=True)
            m2.metric("Cramér–Rao bound on std(r̂)", f"{crb:.4f}", border=True,
                      help="Best achievable standard deviation of any unbiased estimate "
                           "of r from this many shots of this scheme.")
            if measurement_type == "joint":
                st.caption("Joint is the idealized benchmark: exact x and p in the same "
                           "shot is not physically possible.")
            if measurement_type == "heterodyne":
                st.caption("The known +1 unit of heterodyne vacuum noise is subtracted.")
            if sigma > 0 and not subtract_noise:
                st.caption("Detector noise inflates the estimated variance, biasing "
                           "r, N and S upward. Turn on noise subtraction to correct for it.")
            st.download_button(
                "Download point cloud (CSV)",
                to_csv(samples, r, sigma, measurement_type, seed),
                file_name=f"cloud_{measurement_type}_r{r:.3f}_N{n_samples}_s{sigma}_seed{seed}.csv",
                mime="text/csv", icon=":material/download:")

# Tab 3 - Measurement types

if tab_types.open:
    with tab_types:
        notes = {
            "joint": "Idealized benchmark: both quadratures exact.",
            "heterodyne": "Both quadratures, but +1 unit of vacuum noise: a fatter blob.",
        }
        lim = 4 * np.sqrt(np.cosh(2 * r) + 1 + sigma ** 2)
        for col, mt in zip(st.columns(len(MEASUREMENT_TYPES)), MEASUREMENT_TYPES):
            s = simulate_cloud(r, n_samples, sigma, mt, seed)
            with col:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(s[:, 0], s[:, 1], s=5, alpha=0.3, color=COLORS[mt])
                draw_cov(ax, V_A, color="green", lw=2)
                ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=mt, xlabel="X", ylabel="P")
                ax.set_aspect("equal"); ax.grid(True)
                show(fig)
                st.caption(notes[mt] + " (green = true state covariance)")

# Tab 4 - Accuracy vs N

if tab_sweep.open:
    with tab_sweep:
        st.markdown(
            "Root-mean-square error of r̂ over repeated simulated experiments, per "
            "scheme, against the Cramér–Rao bound (dashed). Uses the current r and σ, "
            "with known noise subtracted.")
        n_trials = st.select_slider("Trials per point", options=[10, 20, 50, 100], value=20)
        n_grid = (50, 100, 300, 1000, 3000, 10000)
        with st.spinner("Simulating…"):
            sweep = accuracy_sweep(r, sigma, n_grid, n_trials, seed)

        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for mt in MEASUREMENT_TYPES:
                pts = [row for row in sweep if row["measurement_type"] == mt]
                ns = [p["n"] for p in pts]
                ax.loglog(ns, [p["rmse"] for p in pts], "o-", color=COLORS[mt], label=mt)
                ax.loglog(ns, [p["crb"] for p in pts], "--", color=COLORS[mt], alpha=0.7)
            ax.set(xlabel="Samples N", ylabel="RMSE of r̂", title=f"r = {r:.3f}, σ = {sigma}")
            ax.grid(True, which="both", alpha=0.3); ax.legend()
            show(fig)
        with c2:
            st.dataframe(
                [{"scheme": row["measurement_type"], "N": row["n"],
                  "RMSE": row["rmse"], "bias": row["bias"], "CRB": row["crb"],
                  "RMSE / CRB": row["rmse"] / row["crb"]} for row in sweep],
                hide_index=True,
                column_config={k: st.column_config.NumberColumn(format="%.4f")
                               for k in ["RMSE", "bias", "CRB", "RMSE / CRB"]})
        st.caption(
            "RMSE/CRB near 1 means the covariance estimator is extracting all the information "
            "in the data. At small N or small r the estimate is clipped at r ≥ 0, which "
            "biases it and can push RMSE below the (unbiased) bound.")
