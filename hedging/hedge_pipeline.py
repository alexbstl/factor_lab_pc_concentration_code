"""
hedge_pipeline.py
=================
A small, explicit pipeline for the hedge memo's running example:

    model  ->  draw a return panel  ->  estimate  ->  pick any portfolio x
           ->  true vs estimated variance of x
           ->  population frame (b̄) vs estimated frame (h), compared by angle

:class:`HedgeRiskModel` bundles one population model with one estimate of it,
so any portfolio can be plugged in::

    hrm = HedgeRiskModel.memo(p=10, n=20, seed=1)
    hrm.report(x)                 # true vs estimated variance and vol of x
    hrm.report(hrm.hedge(x))      # the same, after the block hedge on span(H)
    hrm.show_frames()             # b̄ vs h angles (+ the factor-plane disk at k=2)

Three sets of directions appear, and only the last two are orthonormal:

    β columns of B   the economic loadings (market, style, ...)   NOT orthogonal
    b̄_j              eigenvectors of Σ0 = B Σf B'                 orthonormal (the target)
    h_j              eigenvectors of the sample S = YY'/(np)       orthonormal (the estimate)

Units. Σf and δ² are annualized. Each of the n observations is one trading
day, drawn with covariance (annual / DAYS_PER_YEAR). Estimated variances are
annualized back by the same factor, so true and estimated numbers are directly
comparable. Use :func:`annual_to_window_vol` to quote any vol over w days.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# This repo's fl_plot (used by plot_plane) must win over the sibling factor_lab's.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DAYS_PER_YEAR = 252


# ── Units ────────────────────────────────────────────────────────────────────


def annual_to_window_var(var_annual, window, days_per_year=DAYS_PER_YEAR):
    """Variance over ``window`` days: variances add across independent days."""
    return np.asarray(var_annual) * (window / days_per_year)


def annual_to_window_vol(vol_annual, window, days_per_year=DAYS_PER_YEAR):
    """Vol over ``window`` days: σ_annual · sqrt(w/252). ``window=1`` is daily."""
    return np.asarray(vol_annual) * np.sqrt(window / days_per_year)


# ── Population model ─────────────────────────────────────────────────────────


@dataclass
class Model:
    """y = B f + z with Cov(f) = Sigma_f and Cov(z) = diag(delta2), annualized."""

    B: np.ndarray            # (p, k) loadings; columns need not be orthogonal
    Sigma_f: np.ndarray      # (k, k) factor covariance
    delta2: np.ndarray       # (p,)   idiosyncratic variances
    names: tuple = ()        # optional factor names for the β columns

    def __post_init__(self):
        self.B = np.asarray(self.B, float)
        self.Sigma_f = np.asarray(self.Sigma_f, float)
        self.delta2 = np.asarray(self.delta2, float)
        if not self.names:
            self.names = tuple(f"factor {j + 1}" for j in range(self.k))

    @property
    def p(self) -> int:
        return self.B.shape[0]

    @property
    def k(self) -> int:
        return self.B.shape[1]

    @property
    def Sigma0(self) -> np.ndarray:
        return self.B @ self.Sigma_f @ self.B.T

    @property
    def Sigma_y(self) -> np.ndarray:
        return self.Sigma0 + np.diag(self.delta2)

    def frame(self):
        """Population eigenframe of Σ0: ``(p_mu, b_bar)`` with b_bar (p, k).

        Sign gauge: each b̄_j points along its own β column (⟨b̄_j, β_j⟩ ≥ 0),
        so b̄_1 reads as "mostly market" when it is.
        """
        evals, evecs = np.linalg.eigh(self.Sigma0)
        order = np.argsort(evals)[::-1][: self.k]
        p_mu, b_bar = evals[order], evecs[:, order]
        s = np.sign(np.einsum("pj,pj->j", b_bar, self.B))
        return p_mu, b_bar * np.where(s == 0, 1.0, s)


MEMO_BETA = np.array([[0.880,  0.164], [0.801, -0.740], [0.963, -0.575], [1.063,  0.960],
                      [1.170,  0.122], [1.017, -1.039], [0.917, -0.050], [0.882, -0.698],
                      [1.112, -0.378], [1.245, -0.293]])
MEMO_DELTA2 = np.array([0.0968, 0.0647, 0.1413, 0.0453, 0.1128,
                        0.1399, 0.0603, 0.1437, 0.1402, 0.0415])
MEMO_FACTOR_VOLS = (0.16, 0.10)


def memo_model(p: int = 10, factor_vols=MEMO_FACTOR_VOLS) -> Model:
    """The memo's §1.4 population model, at any cross-section size ``p``.

    p = 10 is exactly the memo's table. Otherwise its 10 rows of (β, δ²) are
    truncated (p < 10) or tiled (p > 10): asset i gets row i mod 10.
    """
    rows = np.arange(p) % len(MEMO_BETA)
    return Model(B=MEMO_BETA[rows], Sigma_f=np.diag(np.square(factor_vols)),
                 delta2=MEMO_DELTA2[rows], names=("market", "style"))


# ── Return samplers ──────────────────────────────────────────────────────────
# A sampler is ``f(rng, shape) -> array`` of i.i.d. draws with mean 0 and
# variance 1. draw_returns applies the model's covariances on top, so Σf and δ²
# hold exactly whatever the distribution; the sampler only sets the shape (tails).


def gaussian(rng: np.random.Generator, shape) -> np.ndarray:
    """Standard normal draws."""
    return rng.standard_normal(shape)


def student_t(df: float):
    """Student-t draws with ``df`` > 2 degrees of freedom, rescaled to unit variance."""
    if df <= 2:
        raise ValueError("student_t needs df > 2 for a finite variance")
    scale = np.sqrt((df - 2) / df)

    def sample(rng: np.random.Generator, shape) -> np.ndarray:
        return rng.standard_t(df, shape) * scale

    sample.__name__ = f"student_t(df={df:g})"
    return sample


def draw_returns(model: Model, n: int, seed: int | None = None,
                 factor_sampler=gaussian, idio_sampler=gaussian) -> np.ndarray:
    """A (p, n) panel of daily returns with per-day covariance annual/252.

    Factor returns are L_f·e_f and idiosyncratic returns sd_z·e_z, where e_f and
    e_z come from ``factor_sampler`` and ``idio_sampler`` (unit-variance, i.i.d.).
    """
    rng = np.random.default_rng(seed)
    L_f = np.linalg.cholesky(annual_to_window_var(model.Sigma_f, 1))
    sd_z = np.sqrt(annual_to_window_var(model.delta2, 1))
    F = L_f @ factor_sampler(rng, (model.k, n))
    Z = sd_z[:, None] * idio_sampler(rng, (model.p, n))
    return model.B @ F + Z


# ── Estimation ───────────────────────────────────────────────────────────────


@dataclass
class Estimate:
    H: np.ndarray            # (p, k) top-k sample eigenvectors, orthonormal
    theta: np.ndarray        # (k,)   top-k eigenvalues of S = YY'/(np), daily scale
    ell: float               # mean of the remaining nonzero eigenvalues of S
    delta2_hat: np.ndarray   # (p,)   residual variances, annualized
    n: int

    @property
    def p(self) -> int:
        return self.H.shape[0]

    @property
    def p_theta_annual(self) -> np.ndarray:
        """p·θ_j annualized: the estimate's counterpart of the population p·μ_j."""
        return self.p * self.theta * DAYS_PER_YEAR


def estimate(Y: np.ndarray, k: int, center: bool = True) -> Estimate:
    """PCA of the doubly normalized sample covariance S = YY'/(np) (memo eq. 1.2).

    δ̂²_i is the variance of asset i's return after projecting out span(H),
    annualized. ``ell`` averages the nonzero eigenvalues beyond the k-th.
    """
    if center:
        Y = Y - Y.mean(axis=1, keepdims=True)
    p, n = Y.shape
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    eig_S = s**2 / (n * p)
    rank = min(p, n - 1 if center else n)
    H = U[:, :k]
    resid = Y - H @ (H.T @ Y)
    delta2_hat = (resid**2).mean(axis=1) * DAYS_PER_YEAR
    return Estimate(H=H, theta=eig_S[:k], ell=float(eig_S[k:rank].mean()),
                    delta2_hat=delta2_hat, n=n)


def gauge_to(est: Estimate, b_bar: np.ndarray) -> Estimate:
    """Flip each h_j so that ⟨b̄_j, h_j⟩ ≥ 0. Changes no angle between subspaces.

    When more directions are estimated than there are true factors, the extra
    h_j have no partner and are flipped to have a nonnegative coordinate sum.
    """
    m = min(b_bar.shape[1], est.H.shape[1])
    s = np.sign(est.H.sum(axis=0))
    s[:m] = np.sign(np.einsum("pj,pj->j", b_bar[:, :m], est.H[:, :m]))
    est.H = est.H * np.where(s == 0, 1.0, s)
    return est


# ── Portfolio variance: true vs estimated ────────────────────────────────────


def portfolio_variance(model: Model, est: Estimate, x: np.ndarray) -> pd.DataFrame:
    """Per-factor and idiosyncratic variance of book ``x``, true vs plug-in.

    True (memo eq. 1.1):      Σ_j p·μ_j (x·b̄_j)²  +  Σ_i x_i² δ_i²
    Estimated (memo eq. 1.3): Σ_j p·θ_j (x·h_j)²  +  Σ_i x_i² δ̂_i²
    All annualized. ``x`` is in dollars.

    The true and estimated factor counts may differ. A factor one side lacks
    contributes zero variance on that side; its eigenvalue and exposure are blank.
    """
    x = np.asarray(x, float)
    p_mu, b_bar = model.frame()
    k_true, k_est = model.k, est.H.shape[1]
    rows = []
    for j in range(max(k_true, k_est)):
        row = {"component": f"factor {j + 1}", "variance true": 0.0, "variance est": 0.0}
        if j < k_true:
            row |= {"eigenvalue true (pμ)": p_mu[j], "exposure true (x·b̄)": x @ b_bar[:, j],
                    "variance true": p_mu[j] * (x @ b_bar[:, j])**2}
        if j < k_est:
            row |= {"eigenvalue est (pθ)": est.p_theta_annual[j], "exposure est (x·h)": x @ est.H[:, j],
                    "variance est": est.p_theta_annual[j] * (x @ est.H[:, j])**2}
        rows.append(row)
    rows.append({"component": "idiosyncratic",
                 "variance true": float(x**2 @ model.delta2),
                 "variance est": float(x**2 @ est.delta2_hat)})
    cols = ["component", "eigenvalue true (pμ)", "eigenvalue est (pθ)", "exposure true (x·b̄)",
            "exposure est (x·h)", "variance true", "variance est"]
    df = pd.DataFrame(rows).reindex(columns=cols).set_index("component")
    df.loc["total"] = df[["variance true", "variance est"]].sum()
    df["share true"] = df["variance true"] / df.loc["total", "variance true"]
    df["share est"] = df["variance est"] / df.loc["total", "variance est"]
    df["error (est − true)"] = df["variance est"] - df["variance true"]
    return df


def portfolio_vol(df: pd.DataFrame, windows=(DAYS_PER_YEAR, 1)) -> pd.DataFrame:
    """Total / factor / idio vol from :func:`portfolio_variance`, over each window."""
    var = pd.DataFrame({
        "true": [df.loc["total", "variance true"],
                 df.filter(like="factor", axis=0)["variance true"].sum(),
                 df.loc["idiosyncratic", "variance true"]],
        "est": [df.loc["total", "variance est"],
                df.filter(like="factor", axis=0)["variance est"].sum(),
                df.loc["idiosyncratic", "variance est"]],
    }, index=["total", "factor", "idiosyncratic"])
    cols = {}
    for w in windows:
        for c in var:
            cols[(_window_label(w), c)] = np.sqrt(annual_to_window_var(var[c], w))
    return pd.DataFrame(cols, index=var.index)


# ── Frames and angles ────────────────────────────────────────────────────────


def _deg(cos):
    return np.degrees(np.arccos(np.clip(np.abs(cos), 0.0, 1.0)))


def loading_geometry(model: Model) -> pd.DataFrame:
    """Cosines among the normalized β columns, and of each β column with each b̄_j.

    The β block is generally not the identity; the b̄ block is, by construction.
    """
    _, b_bar = model.frame()
    beta = model.B / np.linalg.norm(model.B, axis=0)
    cols = list(model.names) + [f"b̄{j + 1}" for j in range(model.k)]
    V = np.c_[beta, b_bar]
    return pd.DataFrame(V.T @ V, index=cols, columns=cols)


def compare_frames(model: Model, est: Estimate) -> dict:
    """Angles between the population frame b̄ and the estimated frame h.

    - ``cross``: M = b̄'H, the cosines ⟨b̄_i, h_j⟩ (k_true × k_est).
    - ``per_factor``: for each h_j, its angle to span(b̄) (= span(B)) and, when
      it has a partner b̄_j (j ≤ min(k_true, k_est)), its angle to b̄_j and the
      split of sin²∠(h_j, b̄_j) into an out-of-subspace part ‖Π⊥h_j‖² and an
      in-plane rotation part. Unpartnered rows are blank in those columns.
    - ``principal``: the min(k_true, k_est) principal angles between span(H)
      and span(b̄).
    """
    _, b_bar = model.frame()
    k_true, k_est = model.k, est.H.shape[1]
    M = b_bar.T @ est.H
    in_span = (M**2).sum(axis=0)                       # ‖Π_B h_j‖²
    oos = 1.0 - in_span                                # sin²∠(h_j, span B)
    partner = np.full(k_est, np.nan)
    m = min(k_true, k_est)
    partner[:m] = np.diag(M[:m, :m])                   # cos∠(h_j, b̄_j)
    pair = 1.0 - partner**2                            # sin²∠(h_j, b̄_j)
    per = pd.DataFrame({
        "angle to b̄_j (deg)": np.where(np.isnan(partner), np.nan, _deg(np.nan_to_num(partner))),
        "angle to span(B) (deg)": _deg(np.sqrt(np.clip(in_span, 0, 1))),
        "sin² to b̄_j": pair,
        "  = out-of-subspace": oos,
        "  + in-plane rotation": pair - oos,
    }, index=[f"h{j + 1}" for j in range(k_est)])
    sv = np.linalg.svd(M, compute_uv=False)
    principal = pd.Series(_deg(sv), index=[f"θ{j + 1}" for j in range(len(sv))],
                          name="principal angles (deg)")
    cross = pd.DataFrame(M, index=[f"b̄{i + 1}" for i in range(k_true)],
                         columns=[f"h{j + 1}" for j in range(k_est)])
    return {"cross": cross, "per_factor": per, "principal": principal}


def _window_label(w: int) -> str:
    return "annual" if w == DAYS_PER_YEAR else ("daily" if w == 1 else f"{w}-day")


def _dollars(v: float) -> str:
    return f"-${-v:,.0f}" if v < 0 else f"${v:,.0f}"


def _as_books(portfolios) -> dict:
    """``{label: x}`` from a dict, or from a list of books labelled 1, 2, ..."""
    if isinstance(portfolios, dict):
        return dict(portfolios)
    return {f"portfolio {i + 1}": x for i, x in enumerate(portfolios)}


# ── Portfolio construction ───────────────────────────────────────────────────


def min_variance(Sigma: np.ndarray, notional: float = 1.0) -> np.ndarray:
    """Fully invested minimum-variance book, shorts allowed: x ∝ Σ⁻¹1, 1'x = notional."""
    w = np.linalg.solve(Sigma, np.ones(len(Sigma)))
    return notional * w / w.sum()


def min_variance_long_only(Sigma: np.ndarray, notional: float = 1.0, method: str = "nnls",
                           tol: float = 1e-14, max_iter: int | None = None) -> np.ndarray:
    """Fully invested, long-only minimum-variance book via a standard solver.

    Minimizes w'Σw subject to 1'w = 1 and w ≥ 0. Σ is first rescaled to unit
    average variance for conditioning. :func:`kkt_gap` checks the result.

    - ``method="nnls"`` (default, exact): the minimizer of v'Σv − 2·1'v over v ≥ 0
      is proportional to the long-only min-var book (same KKT conditions). With
      Σ = LL' that is the nonnegative least-squares problem min ‖L'v − L⁻¹1‖,
      solved by SciPy's Lawson–Hanson active-set ``nnls``, which terminates
      finitely at the exact optimum. Then w = v / 1'v.
    - ``method="slsqp"``: SciPy SLSQP from equal weights. Fine on well-conditioned
      problems, but it can stop well short of the optimum on badly conditioned ones.
    """
    from scipy.linalg import cholesky, solve_triangular
    from scipy.optimize import minimize, nnls

    p = len(Sigma)
    S = np.asarray(Sigma, float) / (np.trace(Sigma) / p)
    if method == "nnls":
        L = cholesky(S, lower=True)
        v, _ = nnls(L.T, solve_triangular(L, np.ones(p), lower=True),
                    maxiter=max_iter or 50 * p)
        w = v
    elif method == "slsqp":
        res = minimize(lambda w: w @ S @ w, np.full(p, 1.0 / p), jac=lambda w: 2.0 * S @ w,
                       method="SLSQP", bounds=[(0.0, None)] * p,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                                     "jac": lambda w: np.ones(p)}],
                       options={"ftol": tol, "maxiter": max_iter or 1_000})
        if not res.success:
            raise RuntimeError(f"SLSQP failed: {res.message}")
        w = np.clip(res.x, 0.0, None)
    else:
        raise ValueError("method must be 'nnls' or 'slsqp'")
    if w.sum() <= 0:
        raise RuntimeError("long-only min-variance solver returned an empty book")
    return notional * w / w.sum()


def ffp_min_variance_long_only(B: np.ndarray, fact_vars: np.ndarray, spec_vars: np.ndarray,
                               notional: float = 1.0, tol: float = 1e-7, max_iter: int = 1_000,
                               kkt_tol: float = 1e-8, fallback: str = "solver",
                               return_info: bool = False):
    """Long-only minimum-variance book by the FFP fixed-point iteration.

    For a factor covariance Σ = B diag(fact_vars) B' + diag(spec_vars), with B (p, k)
    holding unit-length loadings, the fully invested long-only optimum has the form

        w ∝ max(1 − Bθ, 0) / spec_vars

    for a k-vector θ. Writing D = 1{Bθ ≤ 1} / spec_vars for the names held, θ solves

        (diag(1/fact_vars) + B'DB) θ = B'D1,

    and FFP iterates that map from θ = 1 until it stops moving (sup-norm < ``tol``).
    Each step is one k×k solve, O(pk²), with no p×p matrix anywhere.

    **FFP is not guaranteed to converge: it can cycle.** Each step depends on θ
    only through the held set {i : b_i'θ ≤ 1}, so a held set that recurs before θ
    has converged means the sequence is periodic from there on. That is detected
    exactly (the same set on consecutive steps is convergence, not a cycle: the
    next step lands on the fixed point). On a cycle, on hitting ``max_iter``, or if the result fails the KKT
    check (:func:`kkt_gap` > ``kkt_tol``), ``fallback="solver"`` (default) warns and
    returns :func:`min_variance_long_only` (exact NNLS) on the assembled Σ, and
    ``fallback="raise"`` raises instead.

    ``return_info=True`` returns ``(x, info)``, where ``info`` has ``method``
    ("ffp" or "solver"), ``iterations``, ``reason`` (why it fell back, else
    None), ``cycle_length`` and ``kkt_gap``.

    Ported from markowitz-empirics/old/main_multifactor.py (``ffp_minvar`` / ``psi``),
    with the same start and tolerance.
    """
    import warnings

    if fallback not in ("solver", "raise"):
        raise ValueError("fallback must be 'solver' or 'raise'")
    B = np.asarray(B, float)
    f = np.asarray(fact_vars, float)
    s = np.asarray(spec_vars, float)

    def held(th):
        return B @ th <= 1.0

    def psi(th):
        d = held(th) / s
        A = np.diag(1.0 / f) + (d * B.T) @ B
        return np.linalg.solve(A, B.T @ d)

    def sigma():
        return (B * f) @ B.T + np.diag(s)

    info = {"method": "ffp", "iterations": 0, "reason": None, "cycle_length": None,
            "kkt_gap": None}
    th_old, th_new = np.zeros(len(f)), np.ones(len(f))
    seen = {}                                     # held set -> iteration first seen
    for it in range(max_iter):
        if np.max(np.abs(th_new - th_old)) <= tol:
            break
        key = np.packbits(held(th_new)).tobytes()
        # Same held set as the previous step: the next step lands exactly on the
        # fixed point, so that is convergence. Recurring after >= 2 steps is a cycle.
        if key in seen and it - seen[key] >= 2:
            info["reason"], info["cycle_length"] = "cycle", it - seen[key]
            break
        seen.setdefault(key, it)
        th_old, th_new = th_new, psi(th_new)
        info["iterations"] = it + 1
    else:
        info["reason"] = f"no convergence in {max_iter} iterations"

    if info["reason"] is None:
        w = np.maximum(1.0 - B @ th_new, 0.0) / s
        if w.sum() <= 0:
            info["reason"] = "empty book"
        else:
            x = notional * w / w.sum()
            info["kkt_gap"] = kkt_gap(sigma(), x)
            if info["kkt_gap"] > kkt_tol:
                info["reason"] = f"KKT gap {info['kkt_gap']:.1e} > {kkt_tol:.0e}"

    if info["reason"] is not None:
        msg = f"FFP failed ({info['reason']})"
        if fallback == "raise":
            raise RuntimeError(msg)
        warnings.warn(msg + "; falling back to the standard solver", RuntimeWarning, stacklevel=2)
        Sigma = sigma()
        x = min_variance_long_only(Sigma, notional)
        info.update(method="solver", kkt_gap=kkt_gap(Sigma, x))
    return (x, info) if return_info else x


def kkt_gap(Sigma: np.ndarray, x: np.ndarray, support_tol: float = 1e-9) -> float:
    """Worst violation of the long-only min-variance optimality conditions.

    At the optimum, the marginal variance (Σw)_i equals a common λ on every held
    name and is ≥ λ on every name at zero. Returns the largest violation relative
    to λ (0 at an exact optimum).
    """
    w = np.asarray(x, float) / np.sum(x)
    g = Sigma @ w
    held = w > support_tol * w.max()
    lam = g[held].mean()
    viol = np.r_[np.abs(g[held] - lam), np.clip(lam - g[~held], 0.0, None)]
    return float(viol.max() / lam)


# ── The bundle: one model, one estimate, any portfolio ───────────────────────


class HedgeRiskModel:
    """A population model plus one estimate of it; plug in any dollar portfolio.

    Give either a return panel ``Y`` (p, n) of daily returns, or ``n`` and
    ``seed`` to draw one from ``model`` (Gaussian by default; pass
    ``factor_sampler`` / ``idio_sampler``, e.g. ``student_t(4)``, to change the
    distribution). The estimate is sign-gauged so each
    h_j points into its own b̄_j.

    ``k`` is the number of directions the risk model estimates; it defaults to
    the true factor count ``model.k`` (the oracle) but may be set lower or higher.
    """

    def __init__(self, model: Model, Y: np.ndarray | None = None, *,
                 n: int | None = None, seed: int | None = None,
                 k: int | None = None, factor_sampler=gaussian, idio_sampler=gaussian,
                 center: bool = True):
        if Y is None:
            if n is None:
                raise ValueError("give either a return panel Y or n (and seed) to draw one")
            Y = draw_returns(model, n, seed, factor_sampler, idio_sampler)
            self.samplers = (factor_sampler.__name__, idio_sampler.__name__)
        else:
            self.samplers = ("supplied Y", "supplied Y")
        Y = np.asarray(Y, float)
        if Y.shape[0] != model.p:
            raise ValueError(f"Y has {Y.shape[0]} rows, model has p = {model.p}")
        k = model.k if k is None else int(k)
        rank = min(Y.shape[0], Y.shape[1] - 1 if center else Y.shape[1])
        if not 1 <= k < rank:
            raise ValueError(f"k = {k} must satisfy 1 <= k < rank(Y) = {rank}")
        self.model, self.Y = model, Y
        self.p_mu, self.b_bar = model.frame()
        self.est = gauge_to(estimate(Y, k, center), self.b_bar)

    @classmethod
    def memo(cls, p: int = 10, n: int = 20, seed: int | None = None, k: int | None = None,
             factor_vols=MEMO_FACTOR_VOLS, factor_sampler=gaussian,
             idio_sampler=gaussian) -> "HedgeRiskModel":
        """The memo's §1.4 model at cross-section ``p``, estimated from ``n`` days."""
        return cls(memo_model(p, factor_vols), n=n, seed=seed, k=k,
                   factor_sampler=factor_sampler, idio_sampler=idio_sampler)

    @property
    def p(self) -> int:
        return self.model.p

    @property
    def k(self) -> int:
        """Number of directions this risk model estimates (may differ from model.k)."""
        return self.est.H.shape[1]

    @property
    def H(self) -> np.ndarray:
        return self.est.H

    def factor_form(self, true: bool = False):
        """``(B, fact_vars, spec_vars)`` with Σ = B diag(fact_vars) B' + diag(spec_vars).

        Default: the risk model's (H, p·θ annualized, δ̂²), so Σ = :attr:`Sigma_hat`.
        ``true=True``: the population's (b̄, p·μ, δ²), so Σ = ``model.Sigma_y``.
        Unit-length loadings, as :func:`ffp_min_variance_long_only` expects.
        """
        if true:
            return self.b_bar, self.p_mu, self.model.delta2
        return self.est.H, self.est.p_theta_annual, self.est.delta2_hat

    @property
    def Sigma_hat(self) -> np.ndarray:
        """The risk model's covariance, annualized: H diag(p·θ) H' + diag(δ̂²).

        Its quadratic form x'Σ̂x is exactly the plug-in variance (memo eq. 1.3);
        the true counterpart is ``model.Sigma_y``.
        """
        e = self.est
        return (e.H * e.p_theta_annual) @ e.H.T + np.diag(e.delta2_hat)

    # ── portfolio ──

    def _check(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        if x.shape != (self.p,):
            raise ValueError(f"portfolio has shape {x.shape}, model has p = {self.p}")
        return x

    def _truth(self, truth: Model | None) -> Model:
        """The model the "true" side is scored under: ``truth`` if given, else the data's model."""
        if truth is None:
            return self.model
        if truth.p != self.p:
            raise ValueError(f"truth has p = {truth.p}, risk model has p = {self.p}")
        return truth

    def variance(self, x, truth: Model | None = None) -> pd.DataFrame:
        """True vs plug-in variance of ``x``, per factor and idiosyncratic (annualized $²).

        ``truth`` scores the true side under another :class:`Model` (same p, any k)
        instead of the one the data were drawn from.
        """
        return portfolio_variance(self._truth(truth), self.est, self._check(x))

    def vol(self, x, windows=(DAYS_PER_YEAR, 1), truth: Model | None = None) -> pd.DataFrame:
        """Total / factor / idio vol of ``x``, true vs estimated, over each window (days)."""
        return portfolio_vol(self.variance(x, truth), windows)

    def hedge(self, x, m: int | None = None) -> np.ndarray:
        """Book after hedging out the first ``m`` estimated directions (default all k).

        Returns (I − H_m H_m')x, which equals regressing the book on the m
        factor-mimicking portfolios and trading the opposite coefficients.
        """
        Hm = self.H if m is None else self.H[:, :m]
        x = self._check(x)
        return x - Hm @ (Hm.T @ x)

    def compare(self, portfolios, window: int = DAYS_PER_YEAR,
                truth: Model | None = None) -> pd.DataFrame:
        """One row per book: true vs estimated vol (total, factor, idio) over ``window`` days.

        ``portfolios`` is a dict ``{label: x}`` or a list of books (labelled 1, 2, ...).
        ``est / true`` is the ratio of estimated to true total vol: below 1 means the
        risk model understates the book's risk. ``truth`` as in :meth:`variance`.
        """
        books = _as_books(portfolios)
        rows = {}
        for label, x in books.items():
            x = self._check(x)
            v = portfolio_vol(self.variance(x, truth), (window,))
            v.columns = v.columns.droplevel(0)
            rows[label] = {"gross $": np.abs(x).sum(), "net $": x.sum(),
                           "total true": v.loc["total", "true"], "total est": v.loc["total", "est"],
                           "est / true": v.loc["total", "est"] / v.loc["total", "true"],
                           "factor true": v.loc["factor", "true"], "factor est": v.loc["factor", "est"],
                           "idio true": v.loc["idiosyncratic", "true"],
                           "idio est": v.loc["idiosyncratic", "est"]}
        return pd.DataFrame(rows).T.rename_axis(f"vol ({_window_label(window)})")

    def report(self, x, label: str = "portfolio", windows=(DAYS_PER_YEAR, 1, 20),
               detail: bool | None = None, truth: Model | None = None):
        """Print risk for one book or several.

        - One book ``x``: the full variance table and vol table; returns the variance table.
        - Several (a dict ``{label: x}`` or a list of books): one comparison row per
          book, from :meth:`compare`, for each window in ``windows``; returns the
          table for the first window. ``detail=True`` also prints each full report.

        ``truth`` scores every "true" column under another :class:`Model` (same p,
        any k) — e.g. a larger-p or differently specified model — while the
        estimated columns stay this risk model's.
        """
        truth = self._truth(truth)
        if truth is not self.model:
            print(f"(true side scored under a supplied model: p = {truth.p}, k = {truth.k}, "
                  f"factor vols {np.round(np.sqrt(np.diag(truth.Sigma_f)), 4)})")
        if isinstance(x, (dict, list, tuple)):
            books = _as_books(x)
            tables = [self.compare(books, w, truth) for w in windows]
            money = {c: _dollars for c in tables[0].columns if c != "est / true"}
            for t in tables:
                print(t.to_string(formatters=money | {"est / true": "{:.2f}".format}))
                print()
            if detail:
                for lab, xx in books.items():
                    self.report(xx, lab, windows, truth=None if truth is self.model else truth)
                    print()
            return tables[0]
        x = self._check(x)
        print(f"── {label}:  gross ${np.abs(x).sum():,.0f}, net ${x.sum():,.0f}")
        var = self.variance(x, truth)
        fmt = {c: "{:,.0f}".format for c in var.columns
               if c.startswith(("variance", "exposure", "error"))}
        fmt |= {c: "{:.4f}".format for c in var.columns if c.startswith("eigenvalue")}
        fmt |= {c: "{:.1%}".format for c in var.columns if c.startswith("share")}
        print(var.to_string(formatters=fmt, na_rep=""))
        print()
        print(portfolio_vol(var, windows).to_string(float_format="${:,.0f}".format))
        return var

    # ── model and frames ──

    def loadings(self) -> pd.DataFrame:
        """Cosines among the normalized β columns and the b̄_j."""
        return loading_geometry(self.model)

    def frames(self) -> dict:
        """b̄ vs h: cross-Gram, per-factor angles and split, principal angles."""
        return compare_frames(self.model, self.est)

    def summary(self, max_rows: int = 20) -> None:
        """Print the population model, its eigenframe and the estimate."""
        m, e = self.model, self.est
        print(f"p = {m.p}, n = {e.n}, k true = {m.k}, k estimated = {self.k};  factor vols {np.sqrt(np.diag(m.Sigma_f))};  "
              f"mean δ² = {m.delta2.mean():.4f}")
        print(f"returns: factors {self.samplers[0]}, idiosyncratic {self.samplers[1]}")
        print(f"p·μ  (true, Σ0)        = {np.round(self.p_mu, 4)}")
        print(f"p·θ  (est, annualized) = {np.round(e.p_theta_annual, 4)};   ℓ (daily) = {e.ell:.3e}\n")
        cols = ([f"β {nm}" for nm in m.names] + ["δ²", "δ̂²"]
                + [f"b̄{j + 1}" for j in range(m.k)] + [f"h{j + 1}" for j in range(self.k)])
        tbl = pd.DataFrame(np.c_[m.B, m.delta2, e.delta2_hat, self.b_bar, e.H], columns=cols,
                           index=pd.RangeIndex(1, m.p + 1, name="asset"))
        print(tbl.head(max_rows).round(4).to_string())
        if m.p > max_rows:
            print(f"... ({m.p - max_rows} more rows)")
        print("\ncosines among normalized β columns and b̄:")
        print(self.loadings().round(4).to_string())

    def show_frames(self, draw: bool = True) -> dict:
        """Print :meth:`frames`; when the true k is 2 also draw the factor-plane disk."""
        fr = self.frames()
        print("cross-Gram M = b̄'H  (cosines):"); print(fr["cross"].round(4).to_string())
        print("\nper factor:");                 print(fr["per_factor"].round(4).to_string())
        print("\nprincipal angles between span(H) and span(B) (deg):",
              fr["principal"].round(2).to_dict())
        if draw and self.model.k == 2:
            import matplotlib.pyplot as plt
            fig = self.plot_plane(fr)
            try:                                  # in a notebook: render inline as a PNG,
                from io import BytesIO            # independent of the matplotlib backend
                from IPython import get_ipython
                from IPython.display import Image, display
                if get_ipython() is None:
                    raise ImportError
                buf = BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                display(Image(data=buf.getvalue()))
                plt.close(fig)
            except ImportError:
                plt.show()
        return fr

    def plot_plane(self, fr: dict | None = None):
        """The true factor plane in the orthonormal (b̄1, b̄2) basis (true k = 2 only).

        Gray dashed: normalized β columns (unit length, generally not at right
        angles). Colored: each estimated h_j (however many) projected onto the
        plane; the gap to the circle is its out-of-subspace part, its angle off its
        own axis the in-plane rotation.
        """
        import matplotlib.pyplot as plt
        from fl_plot import THEME, draw_disk_frame

        if self.model.k != 2:
            raise ValueError("the plane picture needs a true k = 2")
        fr = fr or self.frames()
        beta_n = self.model.B / np.linalg.norm(self.model.B, axis=0)
        Bc, Mc = self.b_bar.T @ beta_n, fr["cross"].to_numpy()
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        draw_disk_frame(ax, labels=(r"$\bar b_1$", r"$\bar b_2$"))
        for j, name in enumerate(self.model.names):
            ax.annotate("", xy=tuple(Bc[:, j]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=THEME.gray, lw=1.4, ls="--"))
            ax.text(*(Bc[:, j] * 0.72 + [0, 0.07]), rf"$\beta_{{\rm {name}}}$",
                    color=THEME.gray, ha="center", va="center", fontsize=11)
        for j in range(self.k):
            col = THEME.factor_color(j)
            ax.annotate("", xy=tuple(Mc[:, j]), xytext=(0, 0), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
            ax.text(*(Mc[:, j] * 1.12 + [0, -0.06]), rf"$\Pi h_{j + 1}$", color=col,
                    ha="center", va="center", fontsize=12)
        ang = np.degrees(np.arccos(np.clip(beta_n[:, 0] @ beta_n[:, 1], -1, 1)))
        ax.set_title(f"Factor plane: β columns at {ang:.1f}°, b̄ at 90°\n"
                     f"$p={self.p}$, $n={self.est.n}$, estimated $k={self.k}$", color=THEME.navy,
                     fontsize=THEME.title_fontsize)
        return fig
