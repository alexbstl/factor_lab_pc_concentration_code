"""
hedge_lab.py
============
Hedging probes for the PC-concentration model: **does rotation error survive
when the whole k-factor block is hedged at once?**

The claim under test (N.'s note)
--------------------------------
Hedging factor *j* against the estimated direction h_j is a Stiefel-level
operation — it matches one estimated frame vector to one true frame vector — so
it inherits the full per-factor error sin²∠(h_j, b̄_j), rotation term included,
and Theorem 3 says that rotation term is essentially unconstrained by the data.
Hedging the whole block — regressing the portfolio on all k estimated
factor-mimicking portfolios and trading the opposite coefficients — is instead a
Grassmannian operation: it depends on col(H) only, so any rotation of the
estimated frame inside its own span leaves it untouched. What is left is the
out-of-subspace error alone, which Corollary 1 makes estimable as Σ_j ℓ/θ_j.

The algebra the simulations check
---------------------------------
Write M = b_pop @ H  (k×k), the cross-Gram of the true population directions
b̄_i against the estimated PCs h_j. Both frames are orthonormal, so

    pairwise (Stiefel)   Σ_j sin²∠(h_j, b̄_j) = k − Σ_j M_jj²
    subspace (Grassmann) Σ_j sin²∠(h_j, 𝔅)  = k − ‖M‖_F² = ½‖Π_H − Π_𝔅‖_F²

Replacing H by HQ for an orthogonal k×k Q sends M → MQ. The Frobenius norm is
invariant, so the subspace quantity does not move at all; the diagonal is not,
so the pairwise quantity moves anywhere in [0, k]. Everything below is a
consequence of those two lines, measured on simulated data rather than asserted.

What a hedge actually is here
-----------------------------
A manager regressing portfolio returns Yw on the estimated factor-mimicking
returns YH and trading the opposite coefficients gets hedged weights
w − H β̂ with β̂ = (H'Y'YH)^{-1} H'Y'Y w. Because H holds the *sample* PCs of the
same Y, H'Y'YH is the diagonal Λ of top sample eigenvalues and H'Y'Y = ΛH', so
β̂ = H'w exactly: the regression hedge **is** the orthogonal projection
(I − Π_H)w. :func:`regression_hedge_coefficients` checks this numerically rather
than taking it on faith; :func:`hedge` then uses the projection form.

A *partial* hedge of the first m < k estimated directions is (I − H_m H_m')w and
is not rotation-invariant — that is the caveat in the note, and the m-sweep in
the notebook is what makes it visible.

Layout
------
This module supplies analyses + an ``Experiment`` for the existing engine
(``fl_experiment_setup`` / ``fl_experiment_runner``); it changes nothing in
``sim/``. Rows carry ``j = m`` (the hedged block size) so ``fl_plot.grid`` can
use block size as its column dimension.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fl_experiment_setup import BaseExperiment, register_experiment
from factor_lab.analyses.spectral import compute_true_eigenvalues

__all__ = [
    "sample_pcs",
    "cross_gram",
    "subspace_geometry",
    "regression_hedge_coefficients",
    "hedge",
    "worst_case_portfolio",
    "cyclic_relabel",
    "random_rotation",
    "adversarial_rotation",
    "standard_portfolios",
    "HedgingAnalysis",
    "HedgingExperiment",
]


# ── Estimation ────────────────────────────────────────────────────────────────


def sample_pcs(Y: np.ndarray, k: int, center: bool = True):
    """Top-k sample PCs of ``Y`` (p, n) plus the dual spectrum used by Cor. 1.

    Same construction as the ex-2/ex-3 notebooks: eigendecompose the n×n Gram
    Y'Y (cost O(p n²)), keep the top-k left singular vectors as the estimated
    loading directions H (p, k), rescale the dual spectrum by 1/(np), and split
    it into the k spikes θ_j and the bulk average ℓ.

    Returns ``(H, theta, ell)`` with ``H`` orthonormal in its columns.
    """
    if center:
        Y = Y - Y.mean(axis=1, keepdims=True)
    p, n = Y.shape
    gram = Y.T @ Y
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    top_k = order[:k]
    sv = np.sqrt(np.maximum(eigenvalues[top_k], 0.0))
    H = (Y @ eigenvectors[:, top_k]) / np.where(sv > 1e-14, sv, 1.0)   # (p, k)

    dual = eigenvalues[order] / (n * p)
    theta = dual[:k]
    # Drop the last dual eigenvalue when centered: row-demeaning Y costs one
    # degree of freedom, leaving an exact zero that would drag the bulk mean down.
    bulk = dual[k:n - 1] if center else dual[k:]
    return H, theta, float(bulk.mean())


# ── Geometry ──────────────────────────────────────────────────────────────────


def cross_gram(b_pop: np.ndarray, H: np.ndarray) -> np.ndarray:
    """M = b_pop @ H, the (k, k) cross-Gram of true directions against PCs.

    ``b_pop`` is (k, p) with orthonormal *rows* (the population loading
    directions b̄_i); ``H`` is (p, k) with orthonormal columns. M_ij =
    ⟨b̄_i, h_j⟩ = cos of the angle between them, so every quantity in this
    module is a function of M alone.
    """
    return b_pop @ H


def subspace_geometry(b_pop: np.ndarray, H: np.ndarray) -> dict:
    """The Stiefel and Grassmann error quantities for one (b_pop, H) pair.

    Keys:

    - ``oos_j``      ‖Π⊥h_j‖² = sin²∠(h_j, 𝔅), the out-of-subspace term.
    - ``pair_j``     sin²∠(h_j, b̄_j), the full per-factor (Stiefel) error.
    - ``rot_j``      pair_j − oos_j, the in-subspace rotation mass.
    - ``oos_sum``    Σ_j oos_j = k − ‖M‖_F² = ½‖Π_H − Π_𝔅‖_F² — rotation-free,
                     and the quantity Corollary 1 estimates by Σ_j ℓ/θ_j.
    - ``pair_sum``   Σ_j pair_j = k − Σ_j M_jj² — moves under H → HQ.
    - ``sin2_principal``  sin² of the principal angles between col(H) and 𝔅,
                     descending. Their sum is ``oos_sum`` (an identity the
                     notebook checks); the largest is the operator norm
                     ‖Π_𝔅(I − Π_H)‖², i.e. the *tight* mis-hedge ceiling.
    """
    M = cross_gram(b_pop, H)
    k = M.shape[0]
    in_subspace = (M ** 2).sum(axis=0)              # ‖Π_𝔅 h_j‖²
    oos_j = 1.0 - in_subspace
    pair_j = 1.0 - np.diag(M) ** 2
    sin2_principal = np.sort(
        np.clip(1.0 - np.linalg.eigvalsh(M @ M.T), 0.0, 1.0))[::-1]
    return {
        "M": M,
        "oos_j": oos_j,
        "pair_j": pair_j,
        "rot_j": pair_j - oos_j,
        "oos_sum": float(oos_j.sum()),
        "pair_sum": float(pair_j.sum()),
        "subspace_dist_fro2": float(2.0 * oos_j.sum()),   # ‖Π_H − Π_𝔅‖_F²
        "sin2_principal": sin2_principal,
        "k": k,
    }


# ── Hedges ────────────────────────────────────────────────────────────────────


def regression_hedge_coefficients(Y: np.ndarray, w: np.ndarray,
                                  H: np.ndarray) -> np.ndarray:
    """β̂ from regressing portfolio returns Yw on factor-mimicking returns YH.

    The honest, manager's-eye construction: no knowledge of B anywhere, just a
    least-squares fit of the book against the estimated factor portfolios.
    ``Y`` is (p, n) — the same (already centered, if centering) matrix H came
    from. Returned as a (k,) vector; it equals ``H.T @ w`` to machine precision
    whenever H holds that Y's sample PCs, which is what makes the block hedge a
    projection. Kept separate from :func:`hedge` so the notebook can verify the
    identity instead of assuming it.
    """
    G = (w @ Y)                     # (n,)  portfolio returns
    Gf = H.T @ Y                    # (k, n) factor-mimicking returns
    return np.linalg.lstsq(Gf.T, G, rcond=None)[0]


def hedge(w: np.ndarray, H: np.ndarray, m: int | None = None) -> np.ndarray:
    """Hedge ``w`` against the first ``m`` columns of ``H`` (default: all k).

    Returns (I − H_m H_m')w — the residual book after trading away the fitted
    exposure to those m estimated factor portfolios.
    """
    Hm = H if m is None else H[:, :m]
    return w - Hm @ (Hm.T @ w)


def worst_case_portfolio(b_pop: np.ndarray, H: np.ndarray) -> np.ndarray:
    """The unit book whose block hedge leaves the most true factor exposure.

    Maximizes ‖Π_𝔅(I − Π_H)w‖ over ‖w‖ = 1, i.e. the top right singular vector
    of A = b_pop(I − Π_H) (k, p). Computed through the k×k Gram
    A A' = I − MM' instead of anything p×p, so it stays cheap at p = 50,000.
    The attained value is the largest principal sine — the tight ceiling.
    """
    M = cross_gram(b_pop, H)
    vals, vecs = np.linalg.eigh(np.eye(M.shape[0]) - M @ M.T)
    u = vecs[:, int(np.argmax(vals))]              # top left singular vector
    w = b_pop.T @ u                                 # A'u before the (I − Π_H)
    w = w - H @ (H.T @ w)
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-14 else w


# ── Rotations of the estimated frame (same subspace, different basis) ─────────


def cyclic_relabel(H: np.ndarray) -> np.ndarray:
    """Roll the estimated frame one slot: h_j ← h_{j+1 mod k}.

    The most literal form of "you could be mimicking any factor": the manager
    hedges what they believe is factor 1 using what is really the second PC.
    An orthogonal change of basis inside col(H), so Π_H is untouched and the
    full block hedge cannot notice — while every partial hedge does.
    """
    return H[:, np.roll(np.arange(H.shape[1]), -1)]


def random_rotation(k: int, rng: np.random.Generator) -> np.ndarray:
    """A Haar-random orthogonal k×k matrix (QR of a Gaussian, sign-fixed)."""
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


def adversarial_rotation(b_pop: np.ndarray, H: np.ndarray,
                         restarts: int = 8, seed: int = 0) -> np.ndarray:
    """Orthogonal Q maximizing the *pairwise* error Σ_j sin²∠((HQ)_j, b̄_j).

    Equivalently minimizes Σ_j (MQ)_jj² — driving every per-factor angle toward
    π/2 while col(HQ) = col(H) is fixed. Parametrizes Q = expm(A) over
    skew-symmetric A and runs a few random restarts; used only for the
    illustrative worst case in the notebook, not in the sweep.
    """
    from scipy.linalg import expm
    from scipy.optimize import minimize

    M = cross_gram(b_pop, H)
    k = M.shape[0]
    iu = np.triu_indices(k, 1)

    def to_Q(params):
        A = np.zeros((k, k))
        A[iu] = params
        return expm(A - A.T)

    def objective(params):
        return float((np.diag(M @ to_Q(params)) ** 2).sum())

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        res = minimize(objective, rng.standard_normal(len(iu[0])), method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    return to_Q(best.x)


# ── Portfolios ────────────────────────────────────────────────────────────────


def standard_portfolios(p: int, seed: int = 20260920,
                        p_max: int | None = None) -> dict:
    """The fixed books hedged at each p, each normalized to ‖w‖₂ = 1.

    Unit L2 norm (not unit gross notional) is what makes ‖Π_𝔅 w_h‖ directly
    comparable to the principal-sine ceiling, which is an operator norm.

    - ``equal_weight`` — flat book; loads hard on the market-like factor 1.
    - ``long_short``   — a Gaussian long-short book, mostly idiosyncratic.

    Both are prefix-consistent across the nested p grid: the long-short weights
    are drawn once at ``p_max`` from a dedicated stream (never the experiment's
    master RNG, so adding portfolios cannot shift the simulation), then the
    first p are taken and renormalized — the same book on the same assets, so a
    p-sweep is a refinement rather than a fresh draw.
    """
    w_eq = np.ones(p) / np.sqrt(p)
    draw = np.random.default_rng(seed).standard_normal(p_max or p)[:p]
    w_ls = draw / np.linalg.norm(draw)
    return {"equal_weight": w_eq, "long_short": w_ls}


# ── Analysis ──────────────────────────────────────────────────────────────────


class HedgingAnalysis:
    """Per-replicate hedging outcomes for every block size m = 1 … k.

    For each portfolio and each m, hedges against (a) the estimated frame as
    ordered by the PCA and (b) the cyclically relabeled frame — the same
    subspace, a rotated basis — and records the true factor exposure left
    behind. At m = k the two must agree bit for bit; below m = k they need not,
    and that gap is the rotation error the note says the block hedge escapes.

    The ``worst_case`` book is rebuilt per replicate (it depends on H), so its
    residual exposure at m = k sits exactly on the tight ceiling.
    """

    def __init__(self, b_pop: np.ndarray, portfolios: dict, model,
                 center: bool = True, include_worst_case: bool = True):
        self.b_pop = b_pop                      # (k, p) orthonormal rows
        self.portfolios = portfolios
        self.model = model
        self.center = center
        self.include_worst_case = include_worst_case

    def _factor_variance(self, w: np.ndarray) -> float:
        """w'B'FB w — the part of the book's variance the factors explain."""
        exposure = self.model.B @ w
        return float(exposure @ self.model.F @ exposure)

    def analyze(self, context) -> dict:
        k = context.k
        Y = context.security_returns.T                  # (p, n)
        if self.center:
            Y = Y - Y.mean(axis=1, keepdims=True)
        H, theta, ell = sample_pcs(Y, k, center=False)   # already centered
        geom = subspace_geometry(self.b_pop, H)
        H_rot = cyclic_relabel(H)

        ell_over_theta = ell / theta
        ceilings = {
            # tight, but needs the truth: the largest principal sine.
            "ceiling_tight": float(np.sqrt(geom["sin2_principal"][0])),
            # Corollary 1, measured against the truth: sqrt(Σ_j ‖Π⊥h_j‖²).
            "ceiling_measured": float(np.sqrt(geom["oos_sum"])),
            # Corollary 1, estimated from data alone: sqrt(Σ_j ℓ/θ_j).
            "ceiling_estimable": float(np.sqrt(max(ell_over_theta.sum(), 0.0))),
            # what a pairwise reading of the error would claim — inflated by
            # rotation, and by Theorem 3 not estimable in the first place.
            "ceiling_pairwise": float(np.sqrt(geom["pair_sum"])),
        }

        books = dict(self.portfolios)
        if self.include_worst_case:
            books["worst_case"] = worst_case_portfolio(self.b_pop, H)

        rows = []
        for name, w in books.items():
            exposure_unhedged = self.b_pop @ w
            oracle = w - self.b_pop.T @ exposure_unhedged
            for m in range(1, k + 1):
                w_h = hedge(w, H, m)
                w_r = hedge(w, H_rot, m)
                e_h, e_r = self.b_pop @ w_h, self.b_pop @ w_r
                rows.append({
                    "j": m, "portfolio": name,
                    # residual exposure to the m factors the manager meant to kill
                    "target_exposure": float(np.linalg.norm(e_h[:m])),
                    "target_exposure_rot": float(np.linalg.norm(e_r[:m])),
                    # residual exposure to the whole true factor subspace
                    "total_exposure": float(np.linalg.norm(e_h)),
                    "total_exposure_rot": float(np.linalg.norm(e_r)),
                    "resid_factor_var": self._factor_variance(w_h),
                    "resid_factor_var_rot": self._factor_variance(w_r),
                    "unhedged_target_exposure": float(np.linalg.norm(exposure_unhedged[:m])),
                    "unhedged_total_exposure": float(np.linalg.norm(exposure_unhedged)),
                    "unhedged_factor_var": self._factor_variance(w),
                    "oracle_total_exposure": float(np.linalg.norm(self.b_pop @ oracle)),
                    **ceilings,
                    "oos_sum": geom["oos_sum"],
                    "pair_sum": geom["pair_sum"],
                    "rot_sum": float(geom["pair_sum"] - geom["oos_sum"]),
                    "subspace_dist_fro2": geom["subspace_dist_fro2"],
                    "sin2_principal_max": float(geom["sin2_principal"][0]),
                    "sin2_principal_sum": float(geom["sin2_principal"].sum()),
                    "ell_over_theta_sum": float(ell_over_theta.sum()),
                })
        return {"hedge_rows": rows,
                "oos_j": geom["oos_j"], "pair_j": geom["pair_j"],
                "rot_j": geom["rot_j"]}


@register_experiment("subspace_hedging")
class HedgingExperiment(BaseExperiment):
    """Wires :class:`HedgingAnalysis` to the engine.

    ``cell_setup`` builds the population directions once per cell (ARPACK out of
    the replicate loop) and the fixed books for this p; ``record`` tags each
    row with (n, p). Rows use ``j`` for the hedged block size m, so
    ``fl_plot.grid`` renders one column per block size.
    """

    def __init__(self, center: bool = True, portfolio_seed: int = 20260920,
                 p_max: int | None = None, include_worst_case: bool = True):
        self.center = center
        self.portfolio_seed = portfolio_seed
        self.p_max = p_max
        self.include_worst_case = include_worst_case

    def cell_setup(self, model, n: int, p: int):
        _, b_pop = compute_true_eigenvalues(model, model.k)
        books = standard_portfolios(p, seed=self.portfolio_seed, p_max=self.p_max)
        return [HedgingAnalysis(b_pop, books, model, center=self.center,
                                include_worst_case=self.include_worst_case)]

    def record(self, n: int, p: int, merged: dict) -> list[dict]:
        return [{"n": n, "p": p, **row} for row in merged["hedge_rows"]]
