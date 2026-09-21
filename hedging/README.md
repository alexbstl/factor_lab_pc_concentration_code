# hedging — does rotation error survive a whole-block factor hedge?

Simulations for N.'s question: hedging one factor against one estimated
direction carries the full per-factor error $\sin^2\angle(h_j,\bar b_j)$,
rotation term included. Does hedging the *entire* $k$-factor block — regressing
the book on all $k$ estimated factor-mimicking portfolios — escape that rotation
term, leaving only the estimable out-of-subspace error?

**Short answer from the simulations: yes, exactly — for the full block only.**

```
hedge_lab.py                              analyses + Experiment for the existing engine
ex-h1_subspace_vs_pairwise_hedging.ipynb  the figures and the argument
```

Nothing under `sim/` is modified. `hedge_lab.py` plugs into the same
`ModelSpec` / `DesignSpec` / `Experiment` seams as `sim_theorem_partii.py`, and
the sweeps use the paper's design (one shared true `B`, nested assets and time).

## The claim, in one line

With both frames orthonormal, let $M = \bar B H$ be the $k\times k$ cross-Gram.
Then

```
pairwise (Stiefel)    sum_j sin^2 angle(h_j, b_j)  =  k - sum_j M_jj^2
subspace (Grassmann)  sum_j sin^2 angle(h_j, B)    =  k - ||M||_F^2
```

A rotation of the estimated frame inside its own span sends $M \mapsto MQ$. The
Frobenius norm is invariant; the diagonal is not. Everything else follows.

## What the notebook establishes

| | result |
|---|---|
| Invariance | An adversarial $Q$ drives the pairwise error to its maximum $k$ (every factor at $\pi/2$) while the subspace error moves by $\sim10^{-15}$ |
| The hedge is a projection | The regression hedge equals $(I-\Pi_H)w$ exactly, so the hedged **book** — not just its risk — is unchanged by any rotation ($\sim10^{-16}$) |
| Only the full block | Partial hedges ($m<k$) shift by $\approx 0.81$ in residual exposure under a relabelled frame; at $m=k$ the shift is $\sim10^{-15}$ |
| Estimable ceiling | $\sqrt{\sum_j \hat\ell/\hat\theta_j}$ tracks the true out-of-subspace ceiling to within 4–8% over $n\in[20,250]$ |
| The floor is set by $n$ | Growing $p$ from 100 to 20,000 leaves it flat; growing $n$ shrinks it |
| The blind spot | The worst-case book satisfies $\Pi_H w = 0$ — the estimate sees it as factor-neutral, so the hedge is the identity on it and removes exactly 0% of its factor variance |

## Running it

Run the notebook from this folder; it locates the repo root and puts it and
`sim/` on `sys.path`, so there is no install step. Sweeps (300 replicates) cache
to `nb_outputs/hedging/*.parquet` and regenerate in ~15 s if missing. Figures are
written only when `SAVE_FIGS` is on, or per figure via `show(fig, stem, save=True)`.

## Open threads

- Re-run under the heavy-tailed process used in ex-2/ex-3 (`student_t` factor and
  idiosyncratic draws) to confirm none of this leans on Gaussianity.
- Misspecified $k$: when the estimated block does not match the true factor
  count, "hedge the whole block" is ill-defined and the invariance argument has
  no purchase. This is the case most worth understanding next.
