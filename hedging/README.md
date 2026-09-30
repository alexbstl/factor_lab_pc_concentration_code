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
ex-h2_two_factor_radar.ipynb              the k=2 dial
ex-h3_paired_dial_colours.ipynb           linking the two dials by colour
ex-h4_memo_running_example.ipynb          the hedges on the memo's p=10, k=2 example
```

**ex-h2 and ex-h3 are a separate line of work from ex-h1** — the k=2 geometry in
its own right, not a demonstration of the hedging claim. Treat them as unrelated
for now.

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

## The k=2 picture (`ex-h2`)

At $k=2$ the true factor subspace is a single *plane*, so one disk holds the whole
estimate: each $h_j$ is plotted at $(\langle\bar b_1,h_j\rangle,\langle\bar b_2,h_j\rangle)$,
the radial shortfall from the unit circle **is** the out-of-subspace error, and the
angular position is the in-plane rotation. Rotating the estimated frame spins the
arrows — radii, axis angles and even the angle between the arrows all move — while
$\sum_j\lVert\Pi_\mathcal{B}h_j\rVert^2=\lVert C\rVert_F^2$ does not.

The payoff figure: for one book, the $m=1$ residual rides a closed locus as $Q$ turns
through 360°, running from 0.066 (about what the block hedge achieves) all the way up
to 0.899 — the *unhedged* exposure, i.e. a rotation at which the single-factor hedge
removes nothing. The $m=2$ residual is one point, 0.0658, for every $Q$ (spread 3e-16).
A standalone notebook: it borrows the picture from `notebooks/ex-10_two_factor_paired_radar_sharedB`
but none of its machinery.

## ex-h3 — linking the two dials by colour

Explores a proposal from N.: keep the factor-1 points $\{p_i\}$ and factor-2 points
$\{q_i\}$ in their own positions but link them **by colour rather than geometry**,
keying the colour to each $p_i$'s own angle through a cyclic map and giving $q_i$ its
partner's colour. If the $\approx90\degree$ relation is clean, the $q$ pattern is the
$p$ pattern rotated a quarter turn; where it breaks, colours bleed into the wrong
sector. It avoids drawing 1000 illegible pairing lines.

**Finding: the quarter turn is clean, and there is an exact reason.** Median hue
mismatch after rotating $q$ by $-90\degree$ onto $p$ is 0.25° at $p=100$ and 0.05° at
$p=50{,}000$, with no sector of systematic breakdown. Writing
$\delta=\theta_q-\theta_p-90\degree$, its spread collapses 161× (8.30° → 0.05°)
across the $p$ grid while **both marginal cloud spreads barely move** (~4.4°) — the
frame wanders as a rigid body, and only its internal geometry converges.

The mechanism is an identity, not a fit. $h_1\perp h_2$ in $\mathbb{R}^p$, so splitting
each into in-plane and out-of-plane parts,

$$0=\langle h_1,h_2\rangle=\langle\Pi h_1,\Pi h_2\rangle+\langle\Pi^{\perp}h_1,\Pi^{\perp}h_2\rangle
\;\Longrightarrow\;\langle\Pi h_1,\Pi h_2\rangle=-\langle\Pi^{\perp}h_1,\Pi^{\perp}h_2\rangle,$$

verified to 1.9e-16, with $\delta=\arcsin(\text{cross term})$ exactly. The
out-of-subspace *masses* hardly change in $p$; their two components **decorrelate**,
which is what drives the cross term — and hence the departure from a right angle — to
zero.

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
