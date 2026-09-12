# Archived: B redrawn per replicate

These are the notebooks as they stood before the switch to a shared loading
matrix. They use the runner's default `shared_loadings=False`, under which
**every replicate draws its own superset B** at `p_max` and is scored against
its own truth, with `p` walking the prefixes of that replicate's draw. The
sweep therefore averages over universes drawn from the population law.

The live notebooks in `notebooks/` are the `_sharedB` variants: one B is drawn
once for the whole experiment and all 1000 replicates are scored against that
same truth — the design the paper text describes ("this loading matrix B is
fixed for all 1000 paths").

## What differs

Only `shared_loadings` on the two `DesignSpec`s. The factor and idiosyncratic
draws are bit-identical between the two designs (the runner burns the draws the
per-replicate B would have consumed), so the B-free theory quantities — `floor`,
`rhs`, `rotation`, i.e. every band in the figures — are unchanged. Only the
loading geometry, and therefore the measured errors, move.

## Why both exist

Under a single fixed B the small-p end of each figure is a readout of that one
draw's geometry. Across 24 independent B draws the p=100 factor-2 rotation error
ranges 0.077-0.217 against a band of 0.108, and the curve need not be monotone:
the design seed's B dips below the band at p=500 before returning. Averaging
over B, as these archived notebooks do, removes that. See
`scripts/meta_b_sweep.py` and `nb_outputs/meta_b_sweep/` for the evidence.

## Paths

These write to the original locations (`nb_outputs/ex2v2_df_*.parquet`,
`nb_outputs/ex2v2-combined/`, ...); the `_sharedB` notebooks write to
`*_sharedB` / `*-sharedB` paths, so the two variants never collide and can be
run in either order. Run order within a variant still matters, because the
ex-2/ex-3 notebooks share one cache: run the ex-2 flplot notebook first, then
ex-2_combined and ex-3. ex-6 is independent.
