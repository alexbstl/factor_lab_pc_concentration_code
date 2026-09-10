# factor_lab — PC concentration code

Simulation model and figure notebooks supporting the PC-concentration paper. Migrated
from the paper working repo; this repo is a fork of the original `factor_lab`, with the
pre-migration code available in the git history.

## Layout

```
sim/
  factor_lab/              simulation package (model builder, samplers, analyses)
    analysis/              analysis protocol + context
    analyses/              spectral, eigenvector, manifold analyses
  fl_experiment_setup.py   ModelSpec / DesignSpec / BaseExperiment
  fl_experiment_runner.py  run_experiment — sweeps a design, collects replicates
  sim_theorem_partii.py    Eq6RHSAnalysis and friends
fl_plot.py                 composable figure grammar (grid / Row / Band / MeanCI / ...)
notebooks/                 figure notebooks
nb_outputs/                sweep caches (*.parquet) and figures (*.pdf) — gitignored
```

## Notebooks

| Notebook | Figures |
| --- | --- |
| `ex-2_combined.ipynb` | total + component decomposition, combined figure |
| `ex-2_heavy_tail_v2_flplot_mono.ipynb` | ex-2 v2 heavy-tail sweeps (mono theme) |
| `ex-3_pathwise_formula31_flplot_mono.ipynb` | pathwise Formula 3.1, OOS levels |
| `ex-6_rotation_figures.ipynb` | in-subspace rotation, scalar views only |

Radar / disk notebooks (`ex-6_radar_disks`, `ex-8`, `ex-10`) were deliberately not
migrated.

## Running

Each notebook locates the repo root by walking up for a `sim/` directory and puts both
the root and `sim/` on `sys.path`, so no install step is needed — just run from
`notebooks/` with the project environment.

Sweeps cache to `nb_outputs/ex2v2_df_{p,n}.parquet` (ex-2, ex-3) and
`nb_outputs/ex6_df_{p,n}.parquet` (ex-6). If a cache is missing the notebook regenerates
it (1000 replicates per design point — slow). Figures are written only when the
notebook's `SAVE_FIGS` flag is enabled.
