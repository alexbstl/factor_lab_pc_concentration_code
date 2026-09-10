# factor_lab — PC concentration code

Simulation model and figure notebooks for the PC-concentration paper.

```
sim/factor_lab/   simulation package
sim/fl_experiment_{setup,runner}.py, sim/sim_theorem_partii.py
fl_plot.py        figure grammar
notebooks/        ex-2_combined, ex-2_heavy_tail_v2, ex-3_pathwise_formula31,
                  ex-6_rotation_figures
nb_outputs/       sweep caches and figures — gitignored
```

Run notebooks from `notebooks/`; they put the repo root and `sim/` on `sys.path`, so
no install step is needed. Sweeps cache to `nb_outputs/*.parquet` and regenerate if
missing (slow). Figures are written only when a notebook's `SAVE_FIGS` flag is on.
