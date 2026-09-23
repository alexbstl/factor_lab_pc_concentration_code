# factor_lab — PC concentration code

Simulation model and figure notebooks for the PC-concentration paper.

```
sim/factor_lab/   simulation package
sim/fl_experiment_{setup,runner}.py, sim/sim_theorem_partii.py
fl_plot.py        figure grammar
notebooks/        ex-2_combined, ex-2_heavy_tail_v2, ex-3_pathwise_formula31,
                  ex-6_rotation_figures; disk/radar views: ex-6_radar_disks,
                  ex-8_two_factor_disks, ex-9_two_factor_data (data only),
                  ex-10_two_factor_paired_radar — all _sharedB
notebooks/archive/ per-replicate-B variants, superseded by the shared-B design
nb_outputs/       sweep caches and figures — gitignored
hedging/          subspace-vs-pairwise factor hedging (branch work; see hedging/README.md)
```

Run notebooks from `notebooks/`; they put the repo root and `sim/` on `sys.path`, so
no install step is needed. Sweeps cache to `nb_outputs/*.parquet` and regenerate if
missing (slow). Figures are written only when a notebook's `SAVE_FIGS` flag is on.
