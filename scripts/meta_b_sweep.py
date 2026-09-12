#!/usr/bin/env python3
"""Meta-experiment: how much do the paper figures depend on WHICH B was drawn?

Each "experiment" here fixes ONE loading matrix B — drawn from the population law
at p_max — and runs the full n_reps-path simulation against it, with p walking
PREFIXES of that same B. That is the design described in main-16.tex ("This
loading matrix B is fixed for all 1000 paths"), as opposed to the notebooks'
current design, which redraws B per path. Repeating the whole experiment over
many B seeds shows the sampling distribution of the figures themselves.

Three figures per B seed, matching the paper:

    plot1_combined_p   paper Figure 1   (ex2c_combined_grouped_p)
    plot2_rotation_p   paper Figure 5   (ex6_insub_vs_limit_p_qcolor)
    plot3_combined_n   paper Figure 6   (ex2c_combined_grouped_n)

Plot code is exec'd out of the notebooks themselves, so these are literally the
same marks/themes the paper figures use — not a reimplementation.

PATHS. By default every B draw reuses the SAME n_reps factor/idio paths. Because
`floor`, `rhs` and `rotation` are B-free (they use the fixed limiting
G_B = diag(1.25,1,1)) and depend only on F, this makes every theory band
identical across seeds: flipping through the folders, the bands stay pinned and
only the black dots and boxes move, so what you see is purely the B effect.
Pass --independent-paths to redraw paths per seed instead (fully independent
replications; bands then wobble by Monte Carlo noise too).

Outputs under --out:
    plot1_combined_p/seed_<S>.png ...      one image per B seed, per plot
    plot2_rotation_p/ , plot3_combined_n/
    <plot>/_reference.png                  the notebooks' CURRENT design, same paths
    <plot>/_reference2.png                 the collaborator's own B draw, same paths
    overlay_plot1.png, overlay_plot2.png   ALL seeds' dot-curves on one axes
    contact_<plot>.png                     thumbnail sheet of every seed
    summary.csv                            per seed x p x factor: dot, band, above?
    manifest.csv                           seed -> files, run parameters
    data/seed_<S>_{p,n}.parquet            raw rows (skip with --no-save-data)

Usage:
    python scripts/meta_b_sweep.py --n-seeds 24
    python scripts/meta_b_sweep.py --seeds 1,2,3,99 --reps 300
    python scripts/meta_b_sweep.py --replot-only        # re-render from saved data
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# One BLAS thread per process: with a pool this wide, oversubscription costs more
# than it buys. Must be set before numpy is imported — workers re-import this
# module under the "spawn" start method, so this applies to them too.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")            # must precede any pyplot import (incl. notebook cells)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO), str(REPO / "sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loguru import logger
logger.remove()

from fl_experiment_runner import (build_model, simulate_returns, run_analyses,
                                  _slice_model_to_p, _slice_to_np)

NB6 = REPO / "notebooks" / "ex-6_rotation_figures.ipynb"
NB2 = REPO / "notebooks" / "ex-2_combined.ipynb"
B_P_MAX = 50_000                 # every B is drawn at this size; sweeps use prefixes

PLOTS = {
    "plot1_combined_p": "paper Fig 1 - total error + components, growing p",
    "plot2_rotation_p": "paper Fig 5 - in-subspace rotation vs limit, growing p",
    "plot3_combined_n": "paper Fig 6 - total error + components, growing n",
}


# ── pulling the real notebook code ────────────────────────────────────────────

def _cells(nb_path: Path, specs) -> str:
    """Concatenate selected notebook cell sources.

    `specs` is a list of (cell_index, marker). When marker is not None the cell
    source is truncated at its first occurrence — used to take a definition
    without the driver loop that follows it.
    """
    nb = json.loads(nb_path.read_text())
    out = []
    for idx, marker in specs:
        src = "".join(nb["cells"][idx]["source"])
        if marker is not None:
            cut = src.find(marker)
            if cut < 0:
                raise RuntimeError(f"marker {marker!r} not found in {nb_path.name} cell {idx}")
            src = src[:cut]
        out.append(src)
    return "\n".join(out)


def load_namespaces():
    """Namespaces holding the notebooks' own model, designs, experiment and plot code.

    The cells print on import ("ready", ...); swallow that so pool workers do not
    each echo it into the run log.
    """
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ns6: dict = {"__name__": "__meta_ns6__"}
        exec(_cells(NB6, [(1, None), (3, None), (5, "_REQ ="), (7, "df_p, df_n =")]), ns6)
        ns2: dict = {"__name__": "__meta_ns2__"}
        exec(_cells(NB2, [(1, None), (8, "BOX_SPECS")]), ns2)
    return ns6, ns2


_NS = None


def get_ns():
    """Namespaces, built once per process (workers are spawned, so each pays this once)."""
    global _NS
    if _NS is None:
        _NS = load_namespaces()
    return _NS


# ── the fixed-B runner (build_model hoisted above the replicate loop) ─────────

def run_with_fixed_B(model_spec, design_spec, experiment, B_full, rep_seeds, *, align=True):
    """`_run_nested`, except B is supplied once instead of redrawn per replicate.

    B_full is a model built at B_P_MAX; it is sliced to this design's p_max, so
    the p-sweep and the n-sweep of one seed live in the SAME universe.

    align=True burns exactly the draws `_run_nested` would have spent on this
    replicate's B, leaving the factor/idio stream identical to the per-path
    design. That is what makes bands line up across seeds and against the
    reference run.
    """
    if design_spec.subsample != "prefix":
        raise NotImplementedError(f"subsample={design_spec.subsample!r} unsupported")
    if not design_spec.nest_time:
        raise NotImplementedError("this variant assumes nest_time=True")

    p_values, n_values = list(design_spec.p_values), list(design_spec.n_values)
    p_max, n_max, k = max(p_values), max(n_values), model_spec.k_factors
    model_design = _slice_model_to_p(B_full, p_max)

    # population bases per p: once for the whole experiment, not once per replicate
    analyses_by_p = {p: experiment.cell_setup(_slice_model_to_p(model_design, p), n_values[0], p)
                     for p in p_values}

    records = []
    for r, seed in enumerate(rep_seeds):
        rep_rng = np.random.default_rng(int(seed))
        if align:
            build_model(model_spec, p_max, rep_rng)      # burn B's draws, keep F/Z aligned
        ctx_full = simulate_returns(
            model=model_design, n=n_max,
            factor_return_spec=design_spec.factor_return_sampler,
            idio_return_spec=design_spec.idio_return_sampler,
            k=k, rep_rng=rep_rng)
        for n in n_values:
            for p in p_values:
                merged = run_analyses(_slice_to_np(ctx_full, n, p), analyses_by_p[p])
                for row in experiment.record(n, p, merged):
                    row["rep"] = r
                    records.append(row)
    return pd.DataFrame(records)


COLLAB_SEED = 20240917          # SEED in the collaborator's fig2_simulate.py


def collaborator_model(model_spec, p_max, seed=COLLAB_SEED):
    """Our model object carrying the collaborator's loading matrix.

    B is drawn exactly as fig2_simulate.py draws it — one generator seeded with
    `seed`, columns filled in order N(1,0.5), N(0,1), N(0,1) — then transposed
    into our (k, p) convention. Idio vols and factor variances still come from
    our own builder, so the loading matrix is the only thing that differs.
    """
    import dataclasses
    model = build_model(model_spec, p_max, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    B = np.empty((p_max, model_spec.k_factors))
    B[:, 0] = rng.normal(1.0, 0.5, p_max)
    B[:, 1] = rng.normal(0.0, 1.0, p_max)
    B[:, 2] = rng.normal(0.0, 1.0, p_max)
    return dataclasses.replace(model, B=B.T)


def run_per_path_B(model_spec, design_spec, experiment, rep_seeds):
    """The notebooks' CURRENT design: a fresh B per replicate. Used for the reference image."""
    p_values, n_values = list(design_spec.p_values), list(design_spec.n_values)
    p_max, n_max, k = max(p_values), max(n_values), model_spec.k_factors
    records = []
    for r, seed in enumerate(rep_seeds):
        rep_rng = np.random.default_rng(int(seed))
        model_full = build_model(model_spec, p_max, rep_rng)
        abp = {p: experiment.cell_setup(_slice_model_to_p(model_full, p), n_values[0], p)
               for p in p_values}
        ctx_full = simulate_returns(
            model=model_full, n=n_max,
            factor_return_spec=design_spec.factor_return_sampler,
            idio_return_spec=design_spec.idio_return_sampler,
            k=k, rep_rng=rep_rng)
        for n in n_values:
            for p in p_values:
                merged = run_analyses(_slice_to_np(ctx_full, n, p), abp[p])
                for row in experiment.record(n, p, merged):
                    row["rep"] = r
                    records.append(row)
    return pd.DataFrame(records)


# ── the three figures ─────────────────────────────────────────────────────────

def fig_combined(df, key, ns2, title):
    """paper Fig 1 / Fig 6 - BandStack + total dots over grouped component boxes."""
    grid, Row, BandStack = ns2["grid"], ns2["Row"], ns2["BandStack"]
    THEME, TOTAL_T = ns2["THEME"], ns2["TOTAL_T"]
    grouped = ns2["GroupedBoxDist"]([("gap", "black", "total"),
                                     ("diff_oos", THEME.oos_fill, "out-of-subspace (OE)"),
                                     ("diff_insub", THEME.insub_fill, "in-subspace (IE)")])
    fig, axes = grid(
        df, key,
        rows=[Row(BandStack(labels=("out-of-subspace (OE)", "in-subspace (IE)",
                                    r"$\sin^2\angle_p(h,b)$")),
                  ylabel="average error", ylim=(0, 1),
                  yticks=THEME.sin2_ticks, height=1.0),
              Row(grouped, ylabel="asymptotic − realized error")],
        theme=TOTAL_T, caption=False, suptitle=title,
        formats=("png",), out_path=None)
    axes[1, 0].legend(handles=grouped.legend_handles(TOTAL_T),
                      fontsize=TOTAL_T.legend_fontsize, loc="lower right")
    return fig


def fig_rotation(df, ns6, title):
    """paper Fig 5 - in-subspace rotation against its factor-space limit."""
    grid, Row, Band, MeanCI, BoxDist, Theme = (ns6["grid"], ns6["Row"], ns6["Band"],
                                               ns6["MeanCI"], ns6["BoxDist"], ns6["Theme"])
    rot_t = Theme(factor_colors=("#74c476",) * 6, factor_cmaps=("Greens",) * 6,
                  factor_hatches=ns6["FACTOR_HATCHES"],
                  ci_caps=dict(fmt="none", ecolor="black", elinewidth=0.0, capsize=0))
    fig, _ = grid(
        df, "p",
        rows=[Row([Band("rotation", label="rotation error (RE)",
                        fill="#a1d99b", line="#238b45"),
                   MeanCI("sin2_in", label=r"$\sin^2\angle(\Pi h_j, b_j)$")],
                  ylabel="average error", height=1.0),
              Row(BoxDist("diff_in_rot"), ylabel="asymptotic − realized error")],
        theme=rot_t, caption=False, suptitle=title,
        formats=("png",), out_path=None)
    return fig


def save(fig, stem: Path, formats=("png",), dpi=110, thumb: Path | None = None):
    """Write `fig` as stem.<ext> for each requested format.

    Also drops a small PNG thumbnail when `thumb` is given: contact sheets are
    raster mosaics, so they need one even when the figures themselves are PDFs.
    A same-stem file in a format no longer requested is removed, so switching
    --format does not leave the folders holding two copies of every figure.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    for ext in ("png", "pdf"):
        if ext not in formats:
            stale = stem.with_suffix(f".{ext}")
            if stale.exists():
                stale.unlink()
    if thumb is not None:
        thumb.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(thumb, dpi=52, bbox_inches="tight")
    plt.close(fig)


# ── aggregate views ───────────────────────────────────────────────────────────

def overlay(per_seed_means, band, xs, title, ylabel, stem, formats=("png",), logx=True):
    """Every seed's dot-curve on one axes, with the (common) theory band behind."""
    k = len(band)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 3.6), squeeze=False)
    for j in range(k):
        ax = axes[0, j]
        for seed in sorted(per_seed_means):
            m = per_seed_means[seed]
            ax.plot(xs, m[:, j], marker="o", ms=3, lw=0.8, alpha=0.42, color="black")
        ax.axhline(band[j], color="#238b45", lw=2.0, zorder=5,
                   label="theory limit (common to all seeds)")
        ax.fill_between([min(xs), max(xs)], 0, band[j], color="#a1d99b", alpha=0.30, zorder=0)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"factor {j + 1}")
        ax.set_xlabel("p (assets)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, stem, formats, dpi=120)


def contact_sheet(folder: Path, out_path: Path, cols=4, thumb_w=460):
    """`folder` holds the PNG thumbnails written alongside each figure."""
    from PIL import Image, ImageDraw
    def _seed_num(f):
        m = re.search(r"seed_(\d+)", f.stem)
        return int(m.group(1)) if m else 0
    imgs = sorted(folder.glob("seed_*.png"), key=_seed_num)
    if not imgs:
        return
    imgs = [f for f in (folder / "_reference.png", folder / "_reference2.png")
            if f.exists()] + imgs
    thumbs = []
    for f in imgs:
        im = Image.open(f).convert("RGB")
        im.thumbnail((thumb_w, 10_000))
        thumbs.append((f.stem, im))
    tw = max(t.width for _, t in thumbs)
    th = max(t.height for _, t in thumbs)
    rows = (len(thumbs) + cols - 1) // cols
    pad, label_h = 8, 16
    sheet = Image.new("RGB", (cols * (tw + pad) + pad,
                              rows * (th + pad + label_h) + pad), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, t) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + pad + label_h)
        d.text((x, y), name, fill="black")
        sheet.paste(t, (x, y + label_h))
    sheet.save(out_path)


# ── one experiment = one pool task ────────────────────────────────────────────

def _designs(ns6, reps):
    design_p, design_n = ns6["design_p"], ns6["design_n"]
    if reps:
        from dataclasses import replace
        design_p = replace(design_p, n_reps=reps)
        design_n = replace(design_n, n_reps=reps)
    return design_p, design_n


def run_job(spec: dict) -> dict:
    """Simulate one B draw (or the reference), render its three figures, return its summary.

    Runs in a worker process: everything it needs is rebuilt from `spec`, which
    carries only picklable primitives.
    """
    ns6, ns2 = get_ns()
    model = ns6["model"]
    Experiment = ns6["InSubRotationExperiment"]
    derive6, derive2 = ns6["derive"], ns2["derive"]
    design_p, design_n = _designs(ns6, spec["reps"])
    n_reps = design_p.n_reps

    tag, bseed, kind = spec["tag"], spec["bseed"], spec.get("kind", "fixed")
    out, data_dir = Path(spec["out"]), Path(spec["out"]) / "data"
    fp, fn = data_dir / f"{tag}_p.parquet", data_dir / f"{tag}_n.parquet"

    base = np.random.default_rng(design_p.random_seed).integers(0, 2 ** 31, size=n_reps)
    rs = (np.random.default_rng(10_000_000 + bseed).integers(0, 2 ** 31, size=n_reps)
          if (spec["independent"] and kind == "fixed") else base)

    have = fp.exists() and fn.exists()
    if spec["replot"] or (spec.get("reuse") and have):
        if not have:
            return {"tag": tag, "skipped": True}
        dp, dn = pd.read_parquet(fp), pd.read_parquet(fn)
    else:
        if kind == "perpath":
            dp = run_per_path_B(model, design_p, Experiment(), rs)
            dn = run_per_path_B(model, design_n, Experiment(), rs)
        else:
            B_full = (collaborator_model(model, B_P_MAX) if kind == "collab"
                      else build_model(model, B_P_MAX, np.random.default_rng(bseed)))
            dp = run_with_fixed_B(model, design_p, Experiment(), B_full, rs)
            dn = run_with_fixed_B(model, design_n, Experiment(), B_full, rs)
        if spec["save_data"]:
            data_dir.mkdir(parents=True, exist_ok=True)
            dp.to_parquet(fp)
            dn.to_parquet(fn)

    # ex-6's frame does not record measured_in_subspace, but it equals sin2_j - oos
    # exactly (checked against the ex-2 frame to 1e-15), so both derives can run.
    def both(d):
        d = d.assign(measured_in_subspace=d["sin2_j"] - d["measured_out_of_subspace"])
        return derive6(derive2(d))

    dp, dn = both(dp), both(dn)
    label = {"perpath": "current design: fresh B per path",
             "collab": f"collaborator's B (seed {COLLAB_SEED})"}.get(
                 kind, f"one fixed B — seed {bseed}")
    fmts = tuple(spec["formats"])
    for folder, fig in (("plot1_combined_p", fig_combined(dp, "p", ns2, f"plot 1 — {label}")),
                        ("plot2_rotation_p", fig_rotation(dp, ns6, f"plot 2 — {label}")),
                        ("plot3_combined_n", fig_combined(dn, "n", ns2, f"plot 3 — {label}"))):
        save(fig, out / folder / tag, fmts, thumb=out / ".thumbs" / folder / f"{tag}.png")

    p_values = list(design_p.p_values)
    k = int(dp["j"].max())
    mp = dp.groupby(["p", "j"])["sin2_j"].mean().unstack().reindex(p_values).to_numpy()
    mr = dp.groupby(["p", "j"])["sin2_in"].mean().unstack().reindex(p_values).to_numpy()
    band_tot = dp.groupby("j")["rhs"].mean().to_numpy()
    band_rot = dp.groupby("j")["rotation"].mean().to_numpy()

    rows = []
    for pi, p_ in enumerate(p_values):
        for j in range(k):
            rows.append(dict(tag=tag, kind=kind, B_seed=bseed, p=p_, factor=j + 1,
                             total_dot=mp[pi, j], total_band=band_tot[j],
                             total_above=bool(mp[pi, j] > band_tot[j]),
                             rot_dot=mr[pi, j], rot_band=band_rot[j],
                             rot_above=bool(mr[pi, j] > band_rot[j])))
    return {"tag": tag, "bseed": bseed, "kind": kind, "skipped": False, "rows": rows,
            "mp": mp, "mr": mr, "band_tot": band_tot, "band_rot": band_rot,
            "p_values": p_values, "n_reps": n_reps}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=24,
                    help="number of B draws, seeds 1..N (default 24)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="explicit comma-separated B seeds, overrides --n-seeds")
    ap.add_argument("--reps", type=int, default=None,
                    help="paths per experiment (default: the notebooks' 1000)")
    ap.add_argument("--out", type=Path, default=REPO / "nb_outputs" / "meta_b_sweep")
    ap.add_argument("--independent-paths", action="store_true",
                    help="redraw paths per B seed (bands then move too)")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the current per-path-B reference run")
    ap.add_argument("--format", dest="formats", default="png",
                    choices=("png", "pdf", "both"),
                    help="figure format: png (default), pdf (vector, ~half the size), or both")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel worker processes (default: min(#jobs, cpu_count))")
    ap.add_argument("--no-save-data", action="store_true")
    ap.add_argument("--reuse-data", action="store_true",
                    help="load a job's saved parquet when present, simulate only what is missing")
    ap.add_argument("--replot-only", action="store_true",
                    help="re-render figures from data/ without simulating")
    args = ap.parse_args()

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else list(range(1, args.n_seeds + 1)))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)

    ns6, _ = get_ns()
    design_p, _dn = _designs(ns6, args.reps)
    n_reps = design_p.n_reps
    p_values = list(design_p.p_values)

    jobs = [] if args.no_reference else [
        {"tag": "_reference", "bseed": None, "kind": "perpath"},
        {"tag": "_reference2", "bseed": COLLAB_SEED, "kind": "collab"}]
    jobs += [{"tag": f"seed_{s_}", "bseed": s_, "kind": "fixed"} for s_ in seeds]
    fmts = ("png", "pdf") if args.formats == "both" else (args.formats,)
    for jb in jobs:
        jb.update(reps=args.reps, independent=args.independent_paths, out=str(out),
                  save_data=not args.no_save_data, replot=args.replot_only,
                  reuse=args.reuse_data, formats=fmts)

    workers = args.workers or min(len(jobs), os.cpu_count() or 4)
    print(f"B seeds: {seeds}")
    print(f"{n_reps} paths/experiment | paths "
          f"{'redrawn per seed' if args.independent_paths else 'SHARED across seeds'}")
    print(f"{len(jobs)} experiments over {workers} worker processes")
    print(f"out: {out}\n", flush=True)

    summary_rows, manifest_rows = [], []
    means_p, means_rot = {}, {}

    if workers == 1:
        results = [run_job(jb) for jb in jobs]
    else:
        import concurrent.futures as cf
        results = []
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_job, jb): jb["tag"] for jb in jobs}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                tag = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"  [{i}/{len(jobs)}] {tag} done", flush=True)
                except Exception as exc:                      # keep the sweep alive
                    print(f"  [{i}/{len(jobs)}] {tag} FAILED: {exc!r}", flush=True)

    for res in results:
        if res.get("skipped"):
            print(f"{res['tag']}: no saved data, skipped")
            continue
        summary_rows.extend(res["rows"])
        if res["kind"] == "fixed":
            means_p[res["bseed"]] = res["mp"]
            means_rot[res["bseed"]] = res["mr"]
        manifest_rows.append(dict(tag=res["tag"], B_seed=res["bseed"], n_reps=res["n_reps"],
                                  paths="independent" if args.independent_paths else "shared",
                                  b_p_max=B_P_MAX, design_seed=design_p.random_seed))

    if not summary_rows:
        print("nothing produced")
        return

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out / "manifest.csv", index=False)

    fixed = summary[summary.kind == "fixed"]
    if means_rot:
        band_rot = fixed.groupby("factor")["rot_band"].first().to_numpy()
        band_tot = fixed.groupby("factor")["total_band"].first().to_numpy()
        overlay(means_rot, band_rot, p_values,
                f"plot 2 — in-subspace rotation, {len(means_rot)} independent B draws",
                r"mean $\sin^2\angle(\Pi h_j,b_j)$", out / "overlay_plot2", fmts)
        overlay(means_p, band_tot, p_values,
                f"plot 1 — total error, {len(means_p)} independent B draws",
                r"mean $\sin^2\angle(h_j,b_j)$", out / "overlay_plot1", fmts)

    for name in PLOTS:
        contact_sheet(out / ".thumbs" / name, out / f"contact_{name}.png")

    print("\nfraction of B draws with the dot ABOVE the theory line")
    for col, nm in (("rot_above", "plot 2 rotation"), ("total_above", "plot 1 total")):
        print(f"\n  {nm}:")
        tab = fixed.groupby(["p", "factor"])[col].mean().unstack()
        print("   " + tab.round(2).to_string().replace("\n", "\n   "))
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
