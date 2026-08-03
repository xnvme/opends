#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Pivot bench/sweep history.jsonl into comparison tables and a plot.

Reads every history.jsonl under the input dirs (one record per leg, written by
cijoe-scripts/bench_run.py) and, per opends (dataset, mode), renders a MiB/s
matrix over (io_threads x queue_depth). Legs run with
OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1 get their own section rather than sharing
those cells, and stay out of the plot. When gds legs are present among the
records, also renders a GDS-vs-OpenDS table per (queue_depth, io_threads)
point. Writes report.md and sweep.csv (all backends) to the output dir, plus
report.png unless --no-plot (needs matplotlib).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _helpers import ROOT, iter_history, ok

# Ordinal blue ramp endpoints (light -> dark = fewest -> most threads),
# validated for the light surface. Chrome tokens from the reference
# palette.
RAMP_LIGHT = (0x86, 0xb6, 0xef)
RAMP_DARK = (0x0d, 0x36, 0x6b)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _records(in_dirs):
    for _, hist in iter_history(in_dirs):
        for line in hist.read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def _collect(in_dirs):
    grids = {}          # (dataset, mode) -> {(io_threads, queue_depth): mib_s}
    aligned = {}        # same, for OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1 legs
    gds = {}            # (dataset, mode) -> mib_s
    rows = []
    threads_seen, depths_seen = set(), set()
    for rec in _records(in_dirs):
        cfg = rec.get("config", {})
        res = rec.get("result", {})
        t, q = cfg.get("io_threads"), cfg.get("queue_depth")
        aa = cfg.get("assume_aligned_only")
        rows.append({"backend": cfg.get("backend"),
                     "dataset": cfg.get("dataset"), "mode": cfg.get("mode"),
                     "io_threads": t, "queue_depth": q,
                     "assume_aligned_only": int(bool(aa)),
                     "mib_s": res.get("mib_s"), "iops": res.get("iops"),
                     "error": rec.get("error")})
        if cfg.get("backend") == "opends":
            # An aligned leg is a different contract, not another cell in the
            # sweep matrix: keep it out of the default grid (and the plot).
            dst = aligned if aa else grids
            dst.setdefault((cfg.get("dataset"), cfg.get("mode")), {})[
                (t, q)] = res.get("mib_s")
            threads_seen.add(t)
            depths_seen.add(q)
        elif cfg.get("backend") == "gds":
            key = (cfg.get("dataset"), cfg.get("mode"))
            if key in gds:
                print(f"warning: multiple gds records for "
                      f"{key[0]}/{key[1]}; keeping the last", file=sys.stderr)
            gds[key] = res.get("mib_s")
    threads = sorted(x for x in threads_seen if x is not None)
    depths = sorted(x for x in depths_seen if x is not None)
    return grids, aligned, gds, rows, threads, depths


def _pivot_md(ds, mode, grid, threads, depths):
    lines = [f"### {ds} / {mode} (MiB/s)", ""]
    lines.append("| io_threads \\ queue_depth | "
                 + " | ".join(str(q) for q in depths) + " |")
    lines.append("|" + "---|" * (len(depths) + 1))
    for t in threads:
        cells = [f"{grid[(t, q)]:.0f}" if grid.get((t, q)) is not None else "-"
                 for q in depths]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _compare_md(grids, gds, threads, depths):
    keys = sorted(grids, key=lambda k: (k[0] or "", k[1] or ""))
    lines = []
    for q in depths:
        for t in threads:
            body = []
            for ds, mode in keys:
                o = grids[(ds, mode)].get((t, q))
                if o is None:
                    continue
                g = gds.get((ds, mode))
                body.append(f"| {ds} | {mode} | "
                            + (f"{g:.0f}" if g else "-")
                            + f" | {o:.0f} | "
                            + (f"{o / g:.2f}" if g else "-") + " |")
            if not body:
                continue
            lines += [f"### queue_depth={q}, io_threads={t}", "",
                      "| Dataset | Mode | GDS (MiB/s) | OpenDS (MiB/s) | "
                      "Speedup |",
                      "|---|---|---|---|---|"]
            lines += body
            lines.append("")
    return lines


def _series(grid_wm):
    out = {}
    for (t, q), v in grid_wm.items():
        if t is not None and q is not None and v is not None:
            out.setdefault(t, {})[q] = v
    return {t: dict(sorted(d.items())) for t, d in out.items()}


def _ramp(threads):
    """Interpolate the blue ramp over the observed io_threads values."""
    n = len(threads)
    out = {}
    for i, t in enumerate(threads):
        f = i / (n - 1) if n > 1 else 1.0
        out[t] = "#" + "".join(f"{round(a + (b - a) * f):02x}"
                               for a, b in zip(RAMP_LIGHT, RAMP_DARK))
    return out


def _plot(grids, threads, depths, dst, spec_mbs=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not installed; skipping plot", file=sys.stderr)
        return False

    ramp = _ramp(threads)
    spec = spec_mbs * 1e6 / 2**20 if spec_mbs else None
    peak = max((v for g in grids.values() for v in g.values()
                if v is not None), default=0)
    top = 1.07 * max(peak, spec or 0) or 1000

    workloads = sorted({wl for wl, _ in grids})
    modes = ["async", "sync"]

    fig, axes = plt.subplots(
        len(modes), len(workloads),
        figsize=(13, 5.8), dpi=160,
        sharex=True, sharey=True, squeeze=False,
        facecolor=SURFACE,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.11,
                        hspace=0.22, wspace=0.10)

    for r, mode in enumerate(modes):
        for c, wl in enumerate(workloads):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log", base=2)
            ax.set_ylim(0, top)
            ax.set_xlim(0.85, 620)

            ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(BASELINE)
                ax.spines[side].set_linewidth(0.8)
            ax.tick_params(colors=INK_MUTED, labelsize=8, length=3,
                           width=0.8, labelcolor=INK_MUTED)
            ax.set_xticks([1, 8, 64, 512])
            ax.set_xticks(depths, minor=True)
            ax.xaxis.set_major_formatter(lambda v, _: f"{int(v)}")
            ax.xaxis.set_minor_formatter(lambda v, _: "")
            ax.yaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 5, 10]))

            if spec is not None:
                ax.axhline(spec, color=BASELINE, linewidth=0.8, zorder=1)

            for th, series in sorted(_series(grids.get((wl, mode), {})).items()):
                ax.plot(
                    list(series), list(series.values()),
                    color=ramp.get(th, INK_MUTED), linewidth=1.6,
                    marker="o", markersize=4.2,
                    markeredgecolor=SURFACE, markeredgewidth=0.7,
                    zorder=3,
                )

            if r == 0:
                ax.set_title(wl, fontsize=10.5, color=INK_PRIMARY, pad=8)
            if c == 0:
                ax.set_ylabel(f"{mode}\nMiB/s", fontsize=9,
                              color=INK_SECONDARY, labelpad=6)
            if r == len(modes) - 1:
                ax.set_xlabel("queue depth", fontsize=8.5, color=INK_MUTED)

    if spec is not None:
        axes[0][0].annotate(
            f"device spec seq. read ({spec_mbs:g} MB/s)",
            xy=(1, spec), xytext=(0, 4), textcoords="offset points",
            fontsize=7.5, color=INK_MUTED, va="bottom",
        )

    fig.suptitle("NVMe read throughput sweep", x=0.075, y=0.975,
                 ha="left", fontsize=14, color=INK_PRIMARY)
    fig.text(0.075, 0.905,
             "MiB/s by queue depth and io_threads, async vs sync, per workload",
             fontsize=9.5, color=INK_SECONDARY)

    handles = [
        Line2D([], [], color=ramp[th], linewidth=1.6, marker="o",
               markersize=4.2, markeredgecolor=SURFACE, markeredgewidth=0.7,
               label=str(th))
        for th in threads
    ]
    fig.legend(
        handles=handles, title="io_threads",
        loc="upper right", bbox_to_anchor=(0.985, 0.97),
        ncol=4, frameon=False,
        fontsize=8.5, title_fontsize=8.5,
        labelcolor=INK_SECONDARY, alignment="right",
    ).get_title().set_color(INK_SECONDARY)

    fig.savefig(dst, facecolor=SURFACE)
    return True


def _resolve(dirs):
    return [d if d.is_absolute() else ROOT / d for d in (Path(x) for x in dirs)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dirs", action="append", metavar="DIR",
                    help="Dir to scan for history.jsonl (recursive). "
                         "Repeatable. Default cijoe-output-bench.")
    ap.add_argument("--out", metavar="DIR",
                    help="Output dir for report.md/sweep.csv/report.png. "
                         "Default: first --in dir.")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--spec-mbs", type=float, metavar="MBS",
                    help="Device spec sequential-read MB/s; draws a reference "
                         "line in the plot. Default: no line.")
    a = ap.parse_args()

    in_dirs = _resolve(a.in_dirs or ["cijoe-output-bench"])
    out_dir = (Path(a.out) if a.out else in_dirs[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    grids, aligned, gds, rows, threads, depths = _collect(in_dirs)
    if not rows:
        sys.exit("no history.jsonl records found under: "
                 + ", ".join(str(d) for d in in_dirs))

    md = []
    for (ds, mode), grid in sorted(grids.items(),
                                   key=lambda kv: (kv[0][0] or "",
                                                   kv[0][1] or "")):
        md += _pivot_md(ds, mode, grid, threads, depths)
    if aligned:
        md += ["## Aligned-only (OPENDS_AISIO_ASSUME_ALIGNED_ONLY=1)",
               "",
               "Datasets whose files are not LBA-multiples fail by "
               "construction and show as `-`.", ""]
        for (ds, mode), grid in sorted(aligned.items(),
                                       key=lambda kv: (kv[0][0] or "",
                                                       kv[0][1] or "")):
            md += _pivot_md(ds, mode, grid, threads, depths)
    if gds:
        md += ["## GDS vs OpenDS", ""]
        md += _compare_md(grids, gds, threads, depths)
    report = out_dir / "report.md"
    report.write_text("\n".join(md) + "\n")

    csv_path = out_dir / "sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["backend", "dataset", "mode",
                                          "io_threads", "queue_depth",
                                          "assume_aligned_only",
                                          "mib_s", "iops", "error"])
        w.writeheader()
        w.writerows(rows)

    outs = [report, csv_path]
    if not a.no_plot and grids:
        png = out_dir / "report.png"
        if _plot(grids, threads, depths, png, a.spec_mbs):
            outs.append(png)

    print("wrote " + ", ".join(str(p) for p in outs))
    ok("report")


if __name__ == "__main__":
    main()
