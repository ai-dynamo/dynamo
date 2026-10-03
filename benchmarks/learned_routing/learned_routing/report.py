# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tables and figures from ``results.jsonl`` files, with bootstrap confidence intervals.

**Normalization.** For each (policy, cell, k), ``ratio = m(policy) / m(reference)`` with the
reference evaluated on the same cell content and replicate (same workload, same policy seed). Pairs
whose reference value is 0 or missing are dropped and counted.

**Cell content.** A ``cell_id`` names a cell, but a record's numbers depend on its content: the
cell content SHA (load, SLA, measurement rule, engine overrides, trace), the bindings build, the
replicate protocol and the harness version (:data:`CONTENT_KEYS`). Pairing matches all of them, so
a policy is never normalized by a reference run on another load, SLA, engine or build. Inputs that
hold one ``cell_id`` under several contents (a calibration sweep, a timing perturbation, results
spanning a rebuild) are rejected with :class:`MixedContentError` unless the caller selects one
build (``build_id``) or asks to treat each content as its own cell (``contents="split"``, which
relabels such cells ``<cell_id>@<content digest>``).

**Bootstrap (two-stage cluster).** One draw resamples clusters with replacement (cells by default,
or independent workload segments with ``cluster="segment"``, LR-11), then resamples each chosen
cell's replicates with replacement; the statistic is the mean over cells of the mean over
replicates. The CI is the percentile interval of ``n_boot`` draws.

Every record is deduplicated by ``cache_key`` (last error-free record wins).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

GROUP_KEYS = ("split", "holdout_axis", "num_workers", "family", "load_mode")
# Fields that, with cell_id and repeat, identify what a record measured.
CONTENT_KEYS = ("cell_sha", "build_id", "replicate_protocol", "harness_version")
CONTENT_MODES = ("error", "split")
# Reference palette (dataviz skill, light mode), fixed order; never cycled.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def load_records(paths: Iterable[str | Path]) -> list[dict]:
    by_key: dict[str, dict] = {}
    errors: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("cache_key")
            if record.get("error") or key is None:
                errors.append(record)
                continue
            by_key[key] = record
    out = list(by_key.values())
    for record in errors:
        record["_error"] = True
    return out + errors


def iqm(values: Sequence[float]) -> float | None:
    """Interquartile mean (Agarwal et al. 2021): mean of the middle 50%."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    low, high = int(math.floor(0.25 * n)), int(math.ceil(0.75 * n))
    middle = ordered[low:high] or ordered
    return sum(middle) / len(middle)


def geomean(values: Sequence[float]) -> float | None:
    positive = [v for v in values if v > 0]
    if not positive or len(positive) != len(values):
        return None
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def bootstrap_ci(
    cells: dict[str, list[float]],
    cluster_of: dict[str, str] | None = None,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """(point, lo, hi) of the mean over cells of per-cell replicate means."""
    cells = {c: v for c, v in cells.items() if v}
    if not cells:
        return None, None, None
    point = sum(sum(v) / len(v) for v in cells.values()) / len(cells)
    clusters: dict[str, list[str]] = defaultdict(list)
    for cell in sorted(cells):
        clusters[(cluster_of or {}).get(cell, cell)].append(cell)
    names = sorted(clusters)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        chosen = [
            cell for name in rng.choices(names, k=len(names)) for cell in clusters[name]
        ]
        means = []
        for cell in chosen:
            values = cells[cell]
            sample = rng.choices(values, k=len(values))
            means.append(sum(sample) / len(sample))
        draws.append(sum(means) / len(means))
    draws.sort()
    lo = draws[int(math.floor(alpha / 2 * (n_boot - 1)))]
    hi = draws[int(math.ceil((1 - alpha / 2) * (n_boot - 1)))]
    return point, lo, hi


def _metric(record: dict, metric: str, scale: str | None = None):
    if scale is None:
        return record.get(metric)
    return ((record.get("rescore") or {}).get(scale) or {}).get(metric)


def pick_reference(records: Sequence[dict], name: str | None) -> str | None:
    """policy_sha of the reference: by name, else default@defaults, else None."""
    from learned_routing.policy import default_reference

    if name:
        for record in records:
            if record.get("policy_name") == name or record.get(
                "policy_sha", ""
            ).startswith(name):
                return record["policy_sha"]
        raise ValueError(f"reference {name!r} not found among the records")
    default_sha = default_reference().sha
    if any(record.get("policy_sha") == default_sha for record in records):
        return default_sha
    return None


class MixedContentError(ValueError):
    """One cell_id carries several contents (cell SHA, build, protocol or harness version)."""


def content_of(record: dict) -> tuple:
    return tuple(record.get(key) for key in CONTENT_KEYS)


def _pair_key(record: dict) -> tuple:
    return (record["cell_id"], record["repeat"], *content_of(record))


def mixed_contents(records: Iterable[dict]) -> dict[str, list[dict]]:
    """cell_id -> its distinct contents, for every cell_id with more than one."""
    seen: dict[str, dict[tuple, None]] = defaultdict(dict)
    for record in records:
        if not record.get("_error"):
            seen[record["cell_id"]][content_of(record)] = None
    return {
        cell: [dict(zip(CONTENT_KEYS, content)) for content in contents]
        for cell, contents in sorted(seen.items())
        if len(contents) > 1
    }


def select_contents(
    records: Sequence[dict], *, build_id: str | None = None, contents: str = "error"
) -> list[dict]:
    """Records restricted to one build, with every cell_id naming one content.

    ``build_id`` keeps records whose build ID starts with it (errored records without one are
    kept). ``contents="error"`` raises :class:`MixedContentError` if a cell_id still has several
    contents; ``"split"`` relabels each content of such a cell ``<cell_id>@<digest>``.
    """
    if contents not in CONTENT_MODES:
        raise ValueError(f"contents must be one of {CONTENT_MODES}, got {contents!r}")
    if build_id:
        records = [
            record
            for record in records
            if str(record.get("build_id") or "").startswith(build_id)
            or (record.get("_error") and record.get("build_id") is None)
        ]
    mixed = mixed_contents(records)
    if not mixed:
        return list(records)
    if contents == "error":
        detail = "; ".join(
            f"{cell}: {len(variants)} contents"
            + "".join(
                f" [{', '.join(f'{k}={str(v)[:12]}' for k, v in variant.items())}]"
                for variant in variants
            )
            for cell, variants in list(mixed.items())[:5]
        )
        raise MixedContentError(
            f"{len(mixed)} cell_id(s) carry several contents, so paired ratios would mix "
            f"loads, SLAs, engines or builds: {detail}. Select one build (--build-id) or "
            "treat each content as its own cell (--mixed-contents split)."
        )
    out = []
    for record in records:
        if record["cell_id"] in mixed and not record.get("_error"):
            digest = hashlib.sha256(
                json.dumps(content_of(record)).encode()
            ).hexdigest()[:10]
            record = {
                **record,
                "cell_id": f"{record['cell_id']}@{digest}",
                "cell_id_base": record["cell_id"],
            }
        out.append(record)
    return out


def paired_ratios(
    records: Sequence[dict], reference_sha: str, metric: str, scale: str | None = None
) -> tuple[dict, dict]:
    """ratios[(policy_sha, cell_id)] -> list over k, and drop counts per policy.

    A record pairs with the reference record of the same cell_id, repeat and content
    (:data:`CONTENT_KEYS`); a record without one is dropped and counted.
    """
    ref: dict[tuple, float] = {}
    for record in records:
        if record.get("policy_sha") == reference_sha and not record.get("_error"):
            ref[_pair_key(record)] = _metric(record, metric, scale)
    ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    dropped: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("_error"):
            continue
        base = ref.get(_pair_key(record))
        value = _metric(record, metric, scale)
        if base is None or value is None or base <= 0:
            dropped[record["policy_sha"]] += 1
            continue
        ratios[(record["policy_sha"], record["cell_id"])].append(value / base)
    return ratios, dropped


class Report:
    def __init__(
        self,
        records: Sequence[dict],
        *,
        metric: str = "goodput_rps_window",
        reference: str | None = None,
        cluster: str = "cell",
        n_boot: int = 2000,
        seed: int = 0,
        group_keys: Sequence[str] = GROUP_KEYS,
        build_id: str | None = None,
        contents: str = "error",
    ):
        records = select_contents(records, build_id=build_id, contents=contents)
        self.all_records = list(records)
        self.records = [r for r in records if not r.get("_error")]
        self.errors = [r for r in records if r.get("_error")]
        self.metric = metric
        self.reference_sha = pick_reference(self.records, reference)
        self.cluster = cluster
        self.n_boot = n_boot
        self.seed = seed
        self.group_keys = list(group_keys)
        self.names: dict[str, str] = {}
        for record in self.records:
            self.names.setdefault(
                record["policy_sha"],
                record.get("policy_name") or record["policy_sha"][:12],
            )
        self.cell_info: dict[str, dict] = {}
        for record in self.records:
            self.cell_info.setdefault(
                record["cell_id"],
                {
                    key: record.get(key)
                    for key in (*GROUP_KEYS, "segment", "trace_content_sha256")
                },
            )

    def policies(self) -> list[str]:
        return sorted(self.names, key=lambda sha: self.names[sha])

    def cluster_of(self) -> dict[str, str] | None:
        if self.cluster == "cell":
            return None
        return {
            cell: str(info.get("segment") or info.get("trace_content_sha256") or cell)
            for cell, info in self.cell_info.items()
        }

    def summarize(
        self, cells: set[str] | None = None, scale: str | None = None
    ) -> list[dict]:
        rows = []
        cluster = self.cluster_of()
        if self.reference_sha is not None:
            ratios, dropped = paired_ratios(
                self.records, self.reference_sha, self.metric, scale
            )
        for sha in self.policies():
            raw: dict[str, list[float]] = defaultdict(list)
            for record in self.records:
                if record["policy_sha"] != sha or (
                    cells is not None and record["cell_id"] not in cells
                ):
                    continue
                value = _metric(record, self.metric, scale)
                if value is not None:
                    raw[record["cell_id"]].append(value)
            if not raw:
                continue
            row = {"policy": self.names[sha], "policy_sha": sha, "cells": len(raw)}
            row["replicates"] = sum(len(v) for v in raw.values())
            point, lo, hi = bootstrap_ci(
                raw, cluster, n_boot=self.n_boot, seed=self.seed
            )
            row.update(raw_mean=point, raw_lo=lo, raw_hi=hi)
            if self.reference_sha is not None:
                per_cell = {
                    cell: values
                    for (policy, cell), values in ratios.items()
                    if policy == sha and (cells is None or cell in cells)
                }
                point, lo, hi = bootstrap_ci(
                    per_cell, cluster, n_boot=self.n_boot, seed=self.seed
                )
                cell_means = [sum(v) / len(v) for v in per_cell.values()]
                row.update(
                    norm_mean=point,
                    norm_lo=lo,
                    norm_hi=hi,
                    norm_iqm=iqm(cell_means),
                    norm_geomean=geomean(cell_means),
                    wins=sum(m > 1.0 for m in cell_means),
                    losses=sum(m < 1.0 for m in cell_means),
                    dropped_pairs=dropped.get(sha, 0),
                )
            rows.append(row)
        return rows

    def grouped(self, key: str) -> list[dict]:
        groups: dict[str, set[str]] = defaultdict(set)
        for cell, info in self.cell_info.items():
            groups[str(info.get(key))].add(cell)
        rows = []
        for value in sorted(groups, key=lambda v: (len(v), v)):
            for row in self.summarize(groups[value]):
                rows.append({key: value, **row})
        return rows

    def guards(self) -> list[dict]:
        fields = (
            ("worker_share_max", "guards.worker_share_max"),
            ("session_split_frac", "guards.session_split_frac"),
            ("slowdown_clip_mean", "guards.slowdown_clip_mean"),
            ("frac_slowdown_gt_3S", "guards.frac_slowdown_gt_3S"),
            ("ttft_p90", "ttft_p90"),
            ("itl_p90", "itl_p90"),
            ("prefix_reuse", "prefix_reuse"),
        )
        rows = []
        for sha in self.policies():
            row = {"policy": self.names[sha]}
            for label, path in fields:
                values = []
                for record in self.records:
                    if record["policy_sha"] != sha:
                        continue
                    node = record
                    for part in path.split("."):
                        node = node.get(part) if isinstance(node, dict) else None
                    if isinstance(node, (int, float)):
                        values.append(float(node))
                row[label] = sum(values) / len(values) if values else None
            rows.append(row)
        return rows

    def winners(self) -> list[dict]:
        means: dict[str, dict[str, float]] = defaultdict(dict)
        for sha in self.policies():
            per_cell: dict[str, list[float]] = defaultdict(list)
            for record in self.records:
                if record["policy_sha"] == sha and record.get(self.metric) is not None:
                    per_cell[record["cell_id"]].append(record[self.metric])
            for cell, values in per_cell.items():
                means[cell][sha] = sum(values) / len(values)
        rows = []
        for cell in sorted(means):
            ranked = sorted(means[cell].items(), key=lambda item: -item[1])
            best, value = ranked[0]
            row = {"cell_id": cell, "winner": self.names[best], "value": value}
            if len(ranked) > 1:
                row["runner_up"] = self.names[ranked[1][0]]
                row["margin"] = value / ranked[1][1] - 1.0 if ranked[1][1] > 0 else None
            rows.append(row)
        return rows

    def scale_sweep(self) -> list[dict]:
        from learned_routing.goodput import SCALES

        rows = []
        for scale in SCALES:
            for row in self.summarize(scale=repr(scale)):
                rows.append({"scale": scale, **row})
        return rows

    def isl_buckets(self) -> list[dict]:
        totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for record in self.records:
            for bucket, stats in (record.get("isl_buckets") or {}).items():
                if stats.get("good_frac_window") is not None:
                    totals[(record["policy_sha"], bucket)].append(
                        stats["good_frac_window"]
                    )
        rows = []
        for (sha, bucket), values in sorted(
            totals.items(), key=lambda i: (self.names[i[0][0]], i[0][1])
        ):
            rows.append(
                {
                    "policy": self.names[sha],
                    "isl_bucket": bucket,
                    "good_frac_window": sum(values) / len(values),
                    "n": len(values),
                }
            )
        return rows


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    if not rows:
        return "_no data_\n"
    head = (
        "| " + " | ".join(columns) + " |\n|" + "|".join("---" for _ in columns) + "|\n"
    )
    body = "".join(
        "| " + " | ".join(_fmt(row.get(c)) for c in columns) + " |\n" for row in rows
    )
    return head + body


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def figures(report: Report, out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    rows = report.summarize()
    key, lo_key, hi_key, label = "raw_mean", "raw_lo", "raw_hi", report.metric
    if report.reference_sha is not None:
        key, lo_key, hi_key = "norm_mean", "norm_lo", "norm_hi"
        label = f"{report.metric} / reference (paired)"
    rows = [r for r in rows if r.get(key) is not None]
    if rows:
        rows.sort(key=lambda r: r[key])
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1.4))
        ys = range(len(rows))
        ax.barh(list(ys), [r[key] for r in rows], height=0.6, color=SERIES[0])
        ax.errorbar(
            [r[key] for r in rows],
            list(ys),
            xerr=[
                [r[key] - r[lo_key] for r in rows],
                [r[hi_key] - r[key] for r in rows],
            ],
            fmt="none",
            ecolor=TEXT,
            elinewidth=1.2,
            capsize=3,
        )
        if report.reference_sha is not None:
            ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_yticks(list(ys), [r["policy"] for r in rows], color=TEXT)
        ax.set_xlabel(label + " (95% bootstrap CI)", color=TEXT)
        _style(ax)
        fig.tight_layout()
        path = out_dir / "policy_summary.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(str(path))

    by_n = [r for r in report.grouped("num_workers") if r.get(key) is not None]
    policies = sorted({r["policy"] for r in by_n})
    if by_n and len({r["num_workers"] for r in by_n}) > 1:
        shown = policies[: len(SERIES)]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for color, policy in zip(SERIES, shown):
            pts = sorted(
                (
                    (float(r["num_workers"]), r[key], r[lo_key], r[hi_key])
                    for r in by_n
                    if r["policy"] == policy and r["num_workers"] != "None"
                ),
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ax.plot(
                xs,
                [p[1] for p in pts],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                label=policy,
            )
            ax.fill_between(
                xs,
                [p[2] for p in pts],
                [p[3] for p in pts],
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xscale("log", base=2)
        ticks = sorted(
            {float(r["num_workers"]) for r in by_n if r["num_workers"] != "None"}
        )
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("num_workers N", color=TEXT)
        ax.set_ylabel(label, color=TEXT)
        if report.reference_sha is not None:
            ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        fig.tight_layout()
        path = out_dir / "by_num_workers.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(str(path))

    sweep = [r for r in report.scale_sweep() if r.get(key) is not None]
    if sweep:
        shown = sorted({r["policy"] for r in sweep})[: len(SERIES)]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for color, policy in zip(SERIES, shown):
            pts = sorted((r["scale"], r[key]) for r in sweep if r["policy"] == policy)
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                label=policy,
            )
        ax.set_xscale("log", base=2)
        scales = sorted({r["scale"] for r in sweep})
        ax.set_xticks(scales, [f"{t:g}" for t in scales])
        ax.minorticks_off()
        ax.set_xlabel("SLO scale x (I, S)", color=TEXT)
        ax.set_ylabel(label, color=TEXT)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        fig.tight_layout()
        path = out_dir / "slo_scale_sweep.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(str(path))
    return made


def write_report(
    report: Report,
    out_dir: Path,
    *,
    title: str = "Learned-routing results",
    make_figures: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = report.summarize()
    grouped = {key: report.grouped(key) for key in report.group_keys}
    sweep = report.scale_sweep()
    guards = report.guards()
    winners = report.winners()
    buckets = report.isl_buckets()
    figs = figures(report, out_dir / "fig") if make_figures else []
    ref_name = (
        report.names.get(report.reference_sha, None) if report.reference_sha else None
    )
    norm_cols = [
        "policy",
        "cells",
        "replicates",
        "norm_mean",
        "norm_lo",
        "norm_hi",
        "norm_iqm",
        "norm_geomean",
        "wins",
        "losses",
        "dropped_pairs",
    ]
    raw_cols = ["policy", "cells", "replicates", "raw_mean", "raw_lo", "raw_hi"]
    lines = [
        f"# {title}\n",
        f"- Records: {len(report.records)} error-free, {len(report.errors)} errored.",
        f"- Metric: `{report.metric}`. Reference: {ref_name or 'none (raw values only)'}.",
        f"- Bootstrap: {report.n_boot} two-stage draws over {report.cluster}s then replicates, 95% percentile CI, seed {report.seed}.",
        "- Simulated results (AIS-timed offline replay), not measurements of a live deployment.\n",
        "## Per policy\n",
        markdown_table(summary, norm_cols if report.reference_sha else raw_cols),
        "\n## Raw metric\n",
        markdown_table(summary, raw_cols),
    ]
    for key, rows in grouped.items():
        lines += [
            f"\n## By {key}\n",
            markdown_table(
                rows, [key, *(norm_cols if report.reference_sha else raw_cols)]
            ),
        ]
    lines += [
        "\n## SLO-scale sweep (A2: I and S scaled together)\n",
        markdown_table(
            sweep, ["scale", *(norm_cols[:6] if report.reference_sha else raw_cols)]
        ),
        "\n## Guard metrics (LR-13; means over evaluations)\n",
        markdown_table(
            guards,
            [
                "policy",
                "worker_share_max",
                "session_split_frac",
                "slowdown_clip_mean",
                "frac_slowdown_gt_3S",
                "ttft_p90",
                "itl_p90",
                "prefix_reuse",
            ],
        ),
        "\n## Good fraction by ISL bucket (LR-08)\n",
        markdown_table(buckets, ["policy", "isl_bucket", "good_frac_window", "n"]),
        "\n## Per-cell winner\n",
        markdown_table(winners, ["cell_id", "winner", "value", "runner_up", "margin"]),
    ]
    if figs:
        lines.append("\n## Figures\n")
        lines += [f"![{Path(f).stem}](fig/{Path(f).name})\n" for f in figs]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "metric": report.metric,
        "reference_sha": report.reference_sha,
        "reference": ref_name,
        "cluster": report.cluster,
        "n_boot": report.n_boot,
        "records": len(report.records),
        "errors": len(report.errors),
        "summary": summary,
        "grouped": grouped,
        "scale_sweep": sweep,
        "guards": guards,
        "isl_buckets": buckets,
        "winners": winners,
        "figures": figs,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n"
    )
    return payload
