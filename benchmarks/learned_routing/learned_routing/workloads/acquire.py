# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acquire every campaign trace into ``CR/traces/<family>/`` and write ``CR/traces/MANIFEST.json``.

Usage::

    python -m learned_routing.workloads.acquire --campaign-root CR [--skills-traces DIR]
        [--fast25-local DIR] [--download]

Steps (idempotent; an existing file is reused only if its SHA-256 matches, never overwritten):

1. ``mooncake`` and ``toolagent``: copy from a local trace corpus (``--skills-traces``) and check
   the SHA-256 its index ``_shared/benchmark-traces.md`` lists (pinned in ``SHARED``).
2. ``fast25``: Mooncake's FAST25 ``conversation_trace`` and ``synthetic_trace``. A local copy is used
   when one exists (pass ``--fast25-local``); otherwise they are fetched from GitHub at a pinned
   commit (``--download``). Both are checked against pinned SHA-256s.
3. ``agentx``: the Weka corpus at a pinned Hugging Face revision (``--download`` fetches it when it
   is absent), checked for size and SHA-256. Every complete play whose requests all satisfy
   ``in + max(out, 1) <= 131072`` is written unchanged to ``agentx/plays/<row>-<id>.json``.
4. ``synthetic_sessions``: seeded multi-turn session traces from :mod:`.synthetic`, one per segment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from . import agentx
from .common import (
    MAX_MODEL_LEN,
    campaign_root,
    read_jsonl,
    sha256_file,
    write_atomic,
    write_json,
)
from .synthetic import GENERATOR_VERSION, SessionSpec
from .synthetic import write as write_sessions
from .transform import mooncake_stats, nested_timestamps_relative

SHARED = {
    # name: (family, sha256 from _shared/benchmark-traces.md, rows)
    "mooncake_trace.jsonl": (
        "mooncake",
        "b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711",
        23608,
    ),
    "toolagent_trace.jsonl": (
        "toolagent",
        "48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71",
        23608,
    ),
}
FAST25_COMMIT = "0d1a8040faebb7c127c8901840a38c2ff57e80c5"
FAST25_URL = "https://raw.githubusercontent.com/kvcache-ai/Mooncake/{commit}/FAST25-release/traces/{name}"
FAST25 = {
    "conversation_trace.jsonl": "b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df",
    "synthetic_trace.jsonl": "bd070915a98fc0ed264d7cfef2ce746002eb3076a695ec31ba2674c0111ec131",
}
# Synthetic-session segments: one independent draw per seed (train 0-3, val 4-5, test 6-9).
SYNTHETIC_SEEDS = tuple(range(10))
SYNTHETIC_BASE = {"duration_s": 720.0, "session_rate": 1.0}

ROLE = {
    "mooncake": "primary flat family; time windows are independent segments across train/val/test",
    "toolagent": (
        "excluded from every split: a relabeled copy of mooncake_trace (identical OSL on 23,608/23,608 rows, "
        "0-conflict hash bijection over 409,356 aligned blocks, reuse correlation 0.9998, timestamps x0.9825); "
        "audits/setup/determinism-cost-context-r1.md F1"
    ),
    "fast25_conversation": (
        "test-only held-out flat family (different prefix structure: mean reuse 0.384 vs 0.630); shares Mooncake's "
        "arrival grid and (ts, ISL, OSL) skeleton on 91% of rows (audit r1 F1), so each of its windows is aligned to a "
        "Mooncake test window and shares that window's segment id"
    ),
    "fast25_synthetic": (
        "test-only held-out flat family; independent of mooncake/conversation "
        "(runs/build_workloads/out/identity_fast25_synth.json); near-unique timestamps, so crn-order-v1 replicates "
        "are near-degenerate and its noise must come from segments"
    ),
    "agentx": "agentic family; disjoint play subsets across train/val/test (agentic_lanes load)",
    "synthetic_sessions": "multi-turn chat family; one generator seed per segment, seeds disjoint across splits",
}


def _copy_verified(src: Path, dst: Path, sha: str) -> None:
    if dst.exists():
        actual = sha256_file(dst)
        if actual != sha:
            raise SystemExit(
                f"{dst}: SHA-256 {actual} != expected {sha}; refusing to overwrite"
            )
        return
    actual = sha256_file(src)
    if actual != sha:
        raise SystemExit(f"{src}: SHA-256 {actual} != expected {sha}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(0o444)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "-C",
            "-",
            "-sS",
            "-o",
            str(part),
            url,
        ],
        check=True,
    )
    part.rename(dst)


def _flat_entry(cr: Path, path: Path, family: str, provenance: dict, role: str) -> dict:
    rows = read_jsonl(path)
    return {
        "path": str(path.relative_to(cr)),
        "family": family,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "format": "mooncake",
        "block_size": 512,
        "provenance": provenance,
        "role": role,
        "stats": mooncake_stats(rows, 512),
    }


def acquire_flat(
    cr: Path, skills_traces: Path, fast25_local: Path | None, download: bool
) -> list[dict]:
    entries = []
    for name, (family, sha, rows) in SHARED.items():
        dst = cr / "traces" / family / name
        _copy_verified(skills_traces / name, dst, sha)
        entry = _flat_entry(
            cr,
            dst,
            family,
            {
                "source": str(skills_traces / name),
                "contract": "_shared/benchmark-traces.md",
                "expected_sha256": sha,
                "upstream": "kvcache-ai/Mooncake (toolagent: FAST25-release/traces/toolagent_trace.jsonl, identical SHA)",
            },
            ROLE[family],
        )
        if entry["rows"] != rows:
            raise SystemExit(f"{dst}: {entry['rows']} rows, expected {rows}")
        entries.append(entry)
    for name, sha in FAST25.items():
        dst = cr / "traces" / "fast25" / name
        family = "fast25_" + name.split("_")[0]
        url = FAST25_URL.format(commit=FAST25_COMMIT, name=name)
        local = fast25_local / name if fast25_local is not None else None
        if not dst.exists():
            if local is not None and local.exists():
                _copy_verified(local, dst, sha)
            elif download:
                _download(url, dst)
                dst.chmod(0o444)
            else:
                raise SystemExit(f"{dst} missing: pass --fast25-local or --download")
        if sha256_file(dst) != sha:
            raise SystemExit(f"{dst}: SHA-256 mismatch (expected {sha})")
        entries.append(
            _flat_entry(
                cr,
                dst,
                family,
                {
                    "upstream_url": url,
                    "upstream_commit": FAST25_COMMIT,
                    "expected_sha256": sha,
                    "local_copy_used": str(local)
                    if local is not None and local.exists()
                    else None,
                },
                ROLE[family],
            )
        )
    return entries


def acquire_agentx(
    cr: Path, download: bool, limit: int = MAX_MODEL_LEN
) -> tuple[list[dict], dict]:
    root = cr / "traces" / "agentx"
    source = root / "source" / "traces.jsonl"
    if not source.exists():
        if not download:
            raise SystemExit(f"{source} missing: pass --download")
        _download(agentx.SOURCE_URL, source)
        source.chmod(0o444)
    size = source.stat().st_size
    if size != agentx.SOURCE_BYTES:
        raise SystemExit(f"{source}: {size} bytes, expected {agentx.SOURCE_BYTES}")
    if sha256_file(source) != agentx.SOURCE_SHA256:
        raise SystemExit(
            f"{source}: SHA-256 mismatch (expected {agentx.SOURCE_SHA256})"
        )

    plays_dir = root / "plays"
    plays_dir.mkdir(parents=True, exist_ok=True)
    selected, totals = [], {"plays": 0, "requests": 0}
    with source.open("rb") as handle:
        for row_index, raw in enumerate(handle):
            play = agentx.parse_play(raw)
            totals["plays"] += 1
            totals["requests"] += sum(1 for _ in agentx.iter_requests(play["requests"]))
            if not agentx.play_fits(play, limit):
                continue
            name = agentx.play_file_name(row_index, play)
            path = plays_dir / name
            line_sha = agentx.line_sha256(raw)
            if path.exists():
                if sha256_file(path) != line_sha:
                    raise SystemExit(
                        f"{path}: content differs from source row {row_index}; refusing to overwrite"
                    )
            else:
                write_atomic(path, raw)
                path.chmod(0o444)
            stats = agentx.play_stats(play)
            stats["nested_timestamps_relative"] = nested_timestamps_relative(play)
            selected.append(
                {
                    "path": str(path.relative_to(cr)),
                    "family": "agentx",
                    "sha256": line_sha,
                    "bytes": len(raw),
                    "rows": stats["requests"],
                    "format": "weka",
                    "block_size": stats["block_size"],
                    "provenance": {
                        "source_row_index": row_index,
                        "source_line_sha256": line_sha,
                        "dataset": agentx.SOURCE_DATASET,
                        "revision": agentx.SOURCE_REVISION,
                    },
                    "role": ROLE["agentx"],
                    "stats": stats,
                }
            )
    summary = {
        "source": {
            "path": str(source.relative_to(cr)),
            "url": agentx.SOURCE_URL,
            "dataset": agentx.SOURCE_DATASET,
            "revision": agentx.SOURCE_REVISION,
            "bytes": size,
            "sha256": agentx.SOURCE_SHA256,
            "plays": totals["plays"],
            "requests": totals["requests"],
        },
        "selection": {
            "predicate": f"every normal/streaming request, recursively including subagents, satisfies in + max(out, 1) <= {limit}; whole play kept unchanged",
            "method_reference": "learned_routing/workloads/agentx.py",
            "clipping": False,
            "scaling": False,
            "fallback_32k_needed": False,
            "plays": len(selected),
            "requests": sum(e["rows"] for e in selected),
            "explicit_subagent_groups": sum(
                e["stats"]["explicit_subagent_groups"] for e in selected
            ),
            "single_model_plays": sum(len(e["stats"]["models"]) == 1 for e in selected),
            "relative_nested_timestamp_plays": sum(
                e["stats"]["nested_timestamps_relative"] for e in selected
            ),
        },
    }
    return selected, summary


def acquire_synthetic(cr: Path) -> list[dict]:
    entries = []
    for seed in SYNTHETIC_SEEDS:
        spec = SessionSpec.from_dict({**SYNTHETIC_BASE, "seed": seed})
        path = cr / "traces" / "synthetic_sessions" / f"sessions_seed{seed}.jsonl"
        sha = write_sessions(spec, path)
        rows = read_jsonl(path)
        entries.append(
            {
                "path": str(path.relative_to(cr)),
                "family": "synthetic_sessions",
                "sha256": sha,
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "format": "mooncake",
                "block_size": spec.block_size,
                "provenance": {
                    "generator": "learned_routing.workloads.synthetic",
                    "generator_version": GENERATOR_VERSION,
                    "spec": spec.to_dict(),
                    "spec_key": spec.key(),
                },
                "role": ROLE["synthetic_sessions"],
                "stats": mooncake_stats(rows, spec.block_size),
            }
        )
    return entries


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Acquire campaign traces and write CR/traces/MANIFEST.json."
    )
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument(
        "--skills-traces",
        type=Path,
        required=True,
        help="directory holding mooncake_trace.jsonl and toolagent_trace.jsonl",
    )
    parser.add_argument(
        "--fast25-local",
        type=Path,
        default=None,
        help="directory holding local FAST25 copies",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow network downloads for missing sources",
    )
    args = parser.parse_args(argv)
    cr = campaign_root(args.campaign_root)

    flat = acquire_flat(cr, args.skills_traces, args.fast25_local, args.download)
    plays, agentx_summary = acquire_agentx(cr, args.download)
    sessions = acquire_synthetic(cr)
    manifest = {
        "schema": "learned-routing.traces-manifest.v1",
        "campaign_root": str(cr),
        "paths_relative_to": "campaign_root",
        "max_model_len": MAX_MODEL_LEN,
        "roles": ROLE,
        "agentx": agentx_summary,
        "files": flat + plays + sessions,
    }
    write_json(cr / "traces" / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "agentx": agentx_summary["selection"],
                "synthetic_rows": [e["rows"] for e in sessions],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
