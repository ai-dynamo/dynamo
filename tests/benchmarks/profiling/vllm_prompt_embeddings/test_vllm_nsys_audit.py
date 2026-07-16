# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from benchmarks.profiling.vllm_prompt_embeddings.audit_nsys import audit

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def write_fixture(
    root: Path,
    *,
    graph_prefill: bool = True,
    graph_decode: bool = True,
    include_decode: bool = True,
) -> tuple[Path, Path, Path]:
    rep = root / "trace.nsys-rep"
    rep.write_bytes(b"rep")
    sqlite_path = root / "trace.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        connection.execute(
            "CREATE TABLE NVTX_EVENTS "
            "(start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER)"
        )
        connection.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
            "(start INTEGER, end INTEGER, globalTid INTEGER, nameId INTEGER)"
        )
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)")
        connection.execute("CREATE TABLE CUDA_GRAPH_NODE_EVENTS (id INTEGER)")
        connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [
                (1, "cudaGraphLaunch_v10000"),
                (2, "execute_context_1(515)_generation_0(0)"),
                (3, "execute_context_0(0)_generation_1(1)"),
            ],
        )
        ranges = [(100, 200, 7, None, 2)]
        if include_decode:
            ranges.append((300, 400, 7, None, 3))
        connection.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?)", ranges)
        if graph_prefill:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (150, 151, 7, 1)"
            )
        if graph_decode:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (350, 351, 7, 1)"
            )
        connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1)")
        connection.execute("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (1)")

    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "accepted": True,
                "config": {
                    "requests": 1,
                    "output_tokens": 2,
                    "prompt_tokens": 515,
                    "block_size": 16,
                },
                "resolved_engine": {
                    "cuda_graph_mode": "FULL",
                    "cuda_graph_capture_sizes": [1, 3, 515],
                    "enforce_eager": False,
                    "prefix_caching": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return rep, sqlite_path, summary_path


def test_audit_accepts_graphed_prefill_and_decode(tmp_path: Path) -> None:
    rep, sqlite_path, summary = write_fixture(tmp_path)
    result = audit(rep, sqlite_path, summary)
    assert result["accepted"] is True
    assert result["phase_graphs"]["prefill_graph_launches"] == 1
    assert result["phase_graphs"]["decode_graph_launches"] == 1


@pytest.mark.parametrize(
    ("graph_prefill", "graph_decode", "expected_failure"),
    [
        (False, True, "prefill ranges"),
        (True, False, "decode ranges"),
    ],
)
def test_audit_rejects_phase_without_graph_launch(
    tmp_path: Path,
    graph_prefill: bool,
    graph_decode: bool,
    expected_failure: str,
) -> None:
    rep, sqlite_path, summary = write_fixture(
        tmp_path,
        graph_prefill=graph_prefill,
        graph_decode=graph_decode,
    )
    result = audit(rep, sqlite_path, summary)
    assert result["accepted"] is False
    assert any(expected_failure in failure for failure in result["failures"])


def test_audit_rejects_incomplete_phase_ranges(tmp_path: Path) -> None:
    rep, sqlite_path, summary = write_fixture(tmp_path, include_decode=False)
    result = audit(rep, sqlite_path, summary)
    assert result["accepted"] is False
    assert "decode ranges=0, expected 1" in result["failures"]
