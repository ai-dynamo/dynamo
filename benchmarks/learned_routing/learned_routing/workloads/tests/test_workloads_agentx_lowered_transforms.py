# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Play transforms (think_mult, osl_mult) of the recycled AgentX lowering."""

from __future__ import annotations

import json
import math
from collections import defaultdict

import pytest
from learned_routing.paths import Layout
from learned_routing.workloads import agentx_lowered as ax
from learned_routing.workloads.common import MAX_MODEL_LEN

CR = Layout.resolve().root
REAL = (CR / "cells/SPLIT_MANIFEST.json").exists() and all(
    (CR / f"traces/agentx_lowered/base_{tag}/MANIFEST.json").exists()
    for tag in ("cap300", "cap300_think4", "cap300_osl2.5")
)
needs_real = pytest.mark.skipif(not REAL, reason="campaign base lowerings not built")


def test_from_cell_maps_weka_transforms_and_rejects_the_rest():
    cell = {
        "window": None,
        "plays": ["a.json"],
        "isl_unique_mult": 1.0,
        "isl_prefix_mult": 1.0,
        "osl_mult": 1.0,
        "prefix_root_mult": 1,
        "think_mult": 4.0,
        "think_cap_s": 300.0,
        "seed": 0,
        "max_model_len": MAX_MODEL_LEN,
    }
    transform = ax.PlayTransform.from_cell(cell)
    assert transform == ax.PlayTransform(think_mult=4.0)
    assert ax.weka_dir(CR, transform).name == "weka_cap300_think4"
    # the plain AgentX transform keeps the v1 directories
    assert ax.base_dir(CR, ax.PlayTransform.from_cell({"think_cap_s": 300})).name == (
        "base_cap300"
    )
    assert ax.PlayTransform(think_mult=0.5, osl_mult=1.5).tag() == (
        "cap300_think0.5_osl1.5"
    )
    for bad in (
        {"isl_unique_mult": 1.5},
        {"isl_prefix_mult": 2.0},
        {"prefix_root_mult": 2},
        {"window": [0, 10]},
        {"max_model_len": 32768},
    ):
        with pytest.raises(ValueError):
            ax.PlayTransform.from_cell({**cell, **bad})
    for mult in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            ax.PlayTransform(osl_mult=mult)


def test_identity_spec_key_keeps_its_v1_form():
    """Untransformed specs hash exactly as before multipliers existed (stable trace bytes)."""
    spec = ax.GenSpec(split="train", mode="closed", num_copies=8, seed=1)
    assert spec.key_dict() == {
        "split": "train",
        "mode": "closed",
        "num_copies": 8,
        "seed": 1,
        "play_rate_per_s": None,
        "think_cap_s": ax.DEFAULT_THINK_CAP_S,
        "version": ax.VERSION,
    }
    think = ax.GenSpec(
        split="train", mode="closed", num_copies=8, seed=1, think_mult=2.0
    )
    assert think.key_dict() == {**spec.key_dict(), "think_mult": 2.0}
    assert think.transform == ax.PlayTransform(think_mult=2.0)
    with pytest.raises(ValueError):
        ax.GenSpec(
            split="train", mode="closed", num_copies=8, seed=1, osl_mult=0.0
        ).validate()


B2_TRANSFORMS = (
    ax.PlayTransform(),
    ax.PlayTransform(think_mult=0.5),
    ax.PlayTransform(think_mult=2.0),
    ax.PlayTransform(think_mult=4.0),
    ax.PlayTransform(osl_mult=1.5),
    ax.PlayTransform(osl_mult=2.5),
)


@needs_real
def test_every_b2_agentx_transform_has_a_verified_base():
    for transform in set(B2_TRANSFORMS) | set(ax.cell_transforms(CR)):
        manifest = json.loads(
            (ax.base_dir(CR, transform) / "MANIFEST.json").read_text()
        )
        assert manifest["all_graph_digests_equal"] and len(manifest["plays"]) == 82
        check = manifest["b2_cell_trace_check"]
        assert check["plays_checked"] > 0
        assert check["byte_identical"] == check["plays_checked"]
        assert manifest["think_cap_s"] == transform.think_cap_s
        assert manifest.get("think_mult", 1.0) == transform.think_mult
        assert manifest.get("osl_mult", 1.0) == transform.osl_mult


def _copies(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[":".join(row["request_id"].split(":")[:2])].append(row)
    return groups


def _offsets(rows):
    root = min(r["not_before_ms"] for r in rows if not r.get("dependencies"))
    return {r["request_id"]: r["not_before_ms"] - root for r in rows}


@needs_real
@pytest.mark.parametrize("mode", ["closed", "open"])
def test_think_mult_dilates_every_copy_and_nothing_else(mode):
    rate = 0.01 if mode == "open" else None
    common = dict(split="test", mode=mode, num_copies=40, seed=3, play_rate_per_s=rate)
    _, base_rows, base_meta = ax.generate(ax.GenSpec(**common), CR)
    header, rows, meta = ax.generate(ax.GenSpec(**common, think_mult=4.0), CR)
    assert meta["transform"]["tag"] == "cap300_think4"
    assert meta["spec"]["think_mult"] == 4.0
    assert header["source"] != ax.generate(ax.GenSpec(**common), CR)[0]["source"]
    # same draws, labels, hash ranges and arrivals: the transform changes the plays only
    assert [
        (c["play"], c["label"], c["hash_range"], c["start_ms"]) for c in meta["copies"]
    ] == [
        (c["play"], c["label"], c["hash_range"], c["start_ms"])
        for c in base_meta["copies"]
    ]
    assert [r["request_id"] for r in rows] == [r["request_id"] for r in base_rows]
    plain, dilated = _copies(base_rows), _copies(rows)
    for copy in meta["copies"]:
        a, b = plain[copy["label"]], dilated[copy["label"]]
        for x, y in zip(a, b):
            assert y["hash_ids"] == x["hash_ids"]
            assert y["output_length"] == x["output_length"]
            assert [d["delay_ms"] for d in y.get("dependencies", [])] == [
                4.0 * d["delay_ms"] for d in x.get("dependencies", [])
            ]
        off_a, off_b = _offsets(a), _offsets(b)
        assert off_b == pytest.approx(
            {k: 4.0 * v for k, v in off_a.items()}, rel=1e-12, abs=1e-6
        )
        if mode == "open":
            roots = [r for r in b if not r.get("dependencies")]
            assert [r["not_before_ms"] for r in roots] == [copy["start_ms"]]


@needs_real
def test_osl_mult_scales_outputs_with_the_context_cap():
    common = dict(split="test", mode="closed", num_copies=40, seed=5)
    _, base_rows, _ = ax.generate(ax.GenSpec(**common), CR)
    _, rows, meta = ax.generate(ax.GenSpec(**common, osl_mult=2.5), CR)
    assert meta["transform"]["tag"] == "cap300_osl2.5"
    assert len(rows) == len(base_rows)
    for x, y in zip(base_rows, rows):
        assert y["request_id"] == x["request_id"]
        assert y["hash_ids"] == x["hash_ids"]
        assert y["not_before_ms"] == x["not_before_ms"]
        out = x["output_length"]
        want = (
            0
            if out == 0
            else min(max(1, round(out * 2.5)), MAX_MODEL_LEN - x["input_length"])
        )
        assert y["output_length"] == want


@needs_real
def test_unbuilt_transform_fails_loudly():
    spec = ax.GenSpec(split="val", mode="closed", num_copies=4, seed=0, think_mult=3.0)
    with pytest.raises(FileNotFoundError, match="cap300_think3"):
        ax.generate(spec, CR)
