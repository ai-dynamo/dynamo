# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the Rust differential fixture from an exact vLLM source checkout.

Run with PYTHONPATH=components/src and --source-dir pointing to the checkout's
vllm directory (or flattened vllm_*.py files downloaded from the pinned commit).
No vLLM installation, torch, Mooncake process, or GPU is required. Whole selected
AST definitions execute unchanged, with postponed annotations and stdlib imports.
The actual coordinator, registry, specs, managers and external block pool execute;
only the network/object-residency boundary is replaced by static fixture data.
"""

import argparse
import ast
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, replace
from enum import Enum, EnumMeta, IntEnum
from functools import cached_property
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple, NewType, cast, overload

from dynamo.vllm.mooncake_store_runtime import _PINNED_FILES, VLLM_REVISION

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[5]
    / "lib/llm/src/kv_router/shared_cache/vllm_mooncake_store/oracle.json"
)
EXTRA_SOURCES = {
    "v1/kv_cache_spec_registry.py": "b4a1d9e686574a9f8927720e2bee687f3a0b61b91487f71e6f21032ef4cf229a",
    "utils/math_utils.py": "65f73de5632a6929e8240f1c79e2e0c1ef57ee29c44d5d6de2ad66b72f26bcb6",
    "v1/attention/backends/registry.py": "74375cf729d2504159285b8fd920e00ddd7af2778c806dbf3e78a9b74b96b731",
}
SELECTIONS = {
    "utils/math_utils.py": ("cdiv",),
    "utils/hashing.py": ("sha256", "xxhash", "xxhash_cbor"),
    "v1/attention/backends/registry.py": (
        "_AttentionBackendEnumMeta",
        "MambaAttentionBackendEnum",
    ),
    "v1/kv_cache_spec_registry.py": (
        "KVCacheSpecMetadata",
        "_REGISTRY_KVCACHESPEC_LIST",
        "_REGISTRY_ROLE_MANAGERS",
        "KVCacheSpecRegistry",
    ),
    "v1/kv_cache_interface.py": (
        "KVQuantMode",
        "KVCacheGroupRole",
        "KVCacheSpec",
        "AttentionSpec",
        "FullAttentionSpec",
        "ChunkedLocalAttentionSpec",
        "SlidingWindowSpec",
        "MambaSpec",
        "UniformTypeKVCacheSpecs",
        "KVCacheGroupSpec",
    ),
    "v1/core/kv_cache_utils.py": (
        "BlockHash",
        "BlockHashWithGroupId",
        "DEFAULT_NONE_HASH_SEED",
        "_NON_CRYPTO_HASH_FUNCTIONS",
        "_NONE_HASH_SEED",
        "resolve_none_hash_seed",
        "init_none_hash",
        "hash_block_tokens",
        "maybe_convert_block_hash",
        "KVCacheBlock",
        "BlockHashListWithBlockSize",
        "resolve_block_hashes",
    ),
    "v1/core/single_type_kv_cache_manager.py": (
        "SingleTypeKVCacheManager",
        "FullAttentionManager",
        "SlidingWindowManager",
        "MambaManager",
    ),
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py": (
        "_CompactChunkHashList",
        "chunk_hashes_for_block_size",
    ),
    "distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py": (
        "StoreSpecGroup",
        "ExternalCachedBlockPool",
        "MooncakeStoreCoordinator",
        "_unwrap_spec",
        "partial_hash_hits_enabled",
    ),
}


def load_oracle(source_dir):
    module = ModuleType("_pinned_mooncake_oracle")
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(
        ABC=ABC,
        abstractmethod=abstractmethod,
        defaultdict=defaultdict,
        Sequence=Sequence,
        dataclass=dataclass,
        field=field,
        fields=fields,
        replace=replace,
        Enum=Enum,
        EnumMeta=EnumMeta,
        IntEnum=IntEnum,
        cached_property=cached_property,
        NamedTuple=NamedTuple,
        NewType=NewType,
        cast=cast,
        overload=overload,
        itertools=itertools,
        os=os,
        hashlib=hashlib,
        pickle=pickle,
    )
    namespace["envs"] = SimpleNamespace(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True)
    namespace["logger"] = logging.getLogger(module.__name__)
    # BlockPool is used only in a typing.cast, never instantiated or called.
    namespace["BlockPool"] = object
    fingerprints = dict(_PINNED_FILES) | EXTRA_SOURCES
    provenance = {}
    for relative, names in SELECTIONS.items():
        path = source_dir / relative
        if not path.exists():
            path = source_dir / ("vllm_" + relative.replace("/", "_"))
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != fingerprints[relative]:
            raise ValueError(f"Unrecognized pinned source: {relative}")
        nodes = []
        selected = set()
        for node in ast.parse(content, filename=str(path)).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                node_names = {node.name}
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                node_names = {node.target.id}
            elif isinstance(node, ast.Assign):
                node_names = {n.id for n in node.targets if isinstance(n, ast.Name)}
            else:
                continue
            if node_names.intersection(names):
                nodes.append(node)
                selected.update(node_names.intersection(names))
        if selected != set(names):
            raise ValueError(
                f"Missing definitions in {relative}: {set(names) - selected}"
            )
        future = ast.parse("from __future__ import annotations").body
        tree = ast.Module(body=future + nodes, type_ignores=[])
        # Execution is limited to definitions in the fingerprint-verified sources.
        exec(
            compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace
        )  # noqa: S102
        provenance[relative] = {"sha256": digest, "definitions": list(names)}
    for spec, manager in (
        ("FullAttentionSpec", "FullAttentionManager"),
        ("SlidingWindowSpec", "SlidingWindowManager"),
        ("MambaSpec", "MambaManager"),
    ):
        namespace["KVCacheSpecRegistry"].register(namespace[spec], namespace[manager])
    return SimpleNamespace(**namespace), provenance


def make_case(oracle, name, shapes, resident_ends, *, count=65, tp=1, missing=None):
    hash_span = min(span for _, span, _ in shapes)
    alignment = math.lcm(*(span for _, span, _ in shapes))
    tokens = list(range(count))
    hashes = []
    parent = None
    for start in range(0, len(tokens) - hash_span + 1, hash_span):
        parent = oracle.hash_block_tokens(
            oracle.sha256, parent, tokens[start : start + hash_span]
        )
        hashes.append(parent)
    specs = []
    groups = []
    for gid, (kind, span, window) in enumerate(shapes):
        if kind == "mamba":
            spec = oracle.MambaSpec(
                block_size=span,
                shapes=((1,),),
                dtypes=(None,),
                mamba_cache_mode="align",
            )
        else:
            cls = (
                oracle.FullAttentionSpec
                if kind == "full_attention"
                else oracle.SlidingWindowSpec
            )
            extra = {} if window is None else {"sliding_window": window}
            spec = cls(
                block_size=span, num_kv_heads=2, head_size=8, dtype=None, **extra
            )
        specs.append(oracle.KVCacheGroupSpec([f"layer{gid}"], spec))
        groups.append(
            {
                "group_id": gid,
                "kind": kind,
                "block_size": span,
                "window": window,
                "prefixes": [
                    f"oracle@model@tp_rank:{rank}@group:{gid}" for rank in range(tp)
                ],
            }
        )
    coordinator = oracle.MooncakeStoreCoordinator(specs, alignment, hash_span)
    present = []
    for gid, ends in enumerate(resident_ends):
        for end in ends:
            digest = hashes[end // hash_span - 1].hex()
            for rank, prefix in enumerate(groups[gid]["prefixes"]):
                if missing == (gid, end, rank):
                    continue
                present.append([gid, prefix, digest])
    residency = {(gid, prefix, digest) for gid, prefix, digest in present}
    exists = {
        (gid, bytes.fromhex(digest))
        for gid, _, digest in present
        if all((gid, prefix, digest) in residency for prefix in groups[gid]["prefixes"])
    }
    pool = oracle.ExternalCachedBlockPool(hash_span, exists)

    def hit(request_length):
        return coordinator.find_longest_cache_hit(
            hashes[: request_length // hash_span],
            request_length - 1,
            pool,
        )[1]

    return {
        "name": name,
        "main_event_block_size": shapes[0][1],
        "coordinator_alignment": alignment,
        "partial_hash_hits": coordinator.enable_partial_hash_hits,
        "hash_block_size": hash_span,
        "groups": groups,
        "token_ids": tokens,
        "hashes": [
            {
                "end_token": (i + 1) * hash_span,
                "digest": h.hex(),
                "low64": oracle.maybe_convert_block_hash(h),
            }
            for i, h in enumerate(hashes)
        ],
        "present": present,
        "request_token_count": count,
        "oracle_hit_tokens": hit(count),
        "reusable_endpoints": [end for end in range(1, count) if hit(end + 1) == end],
    }


def generate(source_dir):
    os.environ["PYTHONHASHSEED"] = "0"
    oracle, provenance = load_oracle(source_dir)
    oracle.init_none_hash(oracle.sha256)
    fa16 = ("full_attention", 16, None)
    fa32 = ("full_attention", 32, None)
    swa16 = ("sliding_window", 16, 32)
    m16 = ("mamba", 16, None)
    m32 = ("mamba", 32, None)
    cases = [
        make_case(
            oracle,
            "fa16_swa16_window32_late",
            [fa16, swa16],
            [[16, 32, 48, 64], [48, 64]],
        ),
        make_case(
            oracle,
            "fa16_swa16_window32_final_token",
            [fa16, swa16],
            [[16, 32, 48, 64], [48, 64]],
            count=64,
        ),
        make_case(
            oracle, "fa16_mamba32_aligned", [fa16, m32], [[16, 32, 48, 64], [64]]
        ),
        make_case(
            oracle,
            "fa16_swa16_window32_gap",
            [fa16, swa16],
            [[16, 32, 48, 64], [16, 48, 64]],
        ),
        make_case(
            oracle, "tp2_fa16_mamba16_late", [fa16, m16], [[16, 32, 48, 64], [64]], tp=2
        ),
        make_case(
            oracle,
            "tp2_mamba_missing_member",
            [fa16, m16],
            [[16, 32, 48, 64], [64]],
            tp=2,
            missing=(1, 64, 1),
        ),
        make_case(
            oracle,
            "fa32_swa8_window8",
            [fa32, ("sliding_window", 8, 8)],
            [[32, 64], [64]],
        ),
        make_case(
            oracle, "fa16_mamba32_partial", [fa16, m32], [[16, 32, 48], [48]], count=49
        ),
        make_case(oracle, "fa32_mamba16", [fa32, m16], [[32, 64], [64]]),
        make_case(
            oracle, "fa32_mamba16_unaligned_state", [fa32, m16], [[32, 64], [48]]
        ),
        make_case(
            oracle,
            "final_token_65",
            [fa16, m16],
            [[16, 32, 48, 64], [48, 64]],
            count=65,
        ),
        make_case(
            oracle,
            "final_token_64",
            [fa16, m16],
            [[16, 32, 48, 64], [48, 64]],
            count=64,
        ),
    ]
    return {
        "_generated": "DO NOT EDIT. Regenerate with components/src/dynamo/vllm/tests/generate_mooncake_oracle.py; --check detects stale output.",
        "vllm_revision": VLLM_REVISION,
        "seed_policy": "pythonhashseed-0",
        "pickle_protocol": pickle.HIGHEST_PROTOCOL,
        "provenance": provenance,
        "execution": "Unchanged whole pinned AST definitions, postponed annotations, stdlib imports. Real KVCacheSpecRegistry registration, specs, managers, coordinator and ExternalCachedBlockPool. Static object-residency intersection at all required prefixes; no duplicated hit predicate. No torch tensors, network, installed vLLM, or GPU execution.",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(generate(args.source_dir), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.output.read_text() != rendered:
            raise SystemExit("Oracle fixture is stale")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
