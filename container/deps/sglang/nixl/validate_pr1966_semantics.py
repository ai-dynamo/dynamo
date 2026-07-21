#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed semantic checks for the NIXL 1.3 PR #1966 backport."""

from __future__ import annotations

import re
import sys
from pathlib import Path

BATCH_SIZE = 16


def fail(message: str) -> None:
    raise SystemExit(f"nixl-pr1966-semantics: {message}")


def normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def rail_selection_index(
    base_offset: int, descriptor: int, batch_write: bool, count: int
) -> int:
    if count <= 0:
        fail("selection count must be positive")
    rr_index = base_offset + (descriptor // BATCH_SIZE if batch_write else descriptor)
    return rr_index % count


def fi_more_flags(local_rails: list[list[int] | None], striped: set[int]) -> list[bool]:
    """Model the backport's reverse scan; True means post with FI_MORE."""

    flags = [False] * len(local_rails)
    rail_count = max((max(rails) for rails in local_rails if rails), default=-1) + 1
    posts_since_flush = [-1] * rail_count
    for descriptor in range(len(local_rails) - 1, -1, -1):
        rails = local_rails[descriptor]
        if not rails or descriptor in striped:
            continue
        rail = rails[rail_selection_index(0, descriptor, True, len(rails))]
        seen = posts_since_flush[rail]
        if seen < 0 or seen >= BATCH_SIZE - 1:
            posts_since_flush[rail] = 0
        else:
            flags[descriptor] = True
            posts_since_flush[rail] += 1
    return flags


def assert_fi_more_invariants(
    local_rails: list[list[int] | None], striped: set[int]
) -> None:
    flags = fi_more_flags(local_rails, striped)
    by_rail: dict[int, list[int]] = {}
    for descriptor, rails in enumerate(local_rails):
        if not rails or descriptor in striped:
            if flags[descriptor]:
                fail("invalid or striped descriptor was marked FI_MORE")
            continue
        rail = rails[rail_selection_index(0, descriptor, True, len(rails))]
        by_rail.setdefault(rail, []).append(descriptor)
    for rail, descriptors in by_rail.items():
        if flags[descriptors[-1]]:
            fail(f"rail {rail}'s last post was left with FI_MORE")
        consecutive = 0
        for descriptor in descriptors:
            consecutive = consecutive + 1 if flags[descriptor] else 0
            if consecutive >= BATCH_SIZE:
                fail(f"rail {rail} exceeded {BATCH_SIZE - 1} deferred posts")


def main() -> None:
    if len(sys.argv) != 2:
        fail(f"usage: {Path(sys.argv[0]).name} PATCHED_NIXL_SOURCE")
    source = Path(sys.argv[1])
    rail_cpp = normalized(source / "src/utils/libfabric/libfabric_rail_manager.cpp")
    backend_cpp = normalized(source / "src/plugins/libfabric/libfabric_backend.cpp")
    rail_header = normalized(source / "src/utils/libfabric/libfabric_rail_manager.h")

    required_fragments = (
        "railSelectionIndex(base_offset, desc_idx, batch_write, selected_rails.size())",
        "remote_selected_endpoints[railSelectionIndex( base_offset, desc_idx, batch_write, remote_selected_endpoints.size())]",
        "const uint64_t fi_flags = (batch_write && apply_fi_more) ? FI_MORE : 0",
        "railSelectionIndex( xfer_base_offset, i, /*batch_write=*/true, md->selected_rails_.size())",
        "std::vector<int> posts_since_flush(rail_manager.getNumRails(), -1)",
    )
    combined = " ".join((rail_cpp, backend_cpp, rail_header))
    for fragment in required_fragments:
        if fragment not in combined:
            fail(f"required source invariant is missing: {fragment}")

    # Regression in the first PR head: one local rail accidentally pinned all
    # transfers to remote endpoint zero. Independent counts must select endpoint
    # zero for descriptors 0-15 and endpoint one for descriptors 16-31.
    local = [rail_selection_index(0, descriptor, True, 1) for descriptor in range(32)]
    remote = [rail_selection_index(0, descriptor, True, 2) for descriptor in range(32)]
    if local != [0] * 32 or remote != [0] * 16 + [1] * 16:
        fail("asymmetric 1-local/2-remote endpoint selection regressed")

    # Exercise exact and partial batches, two rails, per-descriptor rail lists,
    # empty metadata, and a striped descriptor. Each touched rail must flush and
    # may have no more than 15 consecutive FI_MORE posts.
    for count in (1, 2, 15, 16, 17, 31, 32, 33, 65):
        assert_fi_more_invariants([[0] for _ in range(count)], set())
    assert_fi_more_invariants([[0, 1] for _ in range(65)], set())
    assert_fi_more_invariants(
        [[0], [1], [0, 1], None] * 17,
        {2, 18, 34, 50, 66},
    )

    print("nixl-pr1966-semantics: endpoint selection and FI_MORE invariants passed")


if __name__ == "__main__":
    main()
