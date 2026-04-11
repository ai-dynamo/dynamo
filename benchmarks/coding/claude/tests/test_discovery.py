# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from benchmarks.coding.claude.discovery import iter_ancestor_roots


def test_iter_ancestor_roots_walks_to_filesystem_root(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)

    roots = list(iter_ancestor_roots(nested))

    assert roots[0] == nested.resolve()
    assert roots[-1].parent == roots[-1]
    assert roots == list(dict.fromkeys(roots))
    assert nested.resolve().parent in roots
