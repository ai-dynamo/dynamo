# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The only module in this package that imports pytest.

Everything else stays importable by a CLI, a planner, or a bundle reader that is
not running under pytest. Keeping that boundary in one file is what makes
``plan`` and ``judge`` work in a bare checkout.

## What this plugin is for

Recording **exactly which tests a given ``-m`` expression selects**, so a change
to markers or to collection can be diffed instead of trusted. That diff is the
gate that makes every later migration step safe: a refactor that silently stops
selecting 40 GPU tests looks identical to a green run otherwise.

## Why it snapshots first and subtracts deselections

The obvious implementation — observe the item list after everything has had its
say, with ``trylast`` — does not work on this repository, for two reasons that
are both measurable:

* ``--collect-only`` is **structurally blind**. ``tests/conftest.py:693`` guards
  *both* the ``--max-vram-gib`` deselection *and* ``write_test_meta`` on
  ``not config.option.collectonly``, so a collect-only run cannot observe the one
  mechanism that removes tests.
* ``--dry-run`` does see the deselection, but ``tests/conftest.py:761`` calls
  ``items.clear()`` before returning. A ``trylast`` observer under ``--dry-run``
  sees an empty list and reports that nothing is selected.

So: snapshot the full set **first** (``tryfirst``), record every deselection as
it happens through the ``pytest_deselected`` hook, and take the difference. That
is correct under both flags and does not depend on where any other hook sits in
the ordering.

The ordering detail worth knowing, because it is easy to model wrongly: the
repository's own hook is declared ``@pytest.hookimpl(trylast=True)``
(``tests/conftest.py:660``), which is what puts pytest's ``-m`` deselection
*before* it. A test fixture that omits that decorator gets the opposite order —
its conftest clears the item list before ``-m`` is ever applied — and then
asserts the gate works on a repository that does not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

__all__ = ["SelectionRecorder", "pytest_addoption", "pytest_configure"]

ENV_OUT = "DYNAMO_TEST_SELECTION_OUT"


class SelectionRecorder:
    """Records the tests a run would actually execute, and their markers."""

    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.collected: list[dict] = []
        self.deselected: set[str] = set()

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        # tryfirst: capture everything before any other hook can remove, clear,
        # or reorder it. Subtraction happens at the end, from pytest_deselected.
        for item in items:
            self.collected.append(
                {
                    "nodeid": item.nodeid,
                    "markers": sorted({m.name for m in item.iter_markers()}),
                    "marker_args": {
                        m.name: [repr(a) for a in m.args]
                        for m in item.iter_markers()
                        if m.args
                    },
                }
            )

    def pytest_deselected(self, items):
        for item in items:
            self.deselected.add(item.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        selected = [c for c in self.collected if c["nodeid"] not in self.deselected]
        payload = {
            "collected": len(self.collected),
            "deselected": len(self.deselected),
            "selected": len(selected),
            "tests": selected,
            "deselected_ids": sorted(self.deselected),
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def pytest_addoption(parser):
    parser.addoption(
        "--dynamo-selection-out",
        default=None,
        help=(
            "Write the set of tests this run would execute, with their markers, "
            "to this JSON file. Used by the marker-parity gate."
        ),
    )


def pytest_configure(config):
    out = config.getoption("--dynamo-selection-out", default=None) or os.environ.get(
        ENV_OUT
    )
    if not out:
        return
    config.pluginmanager.register(SelectionRecorder(Path(out)), "dynamo-selection")
