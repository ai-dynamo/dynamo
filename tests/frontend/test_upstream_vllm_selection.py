# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that collection cannot silently expand the upstream GPU workload."""

import json
from types import SimpleNamespace

import pytest

from tests.utils.upstream_vllm.run import Selection

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


@pytest.mark.parametrize("actual", [["chosen", "extra"], [], ["other"]])
def test_reject_changed_collection(tmp_path, actual):
    """Reject added, missing or substituted nodes before server fixture setup."""
    session = SimpleNamespace(items=[SimpleNamespace(nodeid=node) for node in actual])
    with pytest.raises(RuntimeError, match="Unexpected upstream collection"):
        Selection(["chosen"], tmp_path).pytest_collection_finish(session)
    assert not (tmp_path / "collection.json").exists()


def test_record_original_fixture_dependencies(tmp_path):
    """Retain the fixture names for the exact node set without executing fixtures."""
    session = SimpleNamespace(
        items=[SimpleNamespace(nodeid="chosen", fixturenames=["server", "client"])]
    )
    Selection(["chosen"], tmp_path).pytest_collection_finish(session)
    assert json.loads((tmp_path / "collection.json").read_text()) == {
        "chosen": ["server", "client"]
    }
