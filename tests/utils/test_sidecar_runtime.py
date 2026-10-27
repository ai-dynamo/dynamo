# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required sidecar lanes must exercise their runtime instead of skipping."""

from types import SimpleNamespace

import pytest

from tests.utils import sidecar_runtime

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize("backend", [None, "trtllm", "vllm"])
@pytest.mark.parametrize("available", [False, True])
def test_required_ci_fails_missing_runtime_and_local_runs_skip(
    monkeypatch, backend, available
):
    monkeypatch.delenv("DYNAMO_TEST_SIDECAR_BACKEND", raising=False)
    if backend is not None:
        monkeypatch.setenv("DYNAMO_TEST_SIDECAR_BACKEND", backend)
    monkeypatch.setattr(sidecar_runtime, "trtllm_serves_openengine", lambda: available)
    if available:
        sidecar_runtime.require_trtllm_openengine()
    else:
        error = pytest.fail.Exception if backend == "trtllm" else pytest.skip.Exception
        with pytest.raises(error, match="no OpenEngine servicer"):
            sidecar_runtime.require_trtllm_openengine()


@pytest.mark.parametrize("layout", ["absent", "old", "openengine"])
def test_probe_requires_servicer_file_without_importing_backend(
    monkeypatch, tmp_path, layout
):
    spec = (
        None
        if layout == "absent"
        else SimpleNamespace(submodule_search_locations=[str(tmp_path)])
    )
    if layout == "openengine":
        servicer = tmp_path / "grpc/openengine/servicer.py"
        servicer.parent.mkdir(parents=True)
        servicer.write_text("raise AssertionError('backend must not be imported')\n")
    monkeypatch.setattr(sidecar_runtime.importlib.util, "find_spec", lambda name: spec)
    assert sidecar_runtime.trtllm_serves_openengine() is (layout == "openengine")
