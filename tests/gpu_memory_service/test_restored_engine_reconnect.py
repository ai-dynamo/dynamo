# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-engine consumer probe for an externally checkpointed GMS server.

The CUDA CustomStorage + CRIU controller owns the weights GMS process. It runs
this test twice:

* ``publish`` starts a normal vLLM engine, lets it publish weights to GMS,
  verifies inference, and records the committed allocation identity.
* ``verify`` starts a fresh read-only vLLM engine after GMS restore, verifies
  that it imports the same layout, and runs inference.

The controller remains responsible for checkpoint/restore ordering and cleanup;
this test deliberately does not implement Snapshot or Kubernetes orchestration.
"""

from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path

import pytest
from gpu_memory_service.server.fsm import ServerState

from tests.gpu_memory_service.common.gms import GMSServer, list_allocations
from tests.gpu_memory_service.common.runtime import VLLMWithGMSProcess
from tests.gpu_memory_service.flow_assertions import (
    assert_completion_ok,
    wait_for_weights_state,
)
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess

_EXTERNAL_SERVER_ENV = "DYN_GMS_EXTERNAL_SERVER"
_STAGE_ENV = "DYN_GMS_RESTORED_ENGINE_STAGE"
_IDENTITY_PATH_ENV = "DYN_GMS_RESTORED_IDENTITY_PATH"

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.fault_tolerance,
    pytest.mark.e2e,
    pytest.mark.gpu_1,
    pytest.mark.vllm,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.profiled_vram_gib(3.8),
    pytest.mark.requested_vllm_kv_cache_bytes(1_119_388_000),
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        os.environ.get(_EXTERNAL_SERVER_ENV) != "1",
        reason="requires an externally managed GMS checkpoint/restore controller",
    ),
]


def _allocation_identity(socket_path: str) -> list[dict[str, object]]:
    allocations = list_allocations(socket_path).allocations
    if not allocations:
        raise AssertionError("weights GMS has no committed allocations")
    return sorted(
        (
            {
                "allocation_id": allocation.allocation_id,
                "size": allocation.size,
                "aligned_size": allocation.aligned_size,
                "tag": allocation.tag,
                "layout_slot": allocation.layout_slot,
            }
            for allocation in allocations
        ),
        key=lambda allocation: (
            int(allocation["layout_slot"]),
            str(allocation["allocation_id"]),
        ),
    )


def _write_identity(
    identity_path: Path,
    *,
    memory_layout_hash: str,
    allocations: list[dict[str, object]],
) -> None:
    temporary_path = identity_path.with_suffix(identity_path.suffix + ".tmp")
    with temporary_path.open("x", encoding="utf-8") as output:
        json.dump(
            {
                "memory_layout_hash": memory_layout_hash,
                "allocations": allocations,
            },
            output,
            sort_keys=True,
        )
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, identity_path)


def _load_identity(identity_path: Path) -> dict[str, object]:
    with identity_path.open(encoding="utf-8") as input_file:
        identity = json.load(input_file)
    if not identity.get("memory_layout_hash") or not identity.get("allocations"):
        raise AssertionError(f"invalid pre-checkpoint identity: {identity!r}")
    return identity


def test_restored_gms_reconnects_fresh_vllm_engine(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
):
    stage = os.environ.get(_STAGE_ENV)
    if stage not in {"publish", "verify"}:
        raise ValueError(f"{_STAGE_ENV} must be 'publish' or 'verify', got {stage!r}")

    identity_value = os.environ.get(_IDENTITY_PATH_ENV)
    if not identity_value:
        raise ValueError(f"{_IDENTITY_PATH_ENV} is required")
    identity_path = Path(identity_value)

    weights_gms = GMSServer(device=0, tag="weights")
    expected_identity = _load_identity(identity_path) if stage == "verify" else None
    if expected_identity is not None:
        wait_for_weights_state(
            weights_gms,
            ServerState.COMMITTED,
            expected_hash=str(expected_identity["memory_layout_hash"]),
            timeout=30.0,
        )
        assert (
            _allocation_identity(weights_gms.socket_path)
            == expected_identity["allocations"]
        )

    with ExitStack() as stack:
        # vLLM uses a separate GMS server for its ephemeral KV cache. Only the
        # externally managed weights server is part of this checkpoint probe.
        kv_cache_gms = stack.enter_context(GMSServer(device=0, tag="kv_cache"))
        frontend = stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=0,
                display_name=f"restored-gms-{stage}-frontend",
            )
        )
        engine = stack.enter_context(
            VLLMWithGMSProcess(
                request,
                frontend.frontend_port,
                engine_id=f"restored-gms-{stage}-engine",
                read_only_weights=stage == "verify",
            )
        )

        assert_completion_ok(
            frontend.frontend_port,
            f"GMS {stage}",
            failure_message=f"vLLM inference failed during {stage}",
            success_message=f"vLLM inference passed during {stage}",
        )

        weights_state = wait_for_weights_state(
            weights_gms,
            ServerState.RO,
            min_ro_sessions=1 if stage == "verify" else 0,
            expected_hash=(
                str(expected_identity["memory_layout_hash"])
                if expected_identity is not None
                else None
            ),
            timeout=60.0,
        )
        allocations = _allocation_identity(weights_gms.socket_path)

        if expected_identity is None:
            _write_identity(
                identity_path,
                memory_layout_hash=weights_state.memory_layout_hash,
                allocations=allocations,
            )
        else:
            assert (
                weights_state.memory_layout_hash
                == expected_identity["memory_layout_hash"]
            )
            assert allocations == expected_identity["allocations"]

        assert kv_cache_gms.get_runtime_state().allocation_count > 0
        # Keep the engine alive through every GMS assertion so this proves the
        # imported tensors are usable, not merely that startup returned.
        assert engine.is_running()
