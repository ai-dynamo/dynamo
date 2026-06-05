# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.router.router_process import SlotTrackerProcess
from tests.utils.port_utils import allocate_port, deallocate_port

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.parallel,
    pytest.mark.timeout(30),
]

SLOT_TRACKER_BASE_PORT = 8091


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    try:
        with urlopen(request, timeout=2) as response:
            response_body = response.read()
            return response.status, json.loads(response_body) if response_body else None
    except HTTPError as error:
        response_body = error.read()
        return error.code, json.loads(response_body) if response_body else None


def _post_json(
    base_url: str, path: str, body: dict[str, Any], expected_status: int
) -> Any:
    status, response_body = _request_json(base_url, path, method="POST", body=body)
    assert status == expected_status, response_body
    return response_body


def _assert_json_error(response_body: Any) -> None:
    assert isinstance(response_body, dict)
    assert isinstance(response_body.get("error"), str)


def _loads_by_rank(
    base_url: str, model_name: str
) -> dict[tuple[int, int], dict[str, Any]]:
    status, loads = _request_json(base_url, f"/loads?model_name={model_name}")
    assert status == 200, loads
    return {(load["worker_id"], load["dp_rank"]): load for load in loads}


@pytest.fixture
def slot_tracker_url(request: pytest.FixtureRequest) -> Iterator[str]:
    port = allocate_port(SLOT_TRACKER_BASE_PORT)
    request.addfinalizer(lambda: deallocate_port(port))
    with SlotTrackerProcess(request, port):
        yield f"http://127.0.0.1:{port}"


def test_shared_prefix_accounting_and_unregister(slot_tracker_url: str) -> None:
    base_url = slot_tracker_url
    for worker_id, dp_start, dp_size in [(7, 0, 2), (8, 0, 1)]:
        assert _post_json(
            base_url,
            "/register",
            {
                "worker_id": worker_id,
                "model_name": "llama-3-8b",
                "block_size": 16,
                "dp_start": dp_start,
                "dp_size": dp_size,
            },
            201,
        ) == {"status": "ok"}

    for request_id, worker_id, dp_rank, hashes in [
        ("req-a", 7, 0, [11, -22, 33]),
        ("req-b", 7, 0, [11, -22, 44]),
        ("req-c", 7, 1, [11, -22, 55]),
    ]:
        _post_json(
            base_url,
            "/add",
            {
                "model_name": "llama-3-8b",
                "request_id": request_id,
                "worker_id": worker_id,
                "dp_rank": dp_rank,
                "sequence_hashes": hashes,
            },
            201,
        )

    loads = _loads_by_rank(base_url, "llama-3-8b")
    assert loads[(7, 0)]["active_decode_blocks"] == 4
    assert loads[(7, 1)]["active_decode_blocks"] == 3
    assert loads[(8, 0)]["active_decode_blocks"] == 0

    potential = _post_json(
        base_url,
        "/potential_loads",
        {
            "model_name": "llama-3-8b",
            "sequence_hashes": [11, -22, 33, 66],
        },
        200,
    )
    potential_by_rank = {
        (load["worker_id"], load["dp_rank"]): load["potential_decode_blocks"]
        for load in potential
    }
    assert potential_by_rank == {(7, 0): 5, (7, 1): 5, (8, 0): 4}

    _post_json(
        base_url,
        "/free",
        {"model_name": "llama-3-8b", "request_id": "req-b"},
        200,
    )
    loads = _loads_by_rank(base_url, "llama-3-8b")
    assert loads[(7, 0)]["active_decode_blocks"] == 3

    _post_json(
        base_url,
        "/unregister",
        {"worker_id": 7, "model_name": "llama-3-8b"},
        200,
    )
    assert _loads_by_rank(base_url, "llama-3-8b") == {
        (8, 0): {
            "model_name": "llama-3-8b",
            "tenant_id": "default",
            "worker_id": 8,
            "dp_rank": 0,
            "active_prefill_tokens": 0,
            "active_decode_blocks": 0,
        }
    }
    stale_add = _post_json(
        base_url,
        "/add",
        {
            "model_name": "llama-3-8b",
            "request_id": "stale-selection",
            "worker_id": 7,
            "dp_rank": 0,
            "sequence_hashes": [],
        },
        404,
    )
    _assert_json_error(stale_add)


def test_prefill_lifecycle_updates_both_load_dimensions(slot_tracker_url: str) -> None:
    base_url = slot_tracker_url
    _post_json(
        base_url,
        "/register",
        {
            "worker_id": 4,
            "model_name": "llama-3-8b",
            "block_size": 16,
            "dp_start": 0,
            "dp_size": 1,
        },
        201,
    )
    for request_id, hashes, new_isl_tokens in [
        ("req-a", [1, 2], 48),
        ("req-b", [1, 3], 16),
    ]:
        _post_json(
            base_url,
            "/add",
            {
                "model_name": "llama-3-8b",
                "request_id": request_id,
                "worker_id": 4,
                "dp_rank": 0,
                "sequence_hashes": hashes,
                "new_isl_tokens": new_isl_tokens,
            },
            201,
        )

    assert _loads_by_rank(base_url, "llama-3-8b")[(4, 0)] == {
        "model_name": "llama-3-8b",
        "tenant_id": "default",
        "worker_id": 4,
        "dp_rank": 0,
        "active_prefill_tokens": 64,
        "active_decode_blocks": 3,
    }

    for _ in range(2):
        _post_json(
            base_url,
            "/prefill_complete",
            {"model_name": "llama-3-8b", "request_id": "req-a"},
            200,
        )
    load = _loads_by_rank(base_url, "llama-3-8b")[(4, 0)]
    assert load["active_prefill_tokens"] == 16
    assert load["active_decode_blocks"] == 3

    _post_json(
        base_url,
        "/free",
        {"model_name": "llama-3-8b", "request_id": "req-b"},
        200,
    )
    load = _loads_by_rank(base_url, "llama-3-8b")[(4, 0)]
    assert load["active_prefill_tokens"] == 0
    assert load["active_decode_blocks"] == 2

    for _ in range(2):
        _post_json(
            base_url,
            "/free",
            {"model_name": "llama-3-8b", "request_id": "req-a"},
            200,
        )
    assert _loads_by_rank(base_url, "llama-3-8b")[(4, 0)]["active_decode_blocks"] == 0


def test_unregister_then_register_starts_fresh(slot_tracker_url: str) -> None:
    base_url = slot_tracker_url
    registration = {
        "worker_id": 9,
        "model_name": "llama-3-8b",
        "block_size": 16,
        "dp_start": 0,
        "dp_size": 2,
    }
    _post_json(base_url, "/register", registration, 201)
    _post_json(
        base_url,
        "/add",
        {
            "model_name": "llama-3-8b",
            "request_id": "reused-request-id",
            "worker_id": 9,
            "dp_rank": 0,
            "sequence_hashes": [101, 102],
        },
        201,
    )
    _post_json(
        base_url,
        "/unregister",
        {"worker_id": 9, "model_name": "llama-3-8b"},
        200,
    )

    assert _loads_by_rank(base_url, "llama-3-8b") == {}
    missing_tracker = _post_json(
        base_url,
        "/potential_loads",
        {"model_name": "llama-3-8b", "sequence_hashes": [101]},
        404,
    )
    _assert_json_error(missing_tracker)

    registration.update({"block_size": 32, "dp_start": 2})
    _post_json(base_url, "/register", registration, 201)
    _post_json(
        base_url,
        "/add",
        {
            "model_name": "llama-3-8b",
            "request_id": "reused-request-id",
            "worker_id": 9,
            "dp_rank": 2,
            "sequence_hashes": [201],
        },
        201,
    )
    loads = _loads_by_rank(base_url, "llama-3-8b")
    assert loads[(9, 2)]["active_decode_blocks"] == 1
    assert loads[(9, 3)]["active_decode_blocks"] == 0
