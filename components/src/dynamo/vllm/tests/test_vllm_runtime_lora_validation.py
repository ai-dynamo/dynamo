# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from dynamo.llm import HttpError
from dynamo.vllm.lora_state import LoRAState
from dynamo.vllm.runtime_lora_validation import cache_reservation, validate_snapshot

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _safetensors_blob() -> bytes:
    header = json.dumps(
        {
            "base_model.model.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [1, 2],
                "data_offsets": [0, 8],
            }
        }
    ).encode()
    return len(header).to_bytes(8, "little") + header + bytes(8)


def _snapshot(tmp_path):
    snapshot = tmp_path / "adapter"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 8,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": "base",
            }
        )
    )
    (snapshot / "adapter_model.safetensors").write_bytes(_safetensors_blob())
    return snapshot


@pytest.mark.parametrize("relative_path", ["README.md", "nested/metadata.json"])
def test_snapshot_rejects_files_outside_vllm_allowlist(tmp_path, relative_path):
    snapshot = _snapshot(tmp_path)
    unexpected = snapshot / relative_path
    unexpected.parent.mkdir(parents=True, exist_ok=True)
    unexpected.write_text("untrusted")

    with pytest.raises(HttpError) as error:
        validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",))
    assert error.value.code == 422
    assert error.value.message == "invalid_lora_adapter"


def test_snapshot_accepts_valid_safetensors_embeddings(tmp_path):
    snapshot = _snapshot(tmp_path)
    (snapshot / "new_embeddings.safetensors").write_bytes(_safetensors_blob())

    result = validate_snapshot(snapshot, tmp_path, 1024, 64, ("base",))

    assert result == snapshot


@pytest.mark.asyncio
async def test_cache_reservation_release_survives_repeated_cancellation(tmp_path):
    state = LoRAState()
    settings = SimpleNamespace(max_cache_bytes=1024, max_download_bytes=256)
    entered = asyncio.Event()

    async def hold_reservation():
        async with cache_reservation(state, tmp_path, settings):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_reservation())
    await asyncio.wait_for(entered.wait(), timeout=1)
    guard = state.runtime_cache_guard
    assert guard is not None
    await guard.acquire()

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    guard.release()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state.runtime_cache_reserved_bytes == 0
